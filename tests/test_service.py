"""Synthetic fixtures prove plumbing only; no robot/hardware evidence."""
from dataclasses import replace
import hashlib
import json
import math
import threading

import pytest

from frc_world_model.analytical import (AnalyticalDynamics, DynamicsConfig,
    EgoMechanismState, PhysicalCommand, Rollout, constant_velocity, persistence)
from frc_world_model.service import (CandidateRollout, FrozenPackage, PredictionReply,
    PredictionRequest, PredictionService, RequestIdentity, ServiceLimits, StateSample)

T = 1_000_000_000
STATE = EgoMechanismState(0, 0, 0, 0, 0, 0, 0, 0)
IDENTITY = RequestIdentity("internal-test-v1", "robot-test", "map-test", "field-xy",
    "cal-test", "state-test", "source-epoch-1", "session-test", "reset-1", "replay-ns")


def make_request(service, request_id="request-1", count=2, horizon=4):
    candidates = tuple(CandidateRollout("candidate-%d" % candidate,
        tuple(PhysicalCommand(T + step * 20_000_000, candidate + 1, 0, 0, 0.5)
              for step in range(horizon))) for candidate in range(count))
    return PredictionRequest(service.identity, "snapshot-1", request_id, service.model_id,
        T, T, T, T, T + 400_000_000, True, STATE, candidates, horizon)




def blocked_predictor(started, release):
    def predict(request):
        started.set()
        if not release.wait(3):
            raise RuntimeError("test release timeout")
        baseline = AnalyticalDynamics()
        return {c.candidate_id: replace(baseline.rollout(request.initial_state, c.commands),
                                       kind="learned-forecast") for c in request.candidates}
    return predict


def test_analytical_closed_loop_response_clipping_and_robot_frame():
    baseline = AnalyticalDynamics()
    command = PhysicalCommand(T, 100, 0, 0, 100)
    state, saturated = baseline.step(STATE, command)
    assert saturated == (True, False, False, True)
    assert state.vx_robot_mps == pytest.approx(4 * (1 - math.exp(-0.02 / 0.1)))
    assert state.mechanism_position == pytest.approx(1 - math.exp(-0.02 / 0.2))
    assert state.x_m > 0 and state.y_m == 0
    rotated, _ = baseline.step(replace(STATE, heading_rad=math.pi / 2),
                               PhysicalCommand(T, 1, 0, 0, 0))
    assert abs(rotated.x_m) < 1e-10 and rotated.y_m > 0


@pytest.fixture
def service():
    instance = PredictionService(IDENTITY, clock=lambda: T)
    yield instance
    instance.close()


def test_physical_baseline_clipping_response_and_units():
    baseline = AnalyticalDynamics(DynamicsConfig(dt_s=0.02, chassis_tau_s=0.10))
    next_state, clipped = baseline.step(STATE, PhysicalCommand(T, 20, -20, 20, 20))
    assert clipped == (True, True, True, True)
    assert next_state.vx_robot_mps == pytest.approx(4 * (1 - math.exp(-0.2)))
    assert next_state.vy_robot_mps == -next_state.vx_robot_mps
    assert next_state.mechanism_position == pytest.approx(1 - math.exp(-0.1))
    assert next_state.mechanism_velocity == pytest.approx(next_state.mechanism_position / 0.02)
    state_at_90 = replace(STATE, heading_rad=math.pi / 2)
    rotated, _ = baseline.step(state_at_90, PhysicalCommand(T, 1, 0, 0, 0))
    assert abs(rotated.x_m) < 1e-10 and rotated.y_m > 0


def test_angle_wrap_hold_gap_and_baseline_variants():
    baseline = AnalyticalDynamics()
    state = replace(STATE, heading_rad=math.pi - 0.01, omega_radps=8, vx_robot_mps=1,
                    mechanism_position=0.3, mechanism_velocity=0.4)
    result = baseline.rollout(state, (PhysicalCommand(T, 0, 0, 8, -1, "hold"),))
    assert -math.pi <= result.states[0].heading_rad < math.pi
    assert result.states[0].mechanism_position == 0.3
    assert result.times_ns == (T + 20_000_000,)
    assert result.measured is False
    assert persistence(state, 2) == (state, state)
    assert constant_velocity(state, 2, 0.02)[-1].x_m != state.x_m
    with pytest.raises(ValueError, match="gap"):
        baseline.rollout(STATE, (PhysicalCommand(T, 0, 0, 0, 0),
                                PhysicalCommand(T + 1, 0, 0, 0, 0)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_is_never_unknown_zero(value):
    with pytest.raises(ValueError):
        replace(STATE, x_m=value)
    with pytest.raises(ValueError):
        PhysicalCommand(T, value, 0, 0, 0)


def test_absent_prediction_preserves_every_candidate(service):
    request = make_request(service)
    reply = service.predict(request)
    assert reply.status == "baseline"
    assert tuple(f.candidate_id for f in reply.forecasts) == ("candidate-0", "candidate-1")
    assert reply.forecasts[1].rollout.states[-1].x_m > reply.forecasts[0].rollout.states[-1].x_m
    assert all(f.fallback_reason == "learned-prediction-unavailable" for f in reply.forecasts)
    assert not reply.measured and not reply.influences_commands
    assert service.reply_matches(reply, request, T)


def test_serialization_round_trip_and_nonfinite_rejection(service):
    request = make_request(service)
    assert PredictionRequest.from_json(request.to_json()) == request
    reply = service.predict(request)
    assert PredictionReply.from_json(reply.to_json()) == reply
    assert service.reply_matches(PredictionReply.from_json(reply.to_json()), request, T)
    raw = json.loads(reply.to_json())
    raw["measured"] = True
    with pytest.raises(ValueError):
        PredictionReply.from_json(json.dumps(raw))
    raw = json.loads(request.to_json())
    raw["initial_state"]["x_m"] = float("nan")
    with pytest.raises(ValueError):
        PredictionRequest.from_json(json.dumps(raw))


@pytest.mark.parametrize("changes,reason", [
    ({"sync_valid": False}, "clock-sync-invalid"),
    ({"sync_id": ""}, "clock-sync-invalid"),
    ({"expiry_ns": T}, "expired"),
    ({"capture_time_ns": T + 1}, "noncausal-time-alignment"),
    ({"state_time_ns": T - 1}, "noncausal-time-alignment"),
    ({"issued_ns": T + 1}, "noncausal-time-alignment"),
    ({"state_valid": (False,) * 8}, "missing-state-prerequisite"),
    ({"horizon_steps": 101}, "horizon-limit"),
    ({"layout_id": "wrong"}, "schema-or-layout-mismatch"),
    ({"model_id": "wrong"}, "model-mismatch"),
    ({"mode": "live"}, "live-promotion-requires-separate-approval-and-evidence"),
    ({"history": (StateSample(T + 1, STATE),)}, "history-order-or-cutoff"),
    ({"history": (StateSample(T, STATE, provenance="future-smoothed"),)},
      "future-smoothed-or-unknown-input-provenance"),
])
def test_prerequisite_failures(service, changes, reason):
    reply = service.predict(replace(make_request(service), **changes))
    assert reply.status == "rejected" and reply.rejection_reason == reason
    assert reply.forecasts == ()


@pytest.mark.parametrize("field", ["schema_version", "robot_id", "map_id", "frame_id",
    "calibration_id", "source_id", "source_epoch", "session_id", "reset_epoch",
    "clock_id", "service_epoch"])
def test_identity_and_epoch_failures(service, field):
    request = make_request(service)
    reply = service.predict(replace(request, identity=replace(request.identity, **{field: "other"})))
    assert reply.rejection_reason == "identity-or-epoch-mismatch"


def test_duplicate_command_gap_stale_and_clock_rollback(service):
    request = make_request(service)
    first = service.predict(request)
    assert service.predict(request).rejection_reason == "duplicate-request"
    bad_command = replace(request.candidates[0].commands[0], timestamp_ns=T - 1)
    bad_candidate = replace(request.candidates[0], commands=(bad_command,) + request.candidates[0].commands[1:])
    assert service.predict(replace(request, request_id="gap", candidates=(bad_candidate,))).rejection_reason == "command-time-gap-or-order"
    assert not service.reply_matches(first, request, request.expiry_ns)
    service.clock = lambda: T - 1
    assert service.predict(replace(request, request_id="rollback")).rejection_reason == "clock-rollback"
    service.clock = lambda: T + 600_000_000
    stale = replace(request, request_id="stale", expiry_ns=T + 1_000_000_000)
    assert service.predict(stale).rejection_reason == "stale-state"


def test_crash_and_invalid_output_preserve_baseline():
    def crashing(_):
        raise RuntimeError("private-input-not-disclosed")
    for predictor, reason in [(crashing, "predictor-failure:RuntimeError"),
                              (lambda _: {}, "learned-candidate-mismatch"),
                              (lambda _: {"candidate-0": object()}, "learned-shape-time-or-kind-mismatch")]:
        service = PredictionService(IDENTITY, predictor=predictor, clock=lambda: T)
        try:
            reply = service.predict(make_request(service, count=1))
            assert reply.status == "baseline"
            assert reply.forecasts[0].fallback_reason == reason
        finally:
            service.close()


def test_native_learned_result_and_wrong_candidate_response(service):
    service.predictor = lambda req: {c.candidate_id: replace(
        service.baseline.rollout(req.initial_state, c.commands), kind="learned-forecast")
        for c in req.candidates}
    request = make_request(service)
    reply = service.predict(request)
    assert reply.status == "forecast" and all(f.predictor == "learned" for f in reply.forecasts)
    forged = replace(reply, forecasts=tuple(reversed(reply.forecasts)))
    assert not service.reply_matches(forged, request, T)


def blocked_predictor(started, release):
    def predict(_):
        started.set()
        assert release.wait(2), "test release failed"
        return {}
    return predict


def test_cancellation_overload_and_timeout_preserve_baseline():
    started, release = threading.Event(), threading.Event()
    service = PredictionService(IDENTITY, predictor=blocked_predictor(started, release),
        limits=ServiceLimits(max_pending=1), clock=lambda: T)
    try:
        req = make_request(service)
        future = service.submit(req)
        assert started.wait(1)
        overloaded = service.predict(make_request(service, "overload"))
        assert overloaded.forecasts[0].fallback_reason == "overload-optional-work-shed"
        service.cancel(req.request_id)
        release.set()
        assert future.result(1).forecasts[0].fallback_reason == "cancelled"
    finally:
        release.set()
        service.close()
    started, release = threading.Event(), threading.Event()
    service = PredictionService(IDENTITY, predictor=blocked_predictor(started, release), clock=lambda: T)
    try:
        timed = service.predict(make_request(service), timeout_s=0.01)
        assert timed.forecasts[0].fallback_reason == "optional-inference-timeout"
    finally:
        release.set()
        service.close()


def test_late_reply_cannot_cross_reset_restart_or_network_loss(service):
    old_request = make_request(service)
    old_reply = service.predict(old_request)
    service.reset(replace(service.identity, reset_epoch="reset-2"))
    assert not service.reply_matches(old_reply, old_request, T)
    assert service.predict(replace(old_request, request_id="old-again")).status == "rejected"
    replacement = PredictionService(IDENTITY, clock=lambda: T)
    try:
        assert replacement.identity.service_epoch != old_request.identity.service_epoch
        assert not replacement.reply_matches(old_reply, old_request, T)
        # Transport has no reply: independently available baseline remains computable.
        direct = replacement.baseline.rollout(old_request.initial_state, old_request.candidates[0].commands)
        assert direct == old_reply.forecasts[0].rollout
    finally:
        replacement.close()


def test_inflight_epoch_reset_rejects_delayed_result():
    started, release = threading.Event(), threading.Event()
    service = PredictionService(IDENTITY, predictor=blocked_predictor(started, release), clock=lambda: T)
    try:
        future = service.submit(make_request(service))
        assert started.wait(1)
        service.reset(replace(service.identity, reset_epoch="next"))
        release.set()
        reply = future.result(1)
        assert reply.status == "rejected" and reply.rejection_reason == "late-reply-across-epoch"
    finally:
        release.set()
        service.close()


def package_fixture(root, dimension=6):
    files = {"weights": ("weights.pt", b"synthetic-fixture-only"),
        "schema": ("schema.json", {"version": "test-v1", "layout_id": "test-layout"}),
        "normalization": ("normalization.json", {"fit_split": "train", "dataset_sha256": "fixture-hash",
            "layout": ["feature-%d" % i for i in range(dimension)], "mean": [0] * dimension,
            "scale": [1] * dimension}),
        "calibration": ("calibration.json", {"status": "uncalibrated"}),
        "config": ("config.json", {"state_dim": dimension})}
    artifacts = {}
    for name, (path, content) in files.items():
        data = content if isinstance(content, bytes) else json.dumps(content).encode()
        (root / path).write_bytes(data)
        artifacts[name] = {"path": path, "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"format": "frc-world-model-frozen-v1", "model_id": "fixture", "schema_version": "test-v1",
        "layout_id": "test-layout", "model_kind": "analytical-fixture", "artifacts": artifacts}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_frozen_package_dimensions_checksums_and_provenance(tmp_path):
    manifest = package_fixture(tmp_path, dimension=6)
    package = FrozenPackage.load(tmp_path)
    assert len(package.metadata["normalization"]["mean"]) == 6
    assert len(package.checksum) == 64
    (tmp_path / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        FrozenPackage.load(tmp_path)


@pytest.mark.parametrize("mutation", ["escape", "symlink", "normalization", "dimension", "schema"])
def test_frozen_package_rejects_invalid_artifacts(tmp_path, mutation):
    manifest = package_fixture(tmp_path)
    if mutation == "escape":
        manifest["artifacts"]["weights"]["path"] = "../other.pt"
    elif mutation == "symlink":
        (tmp_path / "weights.pt").unlink()
        (tmp_path / "other.pt").write_bytes(b"synthetic-fixture-only")
        (tmp_path / "weights.pt").symlink_to(tmp_path / "other.pt")
    else:
        name = {"normalization": "normalization", "dimension": "config", "schema": "schema"}[mutation]
        file = tmp_path / manifest["artifacts"][name]["path"]
        content = json.loads(file.read_text())
        content.update({"normalization": {"fit_split": "test"}, "dimension": {"state_dim": 8},
                        "schema": {"layout_id": "other"}}[mutation])
        file.write_text(json.dumps(content))
        manifest["artifacts"][name]["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        FrozenPackage.load(tmp_path)
