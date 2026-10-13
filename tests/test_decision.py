"""End-to-end advisory tests; all physical policy/model evidence is synthetic."""
import copy
from concurrent.futures import CancelledError
import json
from pathlib import Path
import threading
import time

import pytest

from frc_world_model.bindings import ContractError, load_document, make_masked_reply
from frc_world_model.decision_bindings import (
    action_sequence_hash, authority_for, consume_decision_reply, load_decision,
    request_hash, validate_decision_request,
)
from frc_world_model.decision_schema import CONTRACT
from frc_world_model.decision_service import (
    DecisionService, LatestDecisionWorker, LearnedEvidence, safe_decision_roundtrip,
)
from frc_world_model.transport import loopback_server
from tools.benchmark_decisions import benchmark, fixture_policy, fixture_request, synthetic_learned_evidence


def evaluate(raw=None, *, predictor=None, policy=None):
    request = copy.deepcopy(raw if raw is not None else fixture_request())
    policy = policy or fixture_policy()
    now = request["prediction_request"]["issued_us"]
    service = DecisionService(policy, authority=authority_for(request), now_us=lambda: now, predictor=predictor)
    request["service_epoch"] = service.service_epoch
    original = copy.deepcopy(request)
    try:
        reply = service.evaluate(json.dumps(request, allow_nan=False))
        assert request == original, "a prediction must not mutate observed facts or candidate inputs"
        return request, reply
    finally:
        service.close()


def ranking(reply, task_id):
    return next(value for value in reply["rankings"] if value["task_id"] == task_id)


def refresh(request):
    request["action_sequence_hash"] = action_sequence_hash(request["prediction_request"])
    return request


def test_public_packaged_schema_fixture_and_unchanged_prediction_contract():
    public = Path(__file__).resolve().parents[1] / "contracts" / "decision-support-v1"
    for name in ("request-v1.json", "reply-v1.json", "decision-request-v1.schema.json", "decision-reply-v1.schema.json"):
        assert (public / name).read_bytes() == (CONTRACT / name).read_bytes()
    request = fixture_request()
    assert load_decision(json.dumps(request), "request") == request
    assert validate_decision_request(request, fixture_policy(), request["prediction_request"]["issued_us"]) == request
    embedded = load_document(json.dumps(request["prediction_request"]), "request")
    assert embedded["schema_version"] == "frc-prediction/1"
    assert request["action_sequence_hash"] == action_sequence_hash(embedded)


def test_changed_measured_context_requires_new_revision_without_disrupting_service():
    request = fixture_request()
    service = DecisionService(fixture_policy(), authority=authority_for(request), now_us=lambda: 1_010_000)
    request["service_epoch"] = service.service_epoch
    try:
        assert service.evaluate(json.dumps(request))["status"] == "BASELINE"
        changed = copy.deepcopy(request)
        changed["decision_generation"] += 1
        changed["context"]["possession"] = "HELD"
        rejected = service.evaluate(json.dumps(changed))
        assert rejected["status"] == "REJECTED"
        assert rejected["reason"] == "context-changed-without-revision"
        changed["context"]["revision"] += 1
        accepted = service.evaluate(json.dumps(changed))
        assert accepted["status"] == "BASELINE"
        assert accepted["selected_task_id"] == "score-zone"
    finally:
        service.close()


def test_unicode_action_hash_uses_documented_ascii_escaped_canonical_json():
    request = fixture_request()["prediction_request"]
    request["candidates"][0]["candidate_id"] = "hold-\u03bb"
    expected = __import__("hashlib").sha256(json.dumps(request["candidates"], allow_nan=False,
        sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    assert action_sequence_hash(request) == expected


def test_reply_serialization_must_complete_inside_host_deadline(monkeypatch):
    import frc_world_model.decision_service as module
    original = module.load_decision
    entered = threading.Event()
    def delayed_reply(payload, kind="request", **kwargs):
        if kind == "reply":
            entered.set()
            time.sleep(.11)
        return original(payload, kind, **kwargs)
    monkeypatch.setattr(module, "load_decision", delayed_reply)
    _, reply = evaluate(policy=fixture_policy(computation_budget_s=.1))
    assert entered.is_set()
    assert reply["status"] == "REJECTED"
    assert reply["reason"] == "decision-expired-or-budget-exhausted"
    assert reply["selected_task_id"] is None


def test_baseline_is_adaptive_advisory_with_no_fabricated_learned_outcomes():
    request, reply = evaluate()
    assert reply["status"] == "BASELINE"
    assert reply["selected_task_id"] == "pickup-piece-100"
    assert reply["baseline_selected_task_id"] == reply["selected_task_id"]
    assert reply["measured"] is False and reply["influences_commands"] is False
    assert reply["requires_planner_validation"] is True
    assert ranking(reply, "pickup-piece-100")["requires_path_planning"] is True
    for item in reply["rankings"]:
        assert item["forecast_kind"] == "KINEMATIC_ASSUMPTION"
        assert item["outcome"]["task_success_probability"] is None
        assert item["outcome"]["task_completion_time_s"] is None
        assert item["outcome"]["mechanism_outcome"] is None
        assert all(sample["physical_covariance"] is None for sample in item["outcome"]["samples"])
    assert consume_decision_reply(reply, request, policy=fixture_policy(),
                                  now_us=request["prediction_request"]["issued_us"], current_request=request) == reply


@pytest.mark.parametrize("possession,ready,selected", [
    ("EMPTY", True, "pickup-piece-100"), ("HELD", True, "score-zone"),
    ("UNKNOWN", True, "hold"), ("EMPTY", None, "hold"),
    ("HELD", False, "hold"),
])
def test_measured_possession_and_readiness_change_next_task(possession, ready, selected):
    request = fixture_request()
    request["context"].update(possession=possession, mechanism_ready=ready)
    _, reply = evaluate(request)
    assert reply["selected_task_id"] == selected
    assert reply["status"] == "BASELINE"


@pytest.mark.parametrize("change", [
    lambda r: r["context"].update(enabled=False),
    lambda r: r["context"].update(command_authority=False),
    lambda r: r["obstacles"].update(status="UNAVAILABLE", circles=[]),
])
def test_missing_motion_prerequisites_only_admit_neutral_hold(change):
    request = fixture_request()
    change(request)
    _, reply = evaluate(request)
    assert reply["selected_task_id"] == "hold"
    assert all(item["kind"] == "HOLD" or not item["eligible"] for item in reply["rankings"])


def test_changed_obstacle_and_delayed_target_recompute_feasibility():
    request = fixture_request()
    _, original = evaluate(request)
    assert original["selected_task_id"] == "pickup-piece-100"
    request["obstacles"]["circles"] = [{"obstacle_id": "new", "center_m": {"x": 1.73, "y": 2.},
                                          "velocity_mps": {"x": 0., "y": 0.}, "radius_m": .1}]
    _, blocked = evaluate(request)
    assert blocked["selected_task_id"] == "hold"
    assert not ranking(blocked, "pickup-piece-100")["eligible"]
    assert ranking(blocked, "pickup-piece-100")["outcome"]["minimum_clearance_m"] < 0
    request = fixture_request()
    track = request["prediction_request"]["history"][-1]["snapshot"]["tracks"][0]
    track["last_measurement_us"] = 700_000
    for source in track["provenance"]:
        source.update(capture_us=690_000, publication_us=695_000)
    _, dropout = evaluate(request)
    assert dropout["selected_task_id"] == "hold"
    assert not ranking(dropout, "pickup-piece-100")["eligible"]
    assert request["prediction_request"]["history"][-1]["snapshot"]["tracks"][0]["id"] == 100


@pytest.mark.parametrize("change", [
    lambda track: track.update(lifecycle="TENTATIVE"),
    lambda track: track["uncertainty"].update(xx_m2=1.1, yy_m2=1.1),
])
def test_authoritative_target_uncertainty_or_lifecycle_abstains_pickup(change):
    request = fixture_request()
    change(request["prediction_request"]["history"][-1]["snapshot"]["tracks"][0])
    _, reply = evaluate(request)
    assert reply["selected_task_id"] == "hold"
    assert not ranking(reply, "pickup-piece-100")["eligible"]


def test_target_motion_and_disappearance_use_authoritative_snapshot():
    request, prior = evaluate()
    current = copy.deepcopy(request)
    current["context"]["revision"] += 1
    track = current["prediction_request"]["history"][-1]["snapshot"]["tracks"][0]
    track["position_m"].update(x=4., y=3.)
    track["velocity_mps"].update(x=.1, y=.2)
    current["tasks"][0]["goal"]["field_pose"]["position_m"].update(x=3.7, y=3.)
    with pytest.raises(ContractError):
        consume_decision_reply(prior, request, policy=fixture_policy(), now_us=1_010_000, current_request=current)
    _, changed = evaluate(current)
    assert ranking(changed, "pickup-piece-100")["outcome"]["remaining_goal_distance_m"] > ranking(prior, "pickup-piece-100")["outcome"]["remaining_goal_distance_m"]
    current["prediction_request"]["history"][-1]["snapshot"]["tracks"] = []
    current["prediction_request"]["targets"] = current["prediction_request"]["targets"][:1]
    _, disappeared = evaluate(current)
    assert disappeared["status"] == "REJECTED" and disappeared["selected_task_id"] is None
    assert current["prediction_request"]["history"][-1]["snapshot"]["tracks"] == []


def test_conflicting_task_utility_and_switching_cost_share_candidate_constraints():
    request = fixture_request()
    request["tasks"][1].update(kind="TRANSIT", base_utility=20.)
    _, preferred = evaluate(request, policy=fixture_policy(switching_cost=0.))
    assert preferred["selected_task_id"] == "score-zone"
    request["current_task_id"] = "pickup-piece-100"
    _, hysteresis = evaluate(request, policy=fixture_policy(switching_cost=20.))
    assert hysteresis["selected_task_id"] == "pickup-piece-100"
    request["tasks"][0]["path_evidence"] = {"status": "BLOCKED", "distance_m": None,
                                               "planner_request_id": None, "valid_until_us": None}
    _, infeasible = evaluate(request, policy=fixture_policy(switching_cost=20.))
    assert not ranking(infeasible, "pickup-piece-100")["eligible"]
    assert infeasible["selected_task_id"] != "pickup-piece-100"


def test_updated_goal_and_planner_distance_are_not_task_success_labels():
    request = fixture_request()
    _, first = evaluate(request)
    request["tasks"][0]["goal"]["field_pose"]["position_m"]["x"] = 5.
    _, changed = evaluate(request)
    assert ranking(changed, "pickup-piece-100")["outcome"]["remaining_goal_distance_m"] > ranking(first, "pickup-piece-100")["outcome"]["remaining_goal_distance_m"]
    request["tasks"][0]["path_evidence"] = {"status": "REACHABLE", "distance_m": 6.,
                                               "planner_request_id": "planner-1", "valid_until_us": 1_200_000}
    _, routed = evaluate(request)
    outcome = ranking(routed, "pickup-piece-100")["outcome"]
    assert outcome["distance_basis"] == "PLANNER_EVIDENCE" and outcome["route_distance_m"] == 6.
    assert outcome["task_success_probability"] is None and outcome["task_completion_time_s"] is None


def test_field_containment_accounts_for_footprint_even_when_centers_are_inside():
    request = fixture_request()
    for history in request["prediction_request"]["history"]:
        history["snapshot"]["ego"]["field_pose"]["position_m"]["x"] = 19.4
    request["tasks"][0]["goal"]["field_pose"]["position_m"]["x"] = 19.8
    request["tasks"][1]["goal"]["field_pose"]["position_m"]["x"] = 19.8
    _, reply = evaluate(request)
    pickup = ranking(reply, "pickup-piece-100")
    assert all(sample["x_m"] < 20. for sample in pickup["outcome"]["samples"])
    assert not pickup["eligible"] and "forecast-outside-field" in pickup["reasons"]
    assert reply["selected_task_id"] != "pickup-piece-100"


@pytest.mark.parametrize("change", [
    lambda r: r.update(schema_version="frc-decision-support/2"),
    lambda r: r.update(mode="live"),
    lambda r: r.update(decision_generation=True),
    lambda r: r.update(unexpected="unsupported"),
    lambda r: r["context"].update(mechanism_ready=1),
    lambda r: r["context"].update(measured_velocity=0.),
    lambda r: r.update(policy_id="missing"),
    lambda r: r.update(action_sequence_hash="0" * 64),
    lambda r: r["context"].update(time_us=1_000_001),
    lambda r: r["context"].update(time_us=500_000),
    lambda r: r["obstacles"].update(map_version=999),
    lambda r: r["obstacles"]["circles"].append(copy.deepcopy(r["obstacles"]["circles"][0])),
    lambda r: r["tasks"][0].update(candidate_id="hold"),
    lambda r: r["tasks"][0].update(target_track_ids=[999]),
    lambda r: r["tasks"][0].update(target_track_ids=[]),
    lambda r: r["tasks"][0]["goal"]["field_pose"]["position_m"].update(x=21.),
    lambda r: r["tasks"][0]["path_evidence"].update(status="REACHABLE"),
    lambda r: r["prediction_request"].update(valid_until_us=2_000_000),
    lambda r: r["prediction_request"]["history"][-1].update(valid_mask=False),
    lambda r: r["prediction_request"]["history"][-1]["snapshot"]["ego"].update(localization_valid=False),
    lambda r: r["prediction_request"]["history"][-1]["snapshot"].update(epoch=8),
    lambda r: r["prediction_request"]["candidates"][0]["samples"][0]["body_twist"].update(vx_mps=.1),
])
def test_request_rejection_vectors(change):
    request = fixture_request()
    change(request)
    # Updating the hash cannot make physically or semantically invalid actions valid.
    if request["action_sequence_hash"] != "0" * 64:
        refresh(request)
    with pytest.raises(ContractError):
        validate_decision_request(request, fixture_policy(), 1_010_000)


@pytest.mark.parametrize("payload", [
    '{}', '[]', '{', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}',
    '{"schema_version":"frc-decision-support/1","schema_version":"frc-decision-support/1"}',
    '"' + 'x' * 1_000_001 + '"',
])
def test_malformed_duplicate_nonfinite_oversized_rejection(payload):
    with pytest.raises(ContractError):
        load_decision(payload, "request")


def test_resource_bounds_and_hash_bind_entire_context_without_rebasing():
    request = fixture_request()
    with pytest.raises(ContractError):
        validate_decision_request(request, fixture_policy(max_candidates=2), 1_010_000)
    with pytest.raises(ContractError):
        validate_decision_request(request, fixture_policy(max_samples=2), 1_010_000)
    with pytest.raises(ContractError):
        load_decision(json.dumps(request), "request", max_bytes=100)
    original_hash = request_hash(request)
    request["context"]["revision"] += 1
    assert request_hash(request) != original_hash
    request, reply = evaluate()
    assert reply["identity"]["history_cutoff_us"] == 1_000_000
    assert reply["generated_us"] == 1_010_000
    assert ranking(reply, "pickup-piece-100")["outcome"]["samples"][0]["x_m"] == 1.


def test_schema_dimensions_and_authority_constants_are_bounded():
    request = fixture_request()
    for mutation in (
        lambda r: r["tasks"].extend([copy.deepcopy(r["tasks"][0])] * 6),
        lambda r: r["prediction_request"]["candidates"].extend([copy.deepcopy(r["prediction_request"]["candidates"][0])] * 6),
        lambda r: r["prediction_request"]["history"].extend([copy.deepcopy(r["prediction_request"]["history"][0])] * 31),
        lambda r: r["obstacles"]["circles"].extend([copy.deepcopy(r["obstacles"]["circles"][0])] * 128),
        lambda r: r["prediction_request"]["candidates"][0].update(authority="ACCEPTED"),
        lambda r: r["prediction_request"]["candidates"][0].update(semantics="WAYPOINTS"),
    ):
        bad = copy.deepcopy(request)
        mutation(bad)
        with pytest.raises(ContractError):
            load_decision(json.dumps(bad), "request")


@pytest.mark.parametrize("change", [
    lambda r: r.update(service_epoch="old-service"),
    lambda r: r.update(decision_generation=2),
    lambda r: r.update(task_revision=2),
    lambda r: r["context"].update(revision=2),
    lambda r: r["context"].update(possession="HELD"),
    lambda r: r["prediction_request"]["source"].update(robot_boot_id="new-boot"),
    lambda r: r["prediction_request"]["source"].update(localization_revision="new-localization"),
    lambda r: r["prediction_request"]["sync"].update(mapping_revision="new-map"),
    lambda r: r["obstacles"].update(status="UNAVAILABLE", circles=[]),
])
def test_old_reply_rejected_after_any_current_decision_change(change):
    request, reply = evaluate()
    current = copy.deepcopy(request)
    change(current)
    with pytest.raises(ContractError):
        consume_decision_reply(reply, request, policy=fixture_policy(), now_us=1_010_000, current_request=current)


@pytest.mark.parametrize("change", [
    lambda r: r.update(service_epoch="old-service"),
    lambda r: r.update(decision_generation=99),
    lambda r: r.update(request_sha256="0" * 64),
    lambda r: r.update(measured=True),
    lambda r: r.update(influences_commands=True),
    lambda r: r.update(generated_us=1_010_001),
    lambda r: r.update(valid_until_us=1_010_000),
    lambda r: r["identity"].update(epoch=99),
    lambda r: r["rankings"].pop(),
    lambda r: r.update(selected_task_id="score-zone"),
    lambda r: r["rankings"][0]["outcome"].update(task_success_probability=.9),
    lambda r: r["rankings"][0]["outcome"]["samples"][0].update(valid_mask=False, x_m=1.),
    lambda r: r["rankings"][0]["outcome"]["samples"][0].update(physical_covariance={"xx_m2": .01, "xy_m2": 0., "yy_m2": .01}),
])
def test_reply_cannot_claim_authority_or_change_request_semantics(change):
    request, reply = evaluate()
    change(reply)
    with pytest.raises(ContractError):
        consume_decision_reply(reply, request, policy=fixture_policy(), now_us=1_010_000, current_request=request)


def test_synthetic_provider_admitted_with_provenance_and_shared_baseline():
    request, reply = evaluate(predictor=synthetic_learned_evidence)
    assert reply["status"] == "LEARNED_ADVISORY"
    assert reply["selected_task_id"] == reply["baseline_selected_task_id"] == "pickup-piece-100"
    assert ranking(reply, "pickup-piece-100")["forecast_kind"] == "LEARNED_FORECAST"
    assert all(sample["physical_covariance"] is not None for sample in ranking(reply, "pickup-piece-100")["outcome"]["samples"])
    assert ranking(reply, "pickup-piece-100")["outcome"]["task_success_probability"] is None
    assert consume_decision_reply(reply, request, policy=fixture_policy(), now_us=1_010_000, current_request=request) == reply


@pytest.mark.parametrize("change", [
    lambda r: r["rankings"][0].update(score=1000.),
    lambda r: r["rankings"][0].update(eligible=False, score=None),
    lambda r: r["rankings"][0]["outcome"]["samples"][-1].update(x_m=1.21),
    lambda r: r["rankings"][0]["outcome"]["samples"][-1]["physical_covariance"].update(xx_m2=.02),
    lambda r: r["rankings"][0]["outcome"].update(remaining_goal_distance_m=0.),
    lambda r: r.update(selected_task_id="hold"),
    lambda r: r.update(learned_domain_id="missing"),
    lambda r: r["learned_prediction_reply"].update(uncertainty_calibration_id="unapproved"),
    lambda r: r["learned_prediction_reply"].update(request_id="older-request"),
])
def test_consumer_recomputes_learned_rankings_from_original_forecast_proof(change):
    request, reply = evaluate(predictor=synthetic_learned_evidence)
    assert reply["status"] == "LEARNED_ADVISORY"
    change(reply)
    with pytest.raises(ContractError):
        consume_decision_reply(reply, request, policy=fixture_policy(), now_us=1_010_000, current_request=request)


@pytest.mark.parametrize("failure", ["ood", "domain", "uncalibrated", "variance", "masked", "reordered", "crash"])
def test_failed_learned_evidence_preserves_baseline(failure):
    def predictor(request, cancellation):
        evidence = synthetic_learned_evidence(request, cancellation)
        if failure == "crash":
            raise RuntimeError("synthetic unavailable")
        if failure == "ood":
            return LearnedEvidence(evidence.reply, False, "synthetic-domain-v1")
        if failure == "domain":
            return LearnedEvidence(evidence.reply, True, "unapproved-domain")
        if failure == "uncalibrated":
            evidence.reply["uncertainty_calibration_id"] = "unapproved"
        if failure == "variance":
            for forecast in evidence.reply["forecasts"]:
                for sample in forecast["samples"]:
                    sample.update(covariance_xx_m2=2., covariance_yy_m2=2.)
        if failure == "masked":
            masked = make_masked_reply(request, request["issued_us"])
            masked["uncertainty_calibration_id"] = "synthetic-calibration-v1"
            return LearnedEvidence(masked, True, "synthetic-domain-v1")
        if failure == "reordered":
            evidence.reply["request_id"] = "prior-request"
        return evidence
    _, reply = evaluate(predictor=predictor)
    assert reply["status"] == "BASELINE"
    assert reply["selected_task_id"] == reply["baseline_selected_task_id"] == "pickup-piece-100"
    assert all(item["forecast_kind"] == "KINEMATIC_ASSUMPTION" for item in reply["rankings"])


def test_inference_timeout_and_overload_are_bounded_and_cooperatively_cancelled():
    started, stopped = threading.Event(), threading.Event()
    def blocked(request, cancellation):
        started.set()
        assert cancellation.wait(.2)
        stopped.set()
        return synthetic_learned_evidence(request, cancellation)
    before = time.perf_counter()
    _, reply = evaluate(predictor=blocked, policy=fixture_policy(inference_timeout_s=.005))
    elapsed = time.perf_counter() - before
    assert started.is_set() and stopped.wait(.2)
    assert elapsed < .5 and reply["status"] == "BASELINE"
    assert reply["selected_task_id"] == "pickup-piece-100"


def test_timed_out_uncooperative_provider_does_not_queue_more_inference():
    release, entered = threading.Event(), threading.Event()
    calls = []
    def uncooperative(prediction, cancellation):
        calls.append(prediction["request_id"])
        entered.set()
        release.wait(.3)
        return synthetic_learned_evidence(prediction, cancellation)
    request = fixture_request()
    service = DecisionService(fixture_policy(inference_timeout_s=.005), authority=authority_for(request),
                              now_us=lambda: 1_010_000, predictor=uncooperative)
    request["service_epoch"] = service.service_epoch
    try:
        first = service.evaluate(json.dumps(request))
        assert entered.is_set() and first["status"] == "BASELINE"
        assert first["learned_status"] == "learned-timeout"
        next_request = copy.deepcopy(request)
        next_request["prediction_request"]["request_id"] = "newer-request"
        next_request["decision_generation"] += 1
        second = service.evaluate(json.dumps(next_request))
        assert second["status"] == "BASELINE" and second["learned_status"] == "learned-overload"
        assert calls == ["prediction-9"]
        assert second["selected_task_id"] == first["selected_task_id"]
    finally:
        release.set()
        service.close()


def test_expiry_during_optional_inference_rejects_without_rebasing_forecast():
    request = fixture_request()
    clock = [1_010_000]
    def expired(prediction, cancellation):
        evidence = synthetic_learned_evidence(prediction, cancellation)
        clock[0] = prediction["valid_until_us"]
        return evidence
    service = DecisionService(fixture_policy(), authority=authority_for(request), now_us=lambda: clock[0], predictor=expired)
    request["service_epoch"] = service.service_epoch
    try:
        reply = service.evaluate(json.dumps(request))
        assert reply["status"] == "REJECTED" and reply["selected_task_id"] is None
        assert reply["identity"]["history_cutoff_us"] == 1_000_000
        assert reply["generated_us"] == 1_400_000
    finally:
        service.close()


def test_inflight_cancellation_and_late_inference_do_not_publish_advice():
    entered, release = threading.Event(), threading.Event()
    request = fixture_request()
    def predictor(prediction, cancellation):
        entered.set()
        assert release.wait(.2)
        return synthetic_learned_evidence(prediction, cancellation)
    service = DecisionService(fixture_policy(inference_timeout_s=.05), authority=authority_for(request),
                              now_us=lambda: 1_010_000, predictor=predictor)
    request["service_epoch"] = service.service_epoch
    results = []
    worker = threading.Thread(target=lambda: results.append(service.evaluate(json.dumps(request))))
    try:
        worker.start()
        assert entered.wait(.2)
        service.cancel()
        release.set()
        worker.join(timeout=.5)
        assert not worker.is_alive()
        assert results[0]["status"] == "CANCELLED"
        assert results[0]["selected_task_id"] is None and results[0]["rankings"] == []
    finally:
        release.set()
        worker.join(timeout=.5)
        service.close()


def test_authority_reboot_and_robot_clock_rollback_require_fresh_session():
    request = fixture_request()
    clock = [1_030_000]
    service = DecisionService(fixture_policy(), authority=authority_for(request), now_us=lambda: clock[0])
    request["service_epoch"] = service.service_epoch
    try:
        assert service.evaluate(json.dumps(request))["status"] == "BASELINE"
        reboot = copy.deepcopy(request)
        reboot["decision_generation"] += 1
        reboot["prediction_request"]["source"]["robot_boot_id"] = "new-robot-boot"
        rejected = service.evaluate(json.dumps(reboot))
        assert rejected["status"] == "REJECTED" and rejected["selected_task_id"] is None
        reordered = copy.deepcopy(request)
        reordered["decision_generation"] += 1
        clock[0] = 1_020_000
        rolled_back = service.evaluate(json.dumps(reordered))
        assert rolled_back["status"] == "REJECTED" and rolled_back["selected_task_id"] is None
        reboot["service_epoch"] = service.reset(authority_for(reboot))
        assert service.evaluate(json.dumps(reboot))["status"] == "BASELINE"
    finally:
        service.close()


def test_loopback_baseline_and_unavailable_or_timeout_recompute_local_baseline():
    request = fixture_request()
    service = DecisionService(fixture_policy(), authority=authority_for(request), now_us=lambda: 1_010_000)
    request["service_epoch"] = service.service_epoch
    try:
        with loopback_server(lambda text: json.dumps(service.evaluate(text))) as address:
            reply = safe_decision_roundtrip(address, request, policy=fixture_policy(), now_us=lambda: 1_010_000,
                                            current_request=request, timeout_s=.05)
            assert reply["status"] == "BASELINE" and reply["selected_task_id"] == "pickup-piece-100"
        unavailable = safe_decision_roundtrip(address, request, policy=fixture_policy(), now_us=lambda: 1_010_000,
                                              current_request=request, timeout_s=.01)
        assert unavailable["status"] == "BASELINE" and unavailable["service_epoch"] == request["service_epoch"]
        assert unavailable["selected_task_id"] == "pickup-piece-100"
        def slow(_):
            time.sleep(.03)
            return '{}'
        with loopback_server(slow) as address:
            timeout = safe_decision_roundtrip(address, request, policy=fixture_policy(), now_us=lambda: 1_010_000,
                                              current_request=request, timeout_s=.005)
        assert timeout["status"] == "BASELINE" and timeout["selected_task_id"] == "pickup-piece-100"
    finally:
        service.close()


def test_reset_reordering_and_cancel_do_not_admit_old_service_reply():
    request = fixture_request()
    service = DecisionService(fixture_policy(), authority=authority_for(request), now_us=lambda: 1_010_000)
    request["service_epoch"] = service.service_epoch
    try:
        reply = service.evaluate(json.dumps(request))
        # Replays cannot advance the service's latest accepted decision.
        repeated = service.evaluate(json.dumps(request))
        assert repeated["status"] in ("REJECTED", "CANCELLED")
        old_epoch = service.service_epoch
        authority = authority_for(request)
        service.reset(authority)
        assert service.service_epoch != old_epoch
        stale = service.evaluate(json.dumps(request))
        assert stale["status"] in ("REJECTED", "CANCELLED")
        current = copy.deepcopy(request)
        current["service_epoch"] = service.service_epoch
        with pytest.raises(ContractError):
            consume_decision_reply(reply, request, policy=fixture_policy(), now_us=1_010_000, current_request=current)
        current["context"]["cancel_requested"] = True
        cancelled = service.evaluate(json.dumps(current))
        assert cancelled["selected_task_id"] is None
        assert not any(item["eligible"] for item in cancelled["rankings"])
    finally:
        service.close()


def test_latest_worker_keeps_one_running_and_replaces_pending_future():
    entered, release = threading.Event(), threading.Event()
    class BlockingService(DecisionService):
        # Worker scheduling is tested independently of learned-provider timeout.
        def __init__(self):
            seed = fixture_request()
            super().__init__(fixture_policy(), authority=authority_for(seed), now_us=lambda: 1_010_000)
            self.calls = []
        def evaluate(self, payload):
            request_id = json.loads(payload)["prediction_request"]["request_id"]
            self.calls.append(request_id)
            if request_id == "first":
                entered.set()
                assert release.wait(.5)
            return {"payload": request_id}
    service = BlockingService()
    worker = LatestDecisionWorker(service)
    try:
        def payload(identifier, generation):
            request = fixture_request()
            request["service_epoch"] = service.service_epoch
            request["prediction_request"]["request_id"] = identifier
            request["decision_generation"] = generation
            return json.dumps(request)
        first = worker.submit(payload("first", 1))
        assert entered.wait(.2)
        superseded = worker.submit(payload("second", 2))
        for invalid in (payload("reordered", 1), payload("repeated", 2)):
            with pytest.raises(ContractError):
                worker.submit(invalid)
            assert not superseded.cancelled()
        wrong_epoch = json.loads(payload("wrong-epoch", 3))
        wrong_epoch["service_epoch"] = "old-service"
        with pytest.raises(ContractError):
            worker.submit(json.dumps(wrong_epoch))
        assert not superseded.cancelled()
        latest = worker.submit(payload("third", 3))
        release.set()
        assert first.result(timeout=.5) == {"payload": "first"}
        assert latest.result(timeout=.5) == {"payload": "third"}
        with pytest.raises(CancelledError):
            superseded.result(timeout=.5)
        assert service.calls == ["first", "third"]
    finally:
        release.set()
        worker.close()
        service.close()


def test_latest_worker_supersedes_popped_request_before_service_begin():
    entered, release = threading.Event(), threading.Event()
    class BeforeBeginService(DecisionService):
        def evaluate(self, payload):
            if json.loads(payload)["decision_generation"] == 1:
                entered.set()
                assert release.wait(.5)
            return super().evaluate(payload)
    request = fixture_request()
    service = BeforeBeginService(fixture_policy(), authority=authority_for(request), now_us=lambda: 1_010_000)
    request["service_epoch"] = service.service_epoch
    worker = LatestDecisionWorker(service)
    try:
        first = worker.submit(json.dumps(request))
        assert entered.wait(.2)
        newer = copy.deepcopy(request)
        newer["decision_generation"] = 2
        newer["prediction_request"]["request_id"] = "latest-request"
        next_reply = worker.submit(json.dumps(newer))
        release.set()
        superseded = first.result(timeout=.5)
        assert superseded["status"] == "REJECTED"
        assert superseded["reason"] == "superseded-before-worker-start"
        assert superseded["selected_task_id"] is None
        latest = next_reply.result(timeout=.5)
        assert latest["status"] == "BASELINE" and latest["decision_generation"] == 2
        assert latest["selected_task_id"] == "pickup-piece-100"
    finally:
        release.set()
        worker.close()
        service.close()


def test_benchmark_limits_new_output_and_reports_actual_local_measurement(tmp_path):
    output = tmp_path / "measurement.json"
    report = benchmark(output, runs=2)
    assert report["runs"] == 2 and report["max_rss_bytes"] > 0
    assert report["baseline_loopback_agreement"] == 2
    assert report["physical_safety_certified"] is False
    assert report["influences_commands"] is False
    assert set(report["timings_ms"]) == {"direct-baseline", "loopback-baseline", "synthetic-provider"}
    for metrics in report["timings_ms"].values():
        assert 0 <= metrics["p50"] <= metrics["max"]
    with pytest.raises(FileExistsError):
        benchmark(output, runs=2)
    with pytest.raises(ValueError):
        benchmark(tmp_path / "other.json", runs=101)
