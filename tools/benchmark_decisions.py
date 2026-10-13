#!/usr/bin/env python3
"""Bounded synthetic decision-support latency and baseline comparison.

The injected learned provider below is a schema/plumbing fixture, not a trained
model or evidence of ranking accuracy. Only loopback transport is exercised.
"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import platform
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frc_world_model.bindings import WirePolicy, make_masked_reply
from frc_world_model.decision_bindings import DecisionPolicy, action_sequence_hash, authority_for
from frc_world_model.decision_dynamics import extrapolated_track_position, kinematic_forecasts
from frc_world_model.decision_schema import CONTRACT
from frc_world_model.decision_service import DecisionService, LearnedEvidence, safe_decision_roundtrip
from frc_world_model.transport import loopback_server


def fixture_request():
    return json.loads((CONTRACT / "request-v1.json").read_text())


def fixture_policy(**changes):
    """Explicit synthetic bounds only; none are recommended for a robot."""
    wire = WirePolicy(
        {"fixture-model-v1": ("synthetic-calibration-v1",)},
        (("OFFSEASON_2026", "synthetic-field-v1", "geometry-v1"),),
        ("robot-config-v1",), -20., -20., 20., 20., 3., 5.,
        1e-6, 1., .2, 2_000_000, 500_000, 1000,
        {"intake_enabled": ("bool", 0., 1.)},
    )
    policy = DecisionPolicy(
        "synthetic-decision-policy-v1", wire,
        footprint_radius_m=.3, clearance_margin_m=.05, uncertainty_sigma=2.,
        ego_error_growth_mps=.01, obstacle_error_growth_mps=.01,
        nominal_speed_mps=1., travel_cost_per_m=.1, time_cost_per_s=.1,
        risk_cost=1., uncertainty_cost_per_m=.1, switching_cost=.5,
        progress_reward_per_m=2., approved_domain_id="synthetic-domain-v1",
    )
    return replace(policy, **changes)


def synthetic_learned_evidence(prediction, cancellation):
    """Deliberately synthetic all-valid covariance for provider admission tests."""
    reply = make_masked_reply(prediction, prediction["issued_us"])
    reply["uncertainty_calibration_id"] = "synthetic-calibration-v1"
    ego = kinematic_forecasts(prediction)
    latest = prediction["history"][-1]["snapshot"]
    tracks = {track["id"]: track for track in latest["tracks"]}
    targets = {target["target_id"]: target for target in prediction["targets"]}
    for forecast in reply["forecasts"]:
        target = targets[forecast["target_id"]]
        for index, sample in enumerate(forecast["samples"]):
            if cancellation.is_set():
                return LearnedEvidence(reply, False, "synthetic-domain-v1")
            if target["kind"] == "EGO":
                state = ego[forecast["candidate_id"]][index]
                x, y, vx, vy = (state[k] for k in ("x_m", "y_m", "vx_mps", "vy_mps"))
            else:
                track = tracks[target["track_id"]]
                x, y = extrapolated_track_position(track, prediction["history_cutoff_us"] + sample["offset_us"])
                vx, vy = track["velocity_mps"]["x"], track["velocity_mps"]["y"]
            sample.update(valid_mask=True, x_m=x, y_m=y, vx_mps=vx, vy_mps=vy,
                          covariance_xx_m2=.01, covariance_xy_m2=0., covariance_yy_m2=.01, confidence=.9)
    return LearnedEvidence(reply, True, "synthetic-domain-v1")


def replay_scenarios():
    """Tiny bounded replay series with authoritative context changes."""
    scenarios = []
    for name, possession, ready in (("empty-ready", "EMPTY", True),
                                    ("held-ready", "HELD", True),
                                    ("unknown-possession", "UNKNOWN", True),
                                    ("readiness-dropout", "EMPTY", None)):
        request = fixture_request()
        request["context"].update(possession=possession, mechanism_ready=ready)
        scenarios.append((name, request))
    request = fixture_request()
    request["obstacles"]["circles"] = [{"obstacle_id": "changed-obstacle", "center_m": {"x": 1.73, "y": 2.},
                                          "velocity_mps": {"x": 0., "y": 0.}, "radius_m": .1}]
    scenarios.append(("obstacle-change", request))
    request = fixture_request()
    track = request["prediction_request"]["history"][-1]["snapshot"]["tracks"][0]
    track["last_measurement_us"] = 700_000
    for source in track["provenance"]:
        source.update(capture_us=690_000, publication_us=695_000)
    scenarios.append(("delayed-target", request))
    return scenarios


def _quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def benchmark(output, *, runs=12):
    if type(runs) is not int or not 1 <= runs <= 100:
        raise ValueError("engineering benchmark requires 1..100 runs")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    policy, seed = fixture_policy(), fixture_request()
    now = seed["prediction_request"]["issued_us"]
    services = {name: DecisionService(policy, authority=authority_for(seed), now_us=lambda: now,
                                     predictor=synthetic_learned_evidence if name == "synthetic-provider" else None)
                for name in ("direct-baseline", "loopback-baseline", "synthetic-provider")}
    observations, timings = [], {name: [] for name in services}
    try:
        remote = services["loopback-baseline"]
        with loopback_server(lambda payload: json.dumps(remote.evaluate(payload), allow_nan=False, separators=(",", ":"))) as address:
            scenarios = replay_scenarios()
            for index in range(runs):
                scenario, raw = scenarios[index % len(scenarios)]
                selections = {}
                for name, service in services.items():
                    request = copy.deepcopy(raw)
                    request["service_epoch"] = service.service_epoch
                    request["decision_generation"] = index + 1
                    request["context"]["revision"] = index + 1
                    request["prediction_request"]["request_id"] = f"benchmark-{index}"
                    request["action_sequence_hash"] = action_sequence_hash(request["prediction_request"])
                    start = time.perf_counter_ns()
                    if name == "loopback-baseline":
                        reply = safe_decision_roundtrip(address, request, policy=policy, now_us=lambda: now,
                                                        current_request=request, timeout_s=.5)
                    else:
                        reply = service.evaluate(json.dumps(request, allow_nan=False, separators=(",", ":")))
                    timings[name].append((time.perf_counter_ns() - start) / 1e6)
                    selections[name] = {"status": reply["status"], "selected_task_id": reply["selected_task_id"],
                                        "baseline_selected_task_id": reply["baseline_selected_task_id"],
                                        "learned_status": reply["learned_status"]}
                observations.append({"scenario": scenario, "results": selections})
    finally:
        for service in services.values():
            service.close()
    report = {
        "host": {"platform": platform.platform(), "architecture": platform.machine(), "python": platform.python_version()},
        "scope": "bounded synthetic replay and localhost NDJSON; no hardware, trained model or physical qualification",
        "learned_provider": "synthetic kinematic fixture with synthetic covariance; no accuracy claim",
        "runs": runs, "candidates": len(seed["tasks"]), "horizon_us": seed["prediction_request"]["horizon_us"],
        "timings_ms": {name: {"p50": _quantile(values, .5), "p95": _quantile(values, .95),
                               "p99": _quantile(values, .99), "max": max(values)} for name, values in timings.items()},
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "baseline_loopback_agreement": sum(o["results"]["direct-baseline"]["selected_task_id"] ==
                                             o["results"]["loopback-baseline"]["selected_task_id"] for o in observations),
        "synthetic_provider_baseline_agreement": sum(o["results"]["direct-baseline"]["selected_task_id"] ==
                                                       o["results"]["synthetic-provider"]["selected_task_id"] for o in observations),
        "observations": observations,
        "physical_safety_certified": False, "robot_latency_qualification": "unrun",
        "simultaneous_perception_load": "unrun", "influences_commands": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def benchmark_ceiling(output):
    """One synthetic 8-candidate/101-sample/128-obstacle resource-bound scene."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    request, policy = fixture_request(), fixture_policy()
    prediction = request["prediction_request"]
    prediction.update(horizon_us=2_000_000, step_us=20_000)
    candidate, task = copy.deepcopy(prediction["candidates"][1]), copy.deepcopy(request["tasks"][1])
    prediction["candidates"], request["tasks"] = [], []
    for index in range(8):
        action = copy.deepcopy(candidate)
        action["candidate_id"] = f"ceiling-candidate-{index}"
        seed = action["samples"][0]
        action["samples"] = [{**copy.deepcopy(seed), "offset_us": offset}
            for offset in range(0, prediction["horizon_us"] + 1, prediction["step_us"])]
        context = copy.deepcopy(task)
        context.update(task_id=f"ceiling-task-{index}", candidate_id=action["candidate_id"],
                       kind="TRANSIT", target_track_ids=[])
        prediction["candidates"].append(action)
        request["tasks"].append(context)
    request["obstacles"]["circles"] = [{"obstacle_id": f"obstacle-{index}",
        "center_m": {"x": 6. + (index % 8) * .1, "y": 4. + (index // 8) * .1},
        "velocity_mps": {"x": 0., "y": 0.}, "radius_m": .05} for index in range(128)]
    request["action_sequence_hash"] = action_sequence_hash(prediction)
    service = DecisionService(policy, authority=authority_for(request), now_us=lambda: prediction["issued_us"])
    request["service_epoch"] = service.service_epoch
    try:
        payload = json.dumps(request, allow_nan=False, sort_keys=True, separators=(",", ":"))
        started = time.perf_counter_ns()
        reply = service.evaluate(payload)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    finally:
        service.close()
    report = {"scope": "one bounded synthetic ceiling scene; no learned model or physical qualification",
        "candidates": 8, "samples_per_candidate": 101, "obstacles": 128,
        "horizon_us": prediction["horizon_us"], "request_bytes": len(payload.encode()),
        "elapsed_ms": elapsed_ms, "computation_budget_ms": policy.computation_budget_s * 1000,
        "status": reply["status"], "reason": reply["reason"],
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "physical_safety_certified": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--ceiling", action="store_true", help="measure one bounded ceiling scene")
    args = parser.parse_args()
    result = benchmark_ceiling(args.output) if args.ceiling else benchmark(args.output, runs=args.runs)
    print(json.dumps(result if args.ceiling else {key: result[key] for key in
        ("runs", "timings_ms", "max_rss_bytes", "baseline_loopback_agreement", "synthetic_provider_baseline_agreement")}, allow_nan=False))
