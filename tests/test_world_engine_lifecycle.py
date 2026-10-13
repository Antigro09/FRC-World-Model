"""World-State-emitted snapshots test lifecycle interoperability without a JDK.

Inputs/configuration are synthetic. Snapshots are genuine WorldEngine output
from the pinned source, not manually assigned lifecycle labels. Regenerate with
tools/generate_lifecycle_fixtures.py and an available World-State checkout/JDK.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from frc_world_model.bindings import ContractError
from frc_world_model.decision_bindings import (
    CONTRACT, action_sequence_hash, authority_for, consume_decision_reply,
)
from frc_world_model.decision_service import DecisionService
from tools.benchmark_decisions import fixture_policy, fixture_request


ROOT = Path(__file__).resolve().parents[1]
NAME = "world-engine-lifecycle-fixtures.json"
FIXTURES = json.loads((ROOT / "contracts/decision-support-v1" / NAME).read_text())
SCENARIOS = {value["name"]: value for value in FIXTURES["scenarios"]}


def request_from_engine(name):
    request = fixture_request()
    snapshot = copy.deepcopy(SCENARIOS[name]["snapshot"])
    prediction = request["prediction_request"]
    cutoff = snapshot["ego"]["time_us"]
    prediction.update(epoch=snapshot["epoch"], snapshot_id=snapshot["id"],
        history_cutoff_us=cutoff, issued_us=cutoff + 10_000, valid_until_us=cutoff + 400_000,
        history=[{"snapshot": snapshot, "valid_mask": True, "accepted_command": None}])
    prediction["targets"][1]["track_id"] = 1
    request["tasks"][0]["target_track_ids"] = [1]
    request["context"]["time_us"] = cutoff
    request["obstacles"].update(map_version=snapshot["obstacle_map_version"], time_us=cutoff)
    return request


def evaluate(request):
    policy = fixture_policy()
    now = request["prediction_request"]["issued_us"]
    service = DecisionService(policy, authority=authority_for(request), now_us=lambda: now)
    request["service_epoch"] = service.service_epoch
    original = copy.deepcopy(request)
    try:
        reply = service.evaluate(json.dumps(request, allow_nan=False))
        assert request == original
        return reply
    finally:
        service.close()


def pickup(reply):
    return next(row for row in reply["rankings"] if row["kind"] == "PICKUP")


def test_world_engine_fixture_identity_and_packaged_parity():
    assert (ROOT / "contracts/decision-support-v1" / NAME).read_bytes() == (CONTRACT / NAME).read_bytes()
    assert FIXTURES["fixture_version"] == "world-engine-lifecycle/1"
    assert FIXTURES["synthetic_inputs"] is True
    assert FIXTURES["source_commit"] == "4590335421306da7964b5bd91dfa4da7922982ea"
    assert len(FIXTURES["source_sha256"]) == 6
    assert all(len(value) == 64 for value in FIXTURES["source_sha256"].values())
    assert FIXTURES["harness_sha256"] == hashlib.sha256(
        (ROOT / "tests/java/WorldEngineLifecycleFixtures.java").read_bytes()).hexdigest()


@pytest.mark.parametrize("name,expected_lifecycle", [
    ("confirmed_at_capture", "CONFIRMED"),
    ("delayed_camera_frame", "COASTING"),
    ("fresh_coasting_after_dropout", "COASTING"),
    ("decision_age_boundary", "COASTING"),
])
def test_established_engine_tracks_remain_eligible_with_bounded_measurement_age(name, expected_lifecycle):
    scenario = SCENARIOS[name]
    assert scenario["first_frame"] == {"status": "ACCEPTED", "accepted_measurements": 1}
    assert scenario["last_frame"] == {"status": "ACCEPTED", "accepted_measurements": 1}
    request = request_from_engine(name)
    track = request["prediction_request"]["history"][0]["snapshot"]["tracks"][0]
    assert track["lifecycle"] == expected_lifecycle
    if expected_lifecycle == "COASTING":
        assert track["estimate_kind"] == "PROPAGATED"
        assert track["estimate_us"] > track["last_measurement_us"]
    reply = evaluate(request)
    assert reply["status"] == "BASELINE"
    assert pickup(reply)["eligible"]
    assert reply["selected_task_id"] == "pickup-piece-100"
    assert consume_decision_reply(reply, request, policy=fixture_policy(),
        now_us=request["prediction_request"]["issued_us"], current_request=request) == reply


def test_coasting_ages_out_before_engine_retention_expiry_without_clearing_the_object():
    request = request_from_engine("decision_age_expired")
    snapshot = request["prediction_request"]["history"][0]["snapshot"]
    assert len(snapshot["tracks"]) == 1
    age = request["prediction_request"]["issued_us"] - snapshot["tracks"][0]["last_measurement_us"]
    assert age == FIXTURES["decision_max_track_age_us"] + 1 < FIXTURES["engine_expiry_us"]
    reply = evaluate(request)
    assert reply["status"] == "BASELINE"
    assert not pickup(reply)["eligible"]
    assert "target-stale-or-unconfirmed" in pickup(reply)["reasons"]
    assert reply["selected_task_id"] == "hold"
    assert len(snapshot["tracks"]) == 1


def test_coasting_age_is_rechecked_at_reply_consumption_without_restamping_capture_time():
    request = request_from_engine("decision_age_boundary")
    reply = evaluate(request)
    assert pickup(reply)["eligible"]
    track = request["prediction_request"]["history"][0]["snapshot"]["tracks"][0]
    measured_us = track["last_measurement_us"]
    with pytest.raises(ContractError, match="sensing prerequisites expired"):
        consume_decision_reply(reply, request, policy=fixture_policy(),
            now_us=request["prediction_request"]["issued_us"] + 1, current_request=request)
    assert track["last_measurement_us"] == measured_us


def test_tentative_engine_track_is_not_admitted_by_delayed_snapshot():
    assert SCENARIOS["tentative_delayed"]["first_frame"] is None
    request = request_from_engine("tentative_delayed")
    assert request["prediction_request"]["history"][0]["snapshot"]["tracks"][0]["lifecycle"] == "TENTATIVE"
    reply = evaluate(request)
    assert not pickup(reply)["eligible"]
    assert reply["selected_task_id"] == "hold"


def test_fresh_coasting_retains_independent_uncertainty_gate():
    request = request_from_engine("delayed_camera_frame")
    request["prediction_request"]["history"][0]["snapshot"]["tracks"][0]["uncertainty"].update(xx_m2=1.1, yy_m2=1.1)
    reply = evaluate(request)
    assert not pickup(reply)["eligible"]
    assert "target-uncertainty-out-of-domain" in pickup(reply)["reasons"]
    assert reply["selected_task_id"] == "hold"


def test_fresh_coasting_retains_configured_target_confidence_gate():
    request = request_from_engine("delayed_camera_frame")
    request["prediction_request"]["history"][0]["snapshot"]["tracks"][0]["confidence"] = .49
    reply = evaluate(request)
    assert not pickup(reply)["eligible"]
    assert "target-confidence-below-policy" in pickup(reply)["reasons"]
    assert reply["selected_task_id"] == "hold"


def test_engine_expired_track_is_absent_and_cannot_be_resurrected_by_old_task_reference():
    request = request_from_engine("engine_expired")
    assert request["prediction_request"]["history"][0]["snapshot"]["tracks"] == []
    reply = evaluate(request)
    assert reply["status"] == "REJECTED"
    assert reply["selected_task_id"] is None
    # A freshly constructed candidate set may still provide neutral hold;
    # measured EMPTY possession means SCORE remains ineligible.
    request["prediction_request"]["targets"] = request["prediction_request"]["targets"][:1]
    request["prediction_request"]["candidates"] = [candidate for candidate in
        request["prediction_request"]["candidates"] if candidate["candidate_id"] != "pickup"]
    request["tasks"] = [task for task in request["tasks"] if task["kind"] != "PICKUP"]
    request["action_sequence_hash"] = action_sequence_hash(request["prediction_request"])
    recomputed = evaluate(request)
    assert recomputed["status"] == "BASELINE"
    assert recomputed["selected_task_id"] == "hold"


def test_invalid_localization_engine_input_produces_no_eligible_sensing_dependent_motion():
    scenario = SCENARIOS["invalid_localization"]
    assert scenario["last_frame"] == {"status": "INVALID_LOCALIZATION", "accepted_measurements": 0}
    assert scenario["snapshot"]["tracks"] == []
    request = request_from_engine("invalid_localization")
    reply = evaluate(request)
    assert reply["status"] == "REJECTED"
    assert reply["selected_task_id"] is None
