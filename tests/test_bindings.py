import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from frc_world_model.bindings import (
    CONTRACT, ContractError, WirePolicy, candidate_body_actions3, decode_ego_history6,
    load_document, make_masked_reply, parse_reply, parse_request, validate_reply, validate_request,
)


def golden(kind):
    return json.loads((CONTRACT / f"{kind}-v1.json").read_text())


def policy():
    return WirePolicy({"fixture-model-v1":("synthetic-calibration-v1",)},
        (("OFFSEASON_2026","synthetic-field-v1","geometry-v1"),),("robot-config-v1",),
        -20.,-20.,20.,20.,3.,5.,1e-6,10.,.2,300000,500000,1000,
        {"intake_enabled":("bool",0.,1.)})


def request(raw=None,now=1030000):
    return parse_request(json.dumps(raw or golden("request")),now_us=now,max_sync_error_us=1000,policy=policy())


def test_exact_schema_and_vector_hashes_and_roundtrip():
    manifest = json.loads((CONTRACT / "manifest.json").read_text())
    assert manifest["source_commit"] == "1f8b9aace3fd320e3545744e3125aa348089bc99"
    for name,sha in manifest["files"].items():
        assert hashlib.sha256((CONTRACT / name).read_bytes()).hexdigest() == sha
    r = request()
    reply = parse_reply(json.dumps(golden("reply")),r,now_us=1030000,policy=policy())
    assert reply == golden("reply")
    assert "status" not in reply
    assert reply["history_cutoff_us"] == 1000000


@pytest.mark.parametrize("change",[
    lambda r:r.update(schema_version="frc-prediction/2"),
    lambda r:r.update(frame="ROBOT_RELATIVE"),
    lambda r:r.update(timestamp_domain="ROBOT_MONOTONIC_NS"),
    lambda r:r.update(issued_us=1030001),
    lambda r:r.update(valid_until_us=1030000),
    lambda r:r["sync"].update(valid=False),
    lambda r:r["sync"].update(maximum_error_us=1001),
    lambda r:r["history"][0]["snapshot"].update(epoch=8),
    lambda r:r["history"][0]["snapshot"]["ego"].update(time_us=1000001),
    lambda r:r["history"][1]["snapshot"].update(id=41),
    lambda r:r["history"][1].update(valid_mask=False),
    lambda r:r["history"][0]["accepted_command"].update(accepted_us=900001),
    lambda r:r["history"][0]["snapshot"]["tracks"][0].update(estimate_us=900001),
    lambda r:r["history"][0]["snapshot"]["tracks"][0]["uncertainty"].update(xy_m2=1.),
    lambda r:r["history"][0]["snapshot"]["tracks"][0]["position_m"].update(x=21.),
    lambda r:r["targets"].append(copy.deepcopy(r["targets"][0])),
    lambda r:r["candidates"][0]["samples"][1].update(offset_us=100001),
    lambda r:r["candidates"][1].update(candidate_id="hold"),
    lambda r:r["candidates"][0]["samples"][0]["body_twist"].update(vx_mps=4.),
    lambda r:r["candidates"][0]["samples"][0]["mechanisms"][0].update(value=.5),
    lambda r:r["source"].update(configuration_revision="missing"),
])
def test_request_rejection_vectors(change):
    raw = golden("request")
    change(raw)
    with pytest.raises(ContractError): request(raw)


@pytest.mark.parametrize("payload",[
    '{"schema_version":"frc-prediction/1","schema_version":"frc-prediction/1"}',
    '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}',
    '[]', '{}', '{', '"' + 'x'*1000000 + '"',
])
def test_nonfinite_duplicate_malformed_and_oversized_documents(payload):
    with pytest.raises(ContractError): load_document(payload,"request")


@pytest.mark.parametrize("change",[
    lambda r:r.update(epoch=8),lambda r:r.update(request_id="old"),
    lambda r:r["source"].update(robot_boot_id="old-boot"),
    lambda r:r.update(history_cutoff_us=999999),lambda r:r.update(generated_us=1030001),
    lambda r:r.update(valid_until_us=1030000),
    lambda r:r.update(uncertainty_calibration_id="unapproved"),
    lambda r:r["forecasts"].pop(),
    lambda r:r["forecasts"][0].update(candidate_id="other"),
    lambda r:r["forecasts"][0]["samples"][1].update(offset_us=100001),
    lambda r:r["forecasts"][0]["samples"][1].update(covariance_xy_m2=.5),
    lambda r:r["forecasts"][0]["samples"][1].update(covariance_xx_m2=0.,covariance_yy_m2=0.),
    lambda r:r["forecasts"][0]["samples"][1].update(x_m=19.),
    lambda r:r["forecasts"][0]["samples"][1].update(vx_mps=4.),
    lambda r:r["forecasts"][1]["samples"][2].update(x_m=1.),
    lambda r:r["forecasts"][0]["samples"][0].update(x_m=float("nan")),
])
def test_reply_rejection_vectors(change):
    reply = golden("reply")
    change(reply)
    with pytest.raises(ContractError): validate_reply(request(),reply,now_us=1030000,policy=policy())


def test_masked_abstention_exact_wire_and_unusable_baseline():
    r = request()
    reply = make_masked_reply(r,1030000)
    assert len(reply["forecasts"]) == len(r["targets"])*len(r["candidates"])
    assert all(not s["valid_mask"] and s["confidence"] is None for f in reply["forecasts"] for s in f["samples"])
    assert load_document(json.dumps(reply),"reply") == reply
    with pytest.raises(ContractError,match="uncalibrated"): validate_reply(r,reply,now_us=1030000,policy=policy())
    # Even an approved calibration ID cannot make a fully masked reply useful.
    reply["uncertainty_calibration_id"] = "synthetic-calibration-v1"
    with pytest.raises(ContractError,match="no remaining"): validate_reply(r,reply,now_us=1030000,policy=policy())


def test_bounded_lag_current_world_acceptance_never_rebases():
    r,reply = request(),golden("reply")
    current = copy.deepcopy(r["history"][-1]["snapshot"])
    current["id"] += 1
    current["ego"]["time_us"] = 1025000
    out = validate_reply(r,reply,now_us=1030000,policy=policy(),current_snapshot=current)
    assert out["snapshot_id"] == 42 and out["history_cutoff_us"] == 1000000
    for attr,value in (("epoch",8),("obstacle_map_version",4)):
        bad = copy.deepcopy(current); bad[attr] = value
        with pytest.raises(ContractError): validate_reply(r,reply,now_us=1030000,policy=policy(),current_snapshot=bad)
    current["tracks"] = []
    with pytest.raises(ContractError,match="no longer"): validate_reply(r,reply,now_us=1030000,policy=policy(),current_snapshot=current)
    with pytest.raises(ContractError,match="too old"): request(now=1200001)


def test_signed_field_plane_rotation_and_missing_mechanisms():
    r = golden("request")
    for entry in r["history"]:
        ego = entry["snapshot"]["ego"]
        ego["field_pose"]["position_m"]["x"] = -1.
        ego["field_pose"]["heading_rad"] = math.pi/2
        ego["field_velocity"]["linear_mps"] = {"x":0.,"y":.2}
    r = request(r)
    assert decode_ego_history6(r)[0][0] == -1.
    assert decode_ego_history6(r)[0][3:5] == pytest.approx((.2,0.))
    with pytest.raises(ContractError,match="mechanism inputs"): candidate_body_actions3(r,"hold")
    with pytest.raises(ContractError,match="no mechanism state"): candidate_body_actions3(r,"hold",state_dim=8,action_dim=4)
    for c in r["candidates"]:
        for s in c["samples"]: s["mechanisms"] = []
    assert candidate_body_actions3(r,"transit") == ((.2,0.,0.),(.2,0.,0.))
    r["history"][0]["valid_mask"] = False
    with pytest.raises(ContractError,match="zero-filled"): decode_ego_history6(r)


def test_caller_cannot_relax_approved_sync_policy():
    r=golden("request");r["sync"]["maximum_error_us"]=1001
    with pytest.raises(ContractError,match="unsynchronized"):
        parse_request(json.dumps(r),now_us=1030000,max_sync_error_us=100000,policy=policy())
