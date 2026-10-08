import copy
import hashlib
import json
import math

import pytest

from frc_world_model.bindings import CONTRACT, parse_request
from frc_world_model.converter import COMMAND_LAYOUT,EGO_LAYOUT,convert_requests
from frc_world_model.data import load_jsonl


def fixture():
    r = json.loads((CONTRACT / "request-v1.json").read_text())
    for h in r["history"]: h["accepted_command"]["mechanisms"] = []
    for c in r["candidates"]:
        for s in c["samples"]:
            s["mechanisms"] = []
            s["body_twist"]["vx_mps"] = 1.9  # never an executed action label
    return r


def metadata(r):
    return {r["request_id"]:{"robot_id":"fixture-robot","session_id":"fixture-session","day":"2026-10-07",
        "calibration_id":"fixture-calibration","source_id":"robot-state-estimator","label_provenance":"synthetic_fixture",
        "ego_capture_us_by_snapshot":{str(h["snapshot"]["id"]):h["snapshot"]["ego"]["time_us"]-1000 for h in r["history"]},
        "time_sync_limit_us":1000,"input_causal":True,"complete_accepted_command_history":True}}


def convert(tmp_path,r,meta=None):
    path = tmp_path/"original.jsonl"
    path.write_text(json.dumps(r)+"\n")
    before = path.read_bytes()
    result,report = convert_requests(path,metadata(r) if meta is None else meta,
        decode=lambda text:parse_request(text,now_us=r["issued_us"],max_sync_error_us=1000))
    assert path.read_bytes() == before
    assert report["source_sha256"] == hashlib.sha256(before).hexdigest()
    return result,report


def test_causal_units_identity_provenance_and_only_accepted_actions(tmp_path):
    r = fixture()
    for h in r["history"]:
        h["snapshot"]["ego"]["field_pose"]["heading_rad"] = math.pi/2
        h["snapshot"]["ego"]["field_velocity"]["linear_mps"] = {"x":0.,"y":.2}
    result,manifest = convert(tmp_path,r)
    states = [x for x in result if x["kind"]=="state"]
    actions = [x for x in result if x["kind"]=="action"]
    assert len(states)==2 and len(actions)==1
    assert states[0]["values"][3:5] == pytest.approx((.2,0.))
    assert states[0]["state_time_ns"]==900000000
    assert states[0]["capture_time_ns"]==899000000
    assert states[0]["identity"]["reset_epoch"]=="7"
    assert states[0]["identity"]["sync_mapping_revision"]=="direct-clock-v1"
    assert actions[0]["values"]==[.2,0.,0.]
    assert actions[0]["accepted_time_ns"]==899000000 and actions[0]["interval_end_ns"]==999000000
    assert manifest["terminal_commands_excluded_unknown_interval"]==1
    assert manifest["retained_wire_provenance"][0]["original_request"]==r
    out = tmp_path/"converted.jsonl"
    out.write_text("".join(json.dumps(x)+"\n" for x in result))
    ingested = load_jsonl(out,EGO_LAYOUT,COMMAND_LAYOUT)
    assert len(ingested.records)==3 and not any(ingested.exclusions.values())
    assert all(x.sync_error_ns==100000 for x in ingested.records)
    assert all(json.loads(x.wire_provenance_json)["requests"][0]["request_id"]=="prediction-9" for x in ingested.records)


@pytest.mark.parametrize("bad",["metadata","capture","provenance","mechanisms"])
def test_missing_prerequisites_never_guessed(tmp_path,bad):
    r = fixture(); meta = metadata(r)
    if bad=="metadata": meta={}
    if bad=="capture": meta[r["request_id"]]["ego_capture_us_by_snapshot"]={}
    if bad=="provenance": meta[r["request_id"]]["label_provenance"]="future_smoothed"
    if bad=="mechanisms": r["history"][0]["accepted_command"]["mechanisms"]=[{"channel":"intake_enabled","unit":"bool","value":1.}]
    with pytest.raises(ValueError): convert(tmp_path,r,meta)


def test_resets_do_not_invent_action_intervals(tmp_path):
    r = fixture(); other=copy.deepcopy(r)
    other["request_id"]="next-epoch"; other["epoch"]+=1
    for h in other["history"]:
        h["snapshot"]["epoch"]+=1
        for track in h["snapshot"]["tracks"]: track["epoch"]+=1
    for h in other["history"]: h["accepted_command"]=None
    path = tmp_path/"original.jsonl"
    path.write_text(json.dumps(r)+"\n"+json.dumps(other)+"\n")
    meta = {**metadata(r),**metadata(other)}
    records,report = convert_requests(path,meta,decode=lambda text:parse_request(text,now_us=1010000,max_sync_error_us=1000))
    assert report["action_records"]==1
    assert [x for x in records if x["kind"]=="action"][0]["identity"]["reset_epoch"]=="7"
    assert len({x["identity"]["reset_epoch"] for x in records})==2


def test_overlapping_requests_merge_context_without_erasing_physical_snapshots(tmp_path):
    r=fixture(); other=copy.deepcopy(r);other["request_id"]="prediction-overlap"
    path=tmp_path/"overlap.jsonl"
    path.write_text(json.dumps(r)+"\n"+json.dumps(other)+"\n")
    records,manifest=convert_requests(path,{**metadata(r),**metadata(other)},
        decode=lambda text:parse_request(text,now_us=1010000,max_sync_error_us=1000))
    assert len(records)==3 and manifest["state_records"]==2
    assert all(len(x["wire_provenance"]["requests"])==2 for x in records)
    out=tmp_path/"merged.jsonl";out.write_text("".join(json.dumps(x)+"\n" for x in records))
    report=load_jsonl(out,EGO_LAYOUT,COMMAND_LAYOUT)
    assert len(report.records)==3 and not any(report.exclusions.values())
