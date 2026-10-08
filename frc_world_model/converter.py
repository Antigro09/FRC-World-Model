"""Read-only World-State request-log conversion with explicit missing metadata.

Only robot-owned ACCEPTED history commands enter training records. CANDIDATE
commands never become executed-action labels. V1 lacks session/day, ego capture
and full label provenance; the caller must supply those facts, not guess them.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .data import FeatureLayout, LOCAL_SCHEMA_VERSION, canonical_hash

EGO_LAYOUT = FeatureLayout(("x_m","y_m","heading_rad","vx_robot_mps","vy_robot_mps","omega_radps"),
                           ("m","m","rad","m/s","m/s","rad/s"),(2,))
COMMAND_LAYOUT = FeatureLayout(("vx_robot_mps","vy_robot_mps","omega_radps"),("m/s","m/s","rad/s"))


def convert_requests(path, metadata, *, decode):
    """`decode(text)` is the exact binding validator, injected for offline replay.

    It returns the validated raw request mapping. Metadata is keyed by request ID
    and contains real robot/session/day/calibration/source identity, capture_us
    by snapshot ID, and label_provenance. Unknown capture or label facts fail.
    """
    original = Path(path).read_bytes()
    states, commands = {}, {}
    line_count = 0
    retained = []
    for line in original.decode().splitlines():
        if not line.strip():
            continue
        line_count += 1
        request = decode(line)
        if not isinstance(request,dict):
            # Bindings may return a wrapper; caller must unwrap explicitly.
            raise ValueError("decoder must provide validated original request mapping")
        request_id = request["request_id"]
        meta = metadata.get(request_id)
        required = ("robot_id","session_id","day","calibration_id","source_id","ego_capture_us_by_snapshot","label_provenance","time_sync_limit_us",
                    "input_causal","complete_accepted_command_history")
        if not isinstance(meta,dict) or any(k not in meta for k in required):
            raise ValueError("missing external log prerequisite for " + request_id)
        if meta["label_provenance"] not in ("measured","fused_estimate","independent_ground_truth","synthetic_fixture"):
            raise ValueError("unknown or future-smoothed label provenance")
        if meta["input_causal"] is not True or meta["complete_accepted_command_history"] is not True:
            raise ValueError("causal input and complete accepted-command history prerequisites required")
        if type(meta["time_sync_limit_us"]) is not int or meta["time_sync_limit_us"]<0:
            raise ValueError("invalid external microsecond sync limit")
        identity = {"robot_id":meta["robot_id"],"map_id":request["field"]["map_id"],"frame_id":request["frame"],
                    "calibration_id":meta["calibration_id"],"source_id":meta["source_id"],
                    "source_epoch":request["source"]["robot_boot_id"],"reset_epoch":str(request["epoch"]),
                    "geometry_revision":request["field"]["geometry_revision"],"localization_revision":request["source"]["localization_revision"],
                    "configuration_revision":request["source"]["configuration_revision"],"sync_mapping_revision":request["sync"]["mapping_revision"],
                    "clock_domain":"ROBOT_MONOTONIC_NS_CONVERTED_FROM_US"}
        common = {"schema_version":LOCAL_SCHEMA_VERSION,"session_id":meta["session_id"],"day":meta["day"],
                  "identity":identity,"time_sync_valid":request["sync"]["valid"],
                  "time_sync_error_ns":request["sync"]["maximum_error_us"]*1000,"time_sync_limit_ns":meta["time_sync_limit_us"]*1000,
                  "wire_provenance":{"request_id":request_id,"model_id":request["model_id"],"snapshot_id":request["snapshot_id"],
                  "candidate_ids":[c["candidate_id"] for c in request["candidates"]],"target_ids":[t["target_id"] for t in request["targets"]],
                  "source":request["source"],"sync":request["sync"],"field":request["field"],"history_cutoff_us":request["history_cutoff_us"],
                  "valid_until_us":request["valid_until_us"],"horizon_us":request["horizon_us"],"step_us":request["step_us"],"timestamp_domain":request["timestamp_domain"]}}
        retained.append({"request_id":request_id,"wire_sha256":canonical_hash(request),"metadata_sha256":canonical_hash(meta),
                         "original_request":request,"external_metadata":meta})
        episode = canonical_hash([meta["session_id"],meta["day"],identity])
        for entry in request["history"]:
            snapshot = entry["snapshot"]
            ego = snapshot["ego"]
            snapshot_id = str(snapshot["id"])
            capture = meta["ego_capture_us_by_snapshot"].get(snapshot_id)
            if type(capture) is not int or not 0 <= capture <= ego["time_us"]:
                raise ValueError("missing or invalid ego capture time for snapshot " + snapshot_id)
            theta = ego["field_pose"]["heading_rad"]
            p, v = ego["field_pose"]["position_m"],ego["field_velocity"]["linear_mps"]
            c,s = math.cos(theta),math.sin(theta)
            valid = entry["valid_mask"] and ego["localization_valid"]
            vector = [p["x"],p["y"],theta,c*v["x"]+s*v["y"],-s*v["x"]+c*v["y"],ego["field_velocity"]["angular_radps"]]
            state = {**common,"record_id":episode+"-state-"+snapshot_id,"kind":"state",
                           "state_time_ns":ego["time_us"]*1000,"capture_time_ns":capture*1000,
                           "input_causal":True,"label_provenance":meta["label_provenance"],"layout_hash":canonical_hash(asdict(EGO_LAYOUT)),
                           "values":vector if valid else [None]*6,"valid":[valid]*6}
            _retain(states,(episode,snapshot_id),state)
            action = entry["accepted_command"]
            if action is None:
                continue
            if action["mechanisms"]:
                raise ValueError("ego-only converter cannot discard configured mechanism inputs")
            twist = action["body_twist"]
            record = {**common,"record_id":episode+"-action-"+action["command_id"],"kind":"action",
                      "accepted_time_ns":action["accepted_us"]*1000,"accepted":True,
                      "layout_hash":canonical_hash(asdict(COMMAND_LAYOUT)),"values":[twist["vx_mps"],twist["vy_mps"],twist["omega_radps"]],
                      "valid":[True]*3,"execution":{"issued_us":action["issued_us"],"command_id":action["command_id"],
                      "source_id":action["source_id"],"authority":"ACCEPTED","interval_semantics":"held accepted command until next accepted command; not measured execution"}}
            key = (episode,action["command_id"])
            _retain(commands,key,record)
    grouped = {}
    for (episode,_),record in commands.items():
        grouped.setdefault(episode,[]).append(record)
    actions = []
    unknown_terminal = 0
    for records in grouped.values():
        ordered = sorted(records,key=lambda r:(r["accepted_time_ns"],r["record_id"]))
        for index,record in enumerate(ordered):
            if index+1 == len(ordered):
                unknown_terminal += 1
                continue
            record["interval_end_ns"] = ordered[index+1]["accepted_time_ns"]
            if record["interval_end_ns"] <= record["accepted_time_ns"]:
                raise ValueError("ambiguous same-time accepted commands")
            actions.append(record)
    result = list(states.values())+actions
    result.sort(key=lambda r:(r["session_id"],r["day"],r.get("state_time_ns",r.get("accepted_time_ns")),r["record_id"]))
    report = {"source_path":str(Path(path).resolve()),"source_sha256":hashlib.sha256(original).hexdigest(),"source_bytes":len(original),"request_lines":line_count,
              "retained_wire_provenance":retained,"state_records":len(states),"action_records":len(actions),
              "terminal_commands_excluded_unknown_interval":unknown_terminal,"output_schema":LOCAL_SCHEMA_VERSION,
              "state_layout":asdict(EGO_LAYOUT),"action_layout":asdict(COMMAND_LAYOUT),
              "not_available_in_v1":["mechanism state","pickup outcomes","execution deviations","clipping/saturation outcomes"],
              "claims":"causal conversion plumbing; accepted commands are not measured outcomes"}
    return result,report


def _retain(records,key,record):
    """Dedup physical facts; request context must not erase repeated snapshots."""
    context = record.pop("wire_provenance")
    record["wire_provenance"] = {"requests":[context]}
    if key not in records:
        records[key] = record
        return
    old = records[key]
    excluded = {"wire_provenance","time_sync_error_ns","time_sync_limit_ns"}
    if {k:v for k,v in old.items() if k not in excluded} != {k:v for k,v in record.items() if k not in excluded}:
        raise ValueError("conflicting physical state/accepted command identity")
    contexts = old["wire_provenance"]["requests"]
    if context not in contexts: contexts.append(context)
    contexts.sort(key=lambda p:(p["request_id"],canonical_hash(p)))
    old["time_sync_error_ns"] = max(old["time_sync_error_ns"],record["time_sync_error_ns"])
    old["time_sync_limit_ns"] = min(old["time_sync_limit_ns"],record["time_sync_limit_ns"])
    if old["time_sync_error_ns"]>old["time_sync_limit_ns"]:
        raise ValueError("conflicting synchronization tolerance across request contexts")
