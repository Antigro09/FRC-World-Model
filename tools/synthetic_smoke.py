#!/usr/bin/env python3
"""Generate small labelled synthetic logs. These prove plumbing only."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frc_world_model.data import FeatureLayout, LOCAL_SCHEMA_VERSION, canonical_hash
from frc_world_model.rwm import ModelConfig


def generate(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    state = FeatureLayout(("x_m", "y_m", "heading_rad", "vx_robot_mps", "vy_robot_mps", "omega_radps", "mechanism_position_m", "mechanism_velocity_mps"),
                          ("m", "m", "rad", "m/s", "m/s", "rad/s", "m", "m/s"), (2,))
    action = FeatureLayout(("vx_robot_mps", "vy_robot_mps", "omega_radps", "mechanism_setpoint_m"), ("m/s", "m/s", "rad/s", "m"))
    config = {"scope": "synthetic-plumbing-only", "state_layout": asdict(state), "action_layout": asdict(action),
              "model": asdict(ModelConfig()), "window": {"timestep_ns": 20_000_000, "history_steps": 32, "horizon_steps": 8},
              "neutral_action": [0.,0.,0.,0.],
              "neutral_action_semantics": "synthetic zero body twist and mechanism 0 m hold target",
              "split": {"ratios": [.25,.25,.25,.25], "seed": "synthetic-v1", "group_by": "day"},
              "ego": {"x":0,"y":1,"heading":2,"vx_robot":3,"vy_robot":4,"omega":5},
              "analytical": {"command_indices":[0,1,2],"velocity_time_constants_s":[.1,.1,.1],
                             "mechanisms":[{"state_index":6,"action_index":3,"time_constant_s":.2}],
                             "calibration_source":"synthetic plumbing constants; not robot calibration"}}
    rows = []
    for session in range(4):
        for tick in range(45):
            time_ns = tick * 20_000_000
            values = [tick*.02, session*.1, .01*session, 1., 0., 0., .001*tick, .05]
            for kind, layout, vector in (("state",state,values), ("action",action,[1.,0.,0.,.1])):
                row = {"schema_version": LOCAL_SCHEMA_VERSION, "record_id":f"synthetic-{session}-{kind}-{tick}",
                       "kind":kind,"session_id":f"synthetic-{session}","day":f"2026-10-0{session+1}",
                       "identity":{"robot_id":"synthetic","map_id":"synthetic","frame_id":"BLUE_FIELD",
                                   "calibration_id":"synthetic-uncalibrated","source_id":"synthetic-generator",
                                   "source_epoch":"0","reset_epoch":"0","geometry_revision":"synthetic-0",
                                   "localization_revision":"0","configuration_revision":"0","sync_mapping_revision":"0","clock_domain":"synthetic-monotonic-ns"},
                       "time_sync_valid":True,"time_sync_error_ns":0,"time_sync_limit_ns":1000,"layout_hash":canonical_hash(asdict(layout)),
                       "values":vector,"valid":[True]*len(vector),"input_causal":True}
                if kind == "state":
                    row.update(state_time_ns=time_ns,capture_time_ns=time_ns,label_provenance="synthetic_fixture")
                else:
                    row.update(accepted_time_ns=time_ns,interval_end_ns=time_ns+20_000_000,accepted=True,
                               execution={"saturated":False,"synthetic":True})
                rows.append(row)
    (root/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    (root/"synthetic.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in rows))
    return root/"synthetic.jsonl", root/"config.json"


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Small synthetic plumbing fixture; no FRC accuracy evidence")
    parser.add_argument("output_directory",type=Path)
    args=parser.parse_args()
    log, config = generate(args.output_directory)
    print(json.dumps({"log":str(log),"config":str(config),"claims":"synthetic plumbing only"}))
