#!/usr/bin/env python3
"""Read-only local-log preparation; no training/device/cloud/robot operations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frc_world_model.data import FeatureLayout, SplitConfig, WindowConfig, load_jsonl, prepare_offline
from frc_world_model.evaluation import EgoLayout, constant_velocity_forecast, evaluate_windows, persistence_forecast


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Explicit local layouts, splits, timing and optional ego index map")
    parser.add_argument("--output", required=True, type=Path, help="New manifest path; originals never modified")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    state_layout, action_layout = FeatureLayout.from_dict(config["state_layout"]), FeatureLayout.from_dict(config["action_layout"])
    split_raw = config.get("split", {})
    split_config = SplitConfig(tuple(split_raw.get("ratios", (0.6, 0.15, 0.15, 0.1))), split_raw.get("seed", "frc-world-model-v1"), split_raw.get("group_by", "day"))
    report = load_jsonl(args.log, state_layout, action_layout)
    split, windows, state_norm, action_norm = prepare_offline(report, state_layout, action_layout, WindowConfig(**config["window"]), split_config)
    manifest = {"ingestion": report.manifest(), "split": split.manifest(), "windows": {name: w.manifest() for name, w in windows.items()},
                "state_normalization": state_norm.metadata(), "action_normalization": action_norm.metadata(),
                "claims": "offline plumbing only; no training or hardware evidence", "evaluation": {}}
    ego = EgoLayout(**config["ego"]) if config.get("ego") else None
    for name, window_report in windows.items():
        # Final test is deliberately not inspected before thresholds are frozen.
        if name == "test":
            continue
        manifest["evaluation"][name] = {"persistence": evaluate_windows(window_report.windows, persistence_forecast, state_layout, ego=ego, name="persistence")}
        if ego:
            manifest["evaluation"][name]["constant_velocity"] = evaluate_windows(window_report.windows, lambda inp: constant_velocity_forecast(inp, ego), state_layout, ego=ego, name="constant_velocity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output creation prevents accidental overwrite of logs or artifacts.
    with args.output.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"manifest": str(args.output.resolve()), "accepted": len(report.records), "windows": {k: len(v.windows) for k, v in windows.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
