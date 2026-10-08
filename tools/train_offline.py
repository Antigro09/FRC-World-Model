#!/usr/bin/env python3
"""Bounded model-only CPU run on an explicitly supplied authorized local log."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frc_world_model.data import FeatureLayout, SplitConfig, WindowConfig, load_jsonl, prepare_offline
from frc_world_model.evaluation import EgoLayout, AnalyticalConfig, MechanismResponse, analytical_response_forecast, evaluate_windows, persistence_forecast, constant_velocity_forecast
from frc_world_model.rwm import ModelConfig
from frc_world_model.training import train_bounded, predictor_callback, write_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    state_layout = FeatureLayout.from_dict(cfg["state_layout"])
    action_layout = FeatureLayout.from_dict(cfg["action_layout"])
    report = load_jsonl(args.log, state_layout, action_layout)
    split_cfg = cfg.get("split", {})
    split, windows, state_norm, action_norm = prepare_offline(report, state_layout, action_layout,
        WindowConfig(**cfg["window"]), SplitConfig(tuple(split_cfg.get("ratios", (.6,.15,.15,.1))),
        split_cfg.get("seed", "frc-world-model-v1"), split_cfg.get("group_by", "day")))
    model_cfg = ModelConfig(**cfg["model"])
    model, training = train_bounded(windows["train"].windows, state_norm, action_norm, model_cfg,
                                    max_updates=args.max_updates, batch_size=args.batch_size)
    ego = EgoLayout(**cfg["ego"]) if cfg.get("ego") else None
    predictor = predictor_callback(model, model_cfg, state_norm, action_norm)
    analytical = None
    if cfg.get("analytical"):
        ac = dict(cfg["analytical"])
        ac["ego"] = ego
        ac["mechanisms"] = tuple(MechanismResponse(**v) for v in ac.get("mechanisms", ()))
        analytical = AnalyticalConfig(**ac)
        analytical.validate(state_layout, action_layout)
    evaluations = {}
    # No final-test access here; thresholds/config must be frozen first.
    for name in ("validation", "calibration"):
        ws = windows[name].windows
        evaluations[name] = {"rwm": evaluate_windows(ws, predictor, state_layout, ego=ego, name="rwm"),
                             "persistence": evaluate_windows(ws, persistence_forecast, state_layout, ego=ego, name="persistence")}
        if ego:
            evaluations[name]["constant_velocity"] = evaluate_windows(ws, lambda x: constant_velocity_forecast(x, ego), state_layout, ego=ego, name="constant_velocity")
        if analytical:
            evaluations[name]["analytical"] = evaluate_windows(ws, lambda x: analytical_response_forecast(x, analytical), state_layout, ego=ego, name="analytical")
    manifest = {"ingestion": report.manifest(), "split": split.manifest(),
                "windows": {k:v.manifest() for k,v in windows.items()}, "configuration": cfg}
    training["evaluation"] = evaluations
    packaged = write_package(args.output, model, model_cfg, state_norm, action_norm, manifest, training)
    print(json.dumps({"output": str(args.output.resolve()), "model_id": packaged["model_id"],
                      "updates": training["updates"], "final_test": "uninspected", "advisory_only": True}))


if __name__ == "__main__":
    main()
