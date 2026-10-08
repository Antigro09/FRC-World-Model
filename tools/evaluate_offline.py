#!/usr/bin/env python3
"""Replay evaluator. Final test requires previously frozen calibration policy."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frc_world_model.data import FeatureLayout, SplitConfig, WindowConfig, load_jsonl, prepare_offline, canonical_hash
from frc_world_model.evaluation import EgoLayout, AnalyticalConfig, MechanismResponse, analytical_response_forecast, evaluate_windows, persistence_forecast, constant_velocity_forecast, ablate_inputs
from frc_world_model.training import load_frozen_model, predictor_callback
from frc_world_model.calibration import validate_thresholds,assess_thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package",type=Path)
    parser.add_argument("log",type=Path)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--split",choices=("validation","calibration","test"),default="validation")
    parser.add_argument("--thresholds",type=Path)
    args = parser.parse_args()
    package, frozen = load_frozen_model(args.package)
    cfg = json.loads(args.config.read_text())
    expected = package.metadata["dataset"]
    if expected.get("configuration") != cfg:
        raise ValueError("evaluation configuration differs from frozen training configuration")
    thresholds=None
    if args.split == "test":
        if args.thresholds is None:
            raise ValueError("final test requires a frozen calibration-session threshold artifact")
        thresholds = json.loads(args.thresholds.read_text())
        validate_thresholds(thresholds,package)
    state, action = FeatureLayout.from_dict(cfg["state_layout"]), FeatureLayout.from_dict(cfg["action_layout"])
    report = load_jsonl(args.log,state,action)
    if report.original_sha256 != expected["ingestion"]["original"]["sha256"]:
        raise ValueError("evaluation log differs from frozen dataset")
    sp = cfg.get("split",{})
    splits, windows, sn, an = prepare_offline(report,state,action,WindowConfig(**cfg["window"]),
        SplitConfig(tuple(sp.get("ratios",(.6,.15,.15,.1))),sp.get("seed","frc-world-model-v1"),sp.get("group_by","day")))
    ego = EgoLayout(**cfg["ego"]) if cfg.get("ego") else None
    learned = predictor_callback(frozen.model,frozen.config,sn,an)
    methods = {"rwm":learned,"persistence":persistence_forecast}
    if ego:
        methods["constant_velocity"] = lambda x: constant_velocity_forecast(x,ego)
    if cfg.get("analytical"):
        ac = dict(cfg["analytical"])
        ac["ego"] = ego
        ac["mechanisms"] = tuple(MechanismResponse(**v) for v in ac.get("mechanisms",()))
        analytical = AnalyticalConfig(**ac)
        analytical.validate(state,action)
        methods["analytical"] = lambda x: analytical_response_forecast(x,analytical)
    if cfg.get("neutral_action") is not None:
        methods["rwm-action-ablation"] = ablate_inputs(learned,actions=True,neutral_action=cfg["neutral_action"])
    # Remove past evidence with a fixed-length repeat of the cutoff state. This
    # is an explicit evaluation ablation, not training or future teacher forcing.
    from dataclasses import replace
    methods["rwm-history-ablation"] = lambda x: learned(replace(x,history=(x.history[-1],)*len(x.history),history_actions=(x.history_actions[-1],)*len(x.history_actions)))
    result = {"split":args.split,"dataset_hash":report.dataset_hash,"model_id":package.manifest["model_id"],
              "advisory_only":True,"reports":{mode:{name:evaluate_windows(windows[args.split].windows,predict,state,ego=ego,name=name,mode=mode)
                                                  for name,predict in methods.items()}
                                             for mode in ("one_step","open_loop","refreshed")}}
    if thresholds is not None:
        result["frozen_thresholds"]=thresholds
        result["threshold_assessment"]=assess_thresholds(result["reports"]["open_loop"]["rwm"],thresholds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output":str(args.output.resolve()),"split":args.split,"modes":list(result["reports"])}))


if __name__ == "__main__":
    main()
