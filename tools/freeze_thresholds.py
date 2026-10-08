#!/usr/bin/env python3
"""Freeze user-supplied physical tolerances before final-test inspection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frc_world_model.calibration import freeze_thresholds
from frc_world_model.service import FrozenPackage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("package",type=Path)
    p.add_argument("--physical-tolerances",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--quantile",type=float,default=.95)
    args=p.parse_args()
    policy=freeze_thresholds(FrozenPackage.load(args.package),json.loads(args.physical_tolerances.read_text()),quantile=args.quantile)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as out:json.dump(policy,out,indent=2,sort_keys=True,allow_nan=False);out.write("\n")
    print(json.dumps({"output":str(args.output.resolve()),"calibration_hash":policy["calibration_hash"],"advisory_only":True}))


if __name__=="__main__": main()
