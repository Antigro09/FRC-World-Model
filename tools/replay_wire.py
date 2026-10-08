#!/usr/bin/env python3
"""Exact v1 replay abstention; no learned covariance or missing state is fabricated."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frc_world_model.bindings import make_masked_reply,parse_request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("request",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--maximum-sync-error-us",type=int,required=True)
    args=p.parse_args()
    text=args.request.read_text()
    raw=json.loads(text)
    request=parse_request(text,now_us=raw["issued_us"],max_sync_error_us=args.maximum_sync_error_us)
    reply=make_masked_reply(request,request["issued_us"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as out: json.dump(reply,out,allow_nan=False);out.write("\n")
    print(json.dumps({"output":str(args.output.resolve()),"mode":"replay","influences_commands":False,
                      "reason":"missing approved FRC weights, model prerequisites and physical calibration"}))


if __name__=="__main__": main()
