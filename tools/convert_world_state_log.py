#!/usr/bin/env python3
"""Convert authorized request logs using exact v1 validation and explicit metadata."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frc_world_model.bindings import ContractError,parse_request
from frc_world_model.converter import convert_requests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("log",type=Path)
    p.add_argument("--metadata",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    args=p.parse_args()
    metadata=json.loads(args.metadata.read_text())
    def decode(text):
        raw=json.loads(text)
        meta=metadata.get(raw.get("request_id"),{})
        limit=meta.get("time_sync_limit_us")
        if type(limit) is not int: raise ContractError("external microsecond sync limit missing")
        # Replay validates at the recorded issue time, never treats replay as live.
        return parse_request(text,now_us=raw.get("issued_us"),max_sync_error_us=limit)
    records,report=convert_requests(args.log,metadata,decode=decode)
    args.output_directory.mkdir(parents=True,exist_ok=False)
    (args.output_directory/"records.jsonl").write_text("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in records))
    (args.output_directory/"conversion.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output_directory.resolve()),"state_records":report["state_records"],"action_records":report["action_records"]}))


if __name__=="__main__": main()
