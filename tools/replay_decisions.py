#!/usr/bin/env python3
"""Replay advisory decisions using explicit fixture policy; no robot transport."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frc_world_model.decision_bindings import authority_for, canonical_json, load_decision, load_policy
from frc_world_model.decision_service import DecisionService, safe_decision_roundtrip
from frc_world_model.transport import loopback_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loopback", action="store_true")
    args = parser.parse_args()
    policy = load_policy(args.policy)
    request = load_decision(args.request.read_text(), max_bytes=policy.max_bytes)
    now = lambda: request["prediction_request"]["issued_us"]
    service = DecisionService(policy, authority=authority_for(request), now_us=now)
    request["service_epoch"] = service.service_epoch
    try:
        if args.loopback:
            with loopback_server(service, max_bytes=policy.max_bytes) as address:
                reply = safe_decision_roundtrip(address, request, policy=policy, now_us=now,
                                                current_request=request, timeout_s=.5)
        else:
            reply = service.evaluate(canonical_json(request))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(canonical_json({"request": request, "reply": reply}) + "\n")
        print(json.dumps({"output": str(args.output.resolve()), "status": reply["status"],
            "selected_task_id": reply["selected_task_id"], "learned_status": reply["learned_status"],
            "influences_commands": False, "physical_qualification": "unavailable"}))
        if reply["status"] in ("REJECTED", "CANCELLED"):
            raise SystemExit(2)
    finally:
        service.close()


if __name__ == "__main__":
    main()
