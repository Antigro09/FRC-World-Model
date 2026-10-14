#!/usr/bin/env python3
"""Serve baseline-only localhost NDJSON for explicit synthetic replay/shadow checks.

This tool never creates an NT client or contacts a camera/controller. The initial
fixture supplies authority and a fixed replay clock, not production robot time.
Every client must copy the new boot service_epoch into its saved request and
validate the reply. No authorization for hardware integration is implied.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frc_world_model.decision_bindings import authority_for, canonical_json, load_decision, load_policy
from frc_world_model.shadow_runtime import ShadowRuntime
from frc_world_model.transport import loopback_server
from frc_world_model.vision_inbox import VisionInbox, VisionSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    request = load_decision(args.request.read_text(), max_bytes=policy.max_bytes)
    # A declared synthetic source only; this starts no subscriptions.
    inbox = VisionInbox([VisionSource("/CustomVision", "front_tags", "apriltag")])
    now = request["prediction_request"]["issued_us"]
    runtime = ShadowRuntime(policy, authority=authority_for(request), now_us=lambda: now, inbox=inbox)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    trace_stream = None
    try:
        if args.trace:
            trace_stream = args.trace.open("x")
        def handler(payload):
            result = runtime(payload)
            if trace_stream is not None:
                trace_stream.write(canonical_json(runtime.latest_trace()) + "\n")
                trace_stream.flush()
            return result
        with loopback_server(handler, max_bytes=policy.max_bytes) as address:
            descriptor = {"host": address[0], "port": address[1], "service_epoch": runtime.service_epoch,
                          "schema_version": "frc-shadow-endpoint/1", "baseline_only": True,
                          "clock": "explicit_fixed_fixture_robot_us", "now_us": now,
                          "influences_commands": False}
            # New output paths only. A restart never overwrites an earlier boot.
            with args.descriptor.open("x") as stream:
                stream.write(canonical_json(descriptor) + "\n")
            print(json.dumps(descriptor), flush=True)
            while not stop.wait(.1):
                pass
    finally:
        runtime.close()
        if trace_stream is not None:
            trace_stream.close()


if __name__ == "__main__":
    main()
