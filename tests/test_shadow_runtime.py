"""Baseline-only runtime and actual localhost process boot/restart checks."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from frc_world_model.bindings import ContractError
from frc_world_model.decision_bindings import authority_for, canonical_json, consume_decision_reply, load_policy
from frc_world_model.shadow_runtime import ShadowRuntime
from frc_world_model.transport import NetworkFailure, roundtrip
from frc_world_model.vision_inbox import VisionInbox
from frc_world_model.vision_nt4 import VisionSubscription
from tests.test_vision_inbox import Clock, SOURCE, ready
from tests.test_vision_subscription import Adapter, publish
from tools.benchmark_decisions import fixture_policy, fixture_request

ROOT = Path(__file__).resolve().parents[1]


def test_direct_vision_is_immutable_trace_evidence_not_second_ego_or_fabricated_tracks():
    request, clock = fixture_request(), Clock()
    inbox = VisionInbox([SOURCE], now_ns=clock, map_capture=lambda source, capture: capture)
    inbox.connection(True, 1)
    ready(inbox, clock)
    runtime = ShadowRuntime(fixture_policy(), authority=authority_for(request), now_us=lambda: 1_010_000, inbox=inbox)
    request["service_epoch"] = runtime.service_epoch
    original = copy.deepcopy(request)
    try:
        reply = runtime.evaluate(canonical_json(request))
        assert reply["status"] == "BASELINE" and reply["influences_commands"] is False
        assert reply["learned_prediction_reply"] is None
        consume_decision_reply(reply, request, policy=fixture_policy(), now_us=1_010_000, current_request=request)
        assert request == original
        trace = runtime.latest_trace()
        assert trace["observations"][0]["observed"] is True
        assert trace["vision_is_shadow_evidence"] is True and trace["influences_commands"] is False
        trace["robot_source"]["robot_boot_id"] = "forged"
        assert runtime.latest_trace()["robot_source"] == request["prediction_request"]["source"]
    finally:
        runtime.close()
    with pytest.raises(ContractError, match="stopped"):
        runtime.evaluate(canonical_json(request))


def wait_descriptor(path, process):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            # Descriptor publication can race the final write; only a complete
            # parse is a ready boot, rather than assuming file existence.
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        if process.poll() is not None:
            raise AssertionError(process.communicate(timeout=1))
        time.sleep(.01)
    raise AssertionError("shadow process startup timeout")


def start_server(path):
    process = subprocess.Popen([sys.executable, str(ROOT / "tools/serve_shadow.py"),
        "--request", str(ROOT / "contracts/decision-support-v1/request-v1.json"),
        "--policy", str(ROOT / "configs/decision-synthetic.json"), "--descriptor", str(path)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return process, wait_descriptor(path, process)


def test_actual_shadow_process_stall_kill_restart_never_blocks_separate_vision_queue(tmp_path):
    request = fixture_request()
    policy = load_policy(ROOT / "configs/decision-synthetic.json")
    process, descriptor = start_server(tmp_path / "boot-a.json")
    clock, adapter = Clock(), Adapter()
    direct = VisionInbox([SOURCE], now_ns=clock)
    subscription = VisionSubscription(adapter, direct, now_ns=clock)
    subscription.step()
    address = descriptor["host"], descriptor["port"]
    request["service_epoch"] = descriptor["service_epoch"]
    try:
        reply = json.loads(roundtrip(address, canonical_json(request), timeout_s=.5))
        assert reply["status"] == "BASELINE"
        consume_decision_reply(reply, request, policy=policy, now_us=descriptor["now_us"], current_request=request)
        os.kill(process.pid, signal.SIGSTOP)
        with __import__('concurrent.futures').futures.ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(roundtrip, address, canonical_json(request), timeout_s=.05)
            for seq in range(1, 31):
                clock.advance(); publish(adapter, seq); subscription.step()
            assert direct.diagnostics()["counters"]["observed"] == 29
            with pytest.raises(NetworkFailure):
                pending.result(timeout=.5)
        process.kill(); process.wait(timeout=1)
        clock.advance(); publish(adapter, 31)
        assert len(subscription.step()) == 1
        old_epoch = descriptor["service_epoch"]
        restarted, descriptor = start_server(tmp_path / "boot-b.json")
        process = restarted
        assert descriptor["service_epoch"] != old_epoch
        address = descriptor["host"], descriptor["port"]
        stale = json.loads(roundtrip(address, canonical_json(request), timeout_s=.5))
        assert stale["status"] == "REJECTED" and stale["reason"] == "service-epoch-mismatch"
        request["service_epoch"] = descriptor["service_epoch"]
        fresh = json.loads(roundtrip(address, canonical_json(request), timeout_s=.5))
        assert fresh["status"] == "BASELINE"
        for seq in range(32, 42):
            clock.advance(); publish(adapter, seq); assert len(subscription.step()) == 1
        assert direct.diagnostics()["counters"]["observed"] == 40
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
        process.communicate(timeout=1)
        subscription.close()
