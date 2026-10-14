"""Actual spawned child lifecycle/IPC checks; all model outputs are synthetic."""
from concurrent.futures import ThreadPoolExecutor
import os
import signal
import threading
import time

import pytest

from frc_world_model.decision_bindings import authority_for, canonical_json
from frc_world_model.decision_service import DecisionService, LearnedEvidence
from frc_world_model.shadow_process import ProcessPredictor
from tools.benchmark_decisions import fixture_policy, fixture_request


def provider(request):
    if request.get("stall"):
        time.sleep(5)
    if request.get("crash"):
        os._exit(17)
    if request.get("large"):
        return LearnedEvidence({"x": "x" * 20_000}, False, "unqualified")
    return LearnedEvidence({"echo": request}, False, "unqualified")


def stalled_wire_provider(request):
    time.sleep(5)
    return LearnedEvidence({}, False, "unqualified")


def test_process_success_bounded_timeout_kill_explicit_restart_and_close():
    predictor = ProcessPredictor(provider, timeout_s=.05)
    try:
        assert predictor({"v": 1}, threading.Event()).reply == {"echo": {"v": 1}}
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            predictor({"stall": True}, threading.Event())
        assert time.monotonic() - start < .5
        assert predictor.diagnostics()["alive"] is False
        with pytest.raises(RuntimeError, match="explicit restart"):
            predictor({}, threading.Event())
        predictor.restart()
        assert predictor({"v": 2}, threading.Event()).reply["echo"]["v"] == 2
        os.kill(predictor.diagnostics()["pid"], signal.SIGKILL)
        with pytest.raises(RuntimeError):
            predictor({}, threading.Event())
        predictor.restart()
        assert predictor({}, threading.Event()).domain_id == "unqualified"
    finally:
        predictor.close()
    assert predictor.diagnostics()["alive"] is False
    with pytest.raises(RuntimeError, match="stopped"):
        predictor.restart()


def test_process_overload_cancellation_and_late_results_cannot_be_reused():
    predictor = ProcessPredictor(provider, timeout_s=1)
    cancellation = threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(predictor, {"stall": True}, cancellation)
            deadline = time.monotonic() + 1
            while predictor.diagnostics()["counters"]["calls"] == 0 and time.monotonic() < deadline:
                time.sleep(.001)
            with pytest.raises(RuntimeError, match="overload"):
                predictor({}, threading.Event())
            with pytest.raises(RuntimeError, match="active"):
                predictor.restart()
            cancellation.set()
            with pytest.raises(TimeoutError):
                pending.result(timeout=.5)
        predictor.restart()
        assert predictor({"after_cancel": True}, threading.Event()).reply == {"echo": {"after_cancel": True}}
        assert predictor.diagnostics()["counters"]["cancelled"] == 1
    finally:
        predictor.close()


def test_process_crash_oversized_input_reply_and_provider_exception_fail_closed():
    predictor = ProcessPredictor(provider, timeout_s=.1, max_bytes=4096)
    try:
        with pytest.raises(RuntimeError, match="invalid IPC"):
            predictor({"x": "x" * 5000}, threading.Event())
        predictor.restart()
        with pytest.raises(RuntimeError, match="provider failure"):
            predictor({"large": True}, threading.Event())
        assert predictor({}, threading.Event()).reply == {"echo": {}}
        with pytest.raises(RuntimeError, match="invalid IPC"):
            predictor({"crash": True}, threading.Event())
        assert predictor.diagnostics()["alive"] is False
    finally:
        predictor.close()


def test_stopped_child_cannot_block_large_ipc_send_beyond_deadline():
    predictor = ProcessPredictor(provider, timeout_s=.05)
    try:
        os.kill(predictor.diagnostics()["pid"], signal.SIGSTOP)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            predictor({"payload": "x" * 800_000}, threading.Event())
        assert time.monotonic() - start < .5
        assert predictor.diagnostics()["alive"] is False
    finally:
        predictor.close()


def test_optional_stalled_process_keeps_baseline_advice_bounded():
    predictor = ProcessPredictor(stalled_wire_provider, timeout_s=.03)
    policy = fixture_policy(inference_timeout_s=.01, computation_budget_s=.25)
    request = fixture_request()
    service = DecisionService(policy, authority=authority_for(request), now_us=lambda: 1_010_000, predictor=predictor)
    request["service_epoch"] = service.service_epoch
    try:
        start = time.monotonic()
        reply = service.evaluate(canonical_json(request))
        assert time.monotonic() - start < .4
        assert reply["status"] == "BASELINE" and reply["influences_commands"] is False
        assert reply["learned_status"] == "learned-timeout"
        deadline = time.monotonic() + .5
        while predictor.diagnostics()["alive"] and time.monotonic() < deadline:
            time.sleep(.005)
        assert predictor.diagnostics()["alive"] is False
    finally:
        service.close()
        predictor.close()
