"""Optional pinned native NT4 localhost test; no external host or hardware."""
import json
import socket
import time

import pytest

from frc_world_model.vision_inbox import VisionInbox
from frc_world_model.vision_nt4 import PyntcoreAdapter, VisionSubscription
from tests.test_vision_inbox import SOURCE, packet

ntcore = pytest.importorskip("ntcore", reason="optional pinned vision_nt4 extra is not installed")


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("synthetic localhost NT4 condition timeout")


def test_native_distinct_consumers_real_server_metadata_and_reconnect(tmp_path):
    server = ntcore.NetworkTableInstance.create()
    port3, port4 = unused_port(), unused_port()
    server.startServer(str(tmp_path / "nt.json"), "127.0.0.1", port3, port4)
    direct = PyntcoreAdapter([SOURCE], server="127.0.0.1", port=port4, identity="synthetic-direct-consumer")
    model = PyntcoreAdapter([SOURCE], server="127.0.0.1", port=port4, identity="synthetic-model-consumer")
    inbox = VisionInbox([SOURCE], receipt_timeout_ns=500_000_000)
    subscription = VisionSubscription(model, inbox)
    publisher = server.getStringTopic(SOURCE.topic).publish(ntcore.PubSubOptions(keepDuplicates=True, sendAll=True))
    try:
        wait_until(lambda: direct.connection()[0] and model.connection()[0])
        wait_until(lambda: publisher.getTopic().exists())
        subscription.step()
        # Connection establishment and native subscription acknowledgement are
        # asynchronous. Publish distinct frames until both have usable evidence.
        direct_packets = []
        for seq in range(1, 21):
            publisher.set(json.dumps(packet(seq)))
            server.flush()
            time.sleep(.02)
            direct_packets.extend(direct.read_queue(SOURCE))
            subscription.step()
            if inbox.diagnostics()["counters"].get("observed", 0) and len(direct_packets) >= 2:
                break
        assert direct_packets and all(stamp > 1 for _, stamp in direct_packets)
        assert inbox.diagnostics()["counters"]["observed"] >= 1
        prior_epoch = model.connection()[1]
        model._instance.stopClient()
        wait_until(lambda: not model.connection()[0])
        subscription.step()
        assert inbox.diagnostics()["sources"][SOURCE.identity]["status"] == "disconnected"
        # Stopping the model client's native connection leaves the other client's
        # subscription and publication metadata active.
        publisher.set(json.dumps(packet(100))); server.flush(); time.sleep(.03)
        assert any(json.loads(payload)["packet_seq"] == 100 for payload, _ in direct.read_queue(SOURCE))
        model._instance.startClient4("synthetic-model-consumer-restarted")
        wait_until(lambda: model.connection()[0] and model.connection()[1] > prior_epoch)
        subscription.step()
        assert inbox.diagnostics()["sources"][SOURCE.identity]["status"] != "observed"
        observed_before = inbox.diagnostics()["counters"].get("observed", 0)
        for seq in range(101, 121):
            publisher.set(json.dumps(packet(seq))); server.flush(); time.sleep(.02)
            subscription.step()
            if inbox.diagnostics()["counters"].get("observed", 0) > observed_before:
                break
        assert inbox.diagnostics()["counters"]["observed"] > observed_before
    finally:
        subscription.close(); direct.close(); publisher.close()
        server.stopServer(); ntcore.NetworkTableInstance.destroy(server)
