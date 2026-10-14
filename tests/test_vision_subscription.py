"""Separate adapter storage, NT metadata profile and subscription lifecycle."""
import json
from types import SimpleNamespace
import threading
import time

import pytest

from frc_world_model.vision_inbox import VisionInbox
from frc_world_model.vision_nt4 import PyntcoreAdapter, VisionSubscription
from tests.test_vision_inbox import Clock, SOURCE, packet


class Adapter:
    poll_storage = 4
    def __init__(self):
        self.connected, self.epoch, self.queue, self.closed = True, 1, [], False
        self.fail = False
    def connection(self):
        if self.fail:
            raise OSError("synthetic disconnected adapter")
        return self.connected, self.epoch
    def read_queue(self, source):
        result, self.queue = tuple(self.queue), []
        return result
    def close(self):
        self.closed = True


def publish(adapter, sequence):
    adapter.queue.append((json.dumps(packet(sequence)), sequence * 10))


def test_separate_subscription_queues_saturation_reconnect_and_close():
    clock = Clock()
    direct, model = Adapter(), Adapter()
    direct_inbox, model_inbox = (VisionInbox([SOURCE], now_ns=clock) for _ in range(2))
    direct_sub, model_sub = (VisionSubscription(adapter, inbox, now_ns=clock)
        for adapter, inbox in ((direct, direct_inbox), (model, model_inbox)))
    direct_sub.step(); model_sub.step()
    for seq in (1, 2):
        clock.advance(); publish(direct, seq); publish(model, seq)
        direct_sub.step(); model_sub.step()
    assert direct_inbox.diagnostics()["sources"][SOURCE.identity]["status"] == "observed"
    for seq in (3, 4, 5, 6):
        publish(model, seq)
    clock.advance(); publish(direct, 3)
    assert len(direct_sub.step()) == 1
    assert model_sub.step() == ()
    assert model_inbox.diagnostics()["sources"][SOURCE.identity]["status"] == "native_queue_saturation"
    model.fail = True
    assert model_sub.step() == ()
    clock.advance(); publish(direct, 4)
    assert len(direct_sub.step()) == 1
    model.fail = False; model.epoch = 3
    model_sub.step()
    for seq in (7, 8):
        clock.advance(); publish(model, seq); result = model_sub.step()
    assert len(result) == 1
    model_sub.close(); model_sub.close()
    assert model.closed and model_sub.step() == ()
    with pytest.raises(RuntimeError):
        model_sub.start()
    clock.advance(); publish(direct, 5)
    assert len(direct_sub.step()) == 1
    direct_sub.close()


def test_thread_lifecycle_stops_without_model_dependency():
    adapter = Adapter(); inbox = VisionInbox([SOURCE])
    subscription = VisionSubscription(adapter, inbox, periodic_s=.002)
    subscription.start()
    assert subscription.diagnostics()["running"]
    with pytest.raises(RuntimeError):
        subscription.start()
    subscription.close()
    assert not subscription.diagnostics()["running"]
    assert adapter.closed and inbox.diagnostics()["sources"][SOURCE.identity]["status"] == "stopped"


def test_reconnect_during_read_fences_batch_instead_of_retained_adoption():
    class Reconnect(Adapter):
        def read_queue(self, source):
            result = super().read_queue(source)
            self.epoch += 2
            return result
    clock, adapter = Clock(), Reconnect()
    inbox = VisionInbox([SOURCE], now_ns=clock)
    subscription = VisionSubscription(adapter, inbox, now_ns=clock)
    subscription.step()
    publish(adapter, 1); publish(adapter, 2)
    clock.advance()
    assert subscription.step() == ()
    assert inbox.diagnostics()["queue_depth"] == 0


class Subscriber:
    def __init__(self):
        self.values = []; self.closed = False
    def readQueue(self):
        values, self.values = self.values, []
        return values
    def close(self):
        self.closed = True


class Instance:
    def __init__(self):
        self.subscriber = Subscriber(); self.connected = False; self.destroyed = False
    def addConnectionListener(self, immediate, callback):
        self.callback = callback
        return 1
    def getStringTopic(self, topic):
        self.topic = topic
        return self
    def subscribe(self, default, options):
        self.options = options
        return self.subscriber
    def setServer(self, server, port):
        self.server, self.port = server, port
    def startClient4(self, identity):
        self.identity = identity
    def isConnected(self):
        return self.connected
    def removeListener(self, listener):
        self.removed = listener
    def stopClient(self):
        self.stopped = True
    def destroy(self):
        self.destroyed = True


def test_pinned_adapter_distinct_instance_listener_epoch_metadata_and_cleanup():
    instances = []
    def create():
        instance = Instance(); instances.append(instance); return instance
    module = SimpleNamespace(NetworkTableInstance=SimpleNamespace(create=create, destroy=lambda instance: instance.destroy()),
        EventFlags=SimpleNamespace(kConnected=1, kDisconnected=2), PubSubOptions=lambda **kwargs: kwargs)
    first, second = (PyntcoreAdapter([SOURCE], server="127.0.0.1", ntcore_module=module,
        package_version="2024.3.2.1") for _ in range(2))
    try:
        assert instances[0] is not instances[1]
        instance = instances[0]
        assert instance.topic == SOURCE.topic and instance.options["pollStorage"] == 32
        assert first.connection() == (False, 0)
        instance.connected = True
        instance.callback(SimpleNamespace(flags=1))
        assert first.connection() == (True, 1)
        instance.callback(SimpleNamespace(flags=2)); instance.callback(SimpleNamespace(flags=1))
        assert first.connection() == (True, 3)
        instance.subscriber.values = [SimpleNamespace(value="one", serverTime=123, time=999),
                                      SimpleNamespace(value="two", serverTime=0, time=888)]
        assert first.read_queue(SOURCE) == (("one", 123), ("two", 0))
        first.close(); first.close()
        assert instance.destroyed and instance.subscriber.closed and instance.removed == 1
        assert first.connection()[0] is False
        instance.callback(SimpleNamespace(flags=1))
        assert first.connection()[0] is False
    finally:
        first.close(); second.close()
    with pytest.raises(RuntimeError, match="unverified"):
        PyntcoreAdapter([SOURCE], server="localhost", ntcore_module=module, package_version="2026.1")
    assert len(instances) == 2  # rejected before any native client I/O
