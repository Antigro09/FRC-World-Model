"""Optional independent NT4 subscription; no camera, HAL or publisher access."""
from collections import Counter
from importlib import metadata
import threading
import time
import math

from .vision_inbox import VisionReceipt, VisionSource


class PyntcoreAdapter:
    """Own a distinct NT client and subscriptions. Constructing is explicit I/O.

    Only the pinned 2024 Python microsecond metadata profile is supported here;
    NT4 protocol can share a robot server with independently versioned clients.
    Missing dependency, connection-listener or timestamp API fails closed.
    """
    def __init__(self, sources, *, server, port=5810, identity="FRC-World-Model-shadow",
                 poll_storage=32, periodic_s=.01, ntcore_module=None, package_version=None):
        if (not server or not isinstance(server, str) or len(server) > 256
                or type(port) is not int or not 1 <= port <= 65535
                or type(poll_storage) is not int or not 2 <= poll_storage <= 128
                or type(periodic_s) not in (int, float) or not math.isfinite(periodic_s)
                or not .005 <= periodic_s <= 1):
            raise ValueError("explicit bounded NT4 connection configuration required")
        if not identity or not isinstance(identity, str) or len(identity) > 128:
            raise ValueError("independent NT4 client identity required")
        sources = tuple(sources)
        if (not 1 <= len(sources) <= 16 or any(type(source) is not VisionSource for source in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError("bounded unique independent vision sources required")
        if ntcore_module is None:
            try:
                import ntcore as ntcore_module
            except ImportError as error:
                raise RuntimeError("optional NT4 adapter requires the pinned vision_nt4 extra (pyntcore); offline shadow needs none") from error
        version = package_version or metadata.version("pyntcore")
        if version != "2024.3.2.1":
            raise RuntimeError("unverified pyntcore timestamp profile; expected 2024.3.2.1")
        self.poll_storage, self._module = poll_storage, ntcore_module
        self._lock, self._connected, self._epoch = threading.Lock(), False, 0
        self._closed, self._subscribers = False, {}
        self._instance = ntcore_module.NetworkTableInstance.create()
        self._listener = None
        try:
            # A listener fences rapid reconnects that polling isConnected can miss.
            self._listener = self._instance.addConnectionListener(True, self._connection_event)
            options = ntcore_module.PubSubOptions(periodic=periodic_s, sendAll=True,
                                                  keepDuplicates=True, pollStorage=poll_storage)
            for source in sources:
                self._subscribers[source] = self._instance.getStringTopic(source.topic).subscribe("", options)
            self._instance.setServer(server, port)
            self._instance.startClient4(identity)
        except Exception as error:
            self.close()
            raise RuntimeError("NT4 subscription/listener profile unavailable") from error

    def _connection_event(self, event):
        flags = event.flags
        with self._lock:
            if self._closed:
                return
            if flags & self._module.EventFlags.kDisconnected:
                self._connected, self._epoch = False, self._epoch + 1
            if flags & self._module.EventFlags.kConnected:
                self._connected, self._epoch = True, self._epoch + 1

    def connection(self):
        with self._lock:
            if self._closed:
                return False, self._epoch
            actual = self._instance.isConnected()
            if type(actual) is not bool:
                raise RuntimeError("invalid NT4 connection state")
            if actual != self._connected:
                self._connected, self._epoch = actual, self._epoch + 1
            return self._connected, self._epoch

    def read_queue(self, source):
        # Native polling storage bounds this drain; pump detects a full batch as
        # possible lost packets, rather than pretending NT delivers losslessly.
        if self._closed:
            return ()
        values = self._subscribers[source].readQueue()
        # A local NT timestamp cannot stand in for server publication metadata.
        # Missing serverTime remains zero and is rejected by the inbox.
        return tuple((value.value, value.serverTime) for value in values)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed, self._connected = True, False
            self._epoch += 1
        cleanup = [subscriber.close for subscriber in self._subscribers.values()]
        if self._listener is not None:
            cleanup.append(lambda: self._instance.removeListener(self._listener))
        cleanup.extend((self._instance.stopClient,
                        lambda: self._module.NetworkTableInstance.destroy(self._instance)))
        for action in cleanup:
            try:
                action()
            except Exception:
                # Attempt every owned resource even after partial startup.
                pass


class VisionSubscription:
    """Injected adapter seam; vision polling never waits for model completion.

    adapter.connection() returns verified (connected, connection_epoch).
    adapter.read_queue(source) returns ordered (coherent_string, original NT-us)
    pairs with a hard native poll_storage bound. A full batch is fenced/dropped.
    """
    def __init__(self, adapter, inbox, *, quantum=16, periodic_s=.01, now_ns=time.monotonic_ns):
        if (type(quantum) is not int or not 1 <= quantum <= 128
                or type(periodic_s) not in (int, float) or not math.isfinite(periodic_s)
                or not .001 <= periodic_s <= 1):
            raise ValueError("bounded subscription quantum/period required")
        if type(adapter.poll_storage) is not int or not 2 <= adapter.poll_storage <= 128:
            raise ValueError("bounded adapter polling storage required")
        self.adapter, self.inbox, self.quantum = adapter, inbox, quantum
        self.periodic_s, self.now_ns = periodic_s, now_ns
        self._stop, self._thread, self._counts = threading.Event(), None, Counter()
        self._closed, self._counter_lock = False, threading.Lock()

    def _count(self, reason, amount=1):
        with self._counter_lock:
            self._counts[reason] += amount

    def step(self):
        if self._closed:
            return ()
        try:
            connected, epoch = self.adapter.connection()
            self.inbox.connection(connected, epoch, now_ns=self.now_ns())
            if connected:
                for source in self.inbox.sources:
                    batch = self.adapter.read_queue(source)
                    if len(batch) >= self.adapter.poll_storage:
                        self.inbox.loss(source, "native_queue_saturation", now_ns=self.now_ns())
                        self._count("dropped_native_packets", len(batch))
                        continue
                    for payload, stamp in batch:
                        self.inbox.offer(VisionReceipt(source, payload, self.now_ns(), stamp, epoch))
                # Fence reconnects occurring during the native queue drain.
                latest_connected, latest_epoch = self.adapter.connection()
                self.inbox.connection(latest_connected, latest_epoch, now_ns=self.now_ns())
            return self.inbox.process(self.quantum)
        except Exception:
            for source in self.inbox.sources:
                self.inbox.loss(source, "subscription_failure", now_ns=self.now_ns())
            self._count("subscription_failure")
            return ()

    def start(self):
        if self._closed or self._thread is not None:
            raise RuntimeError("subscription already started")
        self._thread = threading.Thread(target=self._run, name="model-vision-inbox", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.step()
            self._stop.wait(self.periodic_s)

    def diagnostics(self):
        with self._counter_lock:
            counts = dict(self._counts)
        return {"running": self._thread is not None and self._thread.is_alive(),
                "quantum": self.quantum, "counters": counts, "inbox": self.inbox.diagnostics()}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.1)
        self.adapter.close()
        self.inbox.close()
