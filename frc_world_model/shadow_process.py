"""Killable optional inference with bounded JSON IPC; no actuator interfaces."""
import json
import math
import multiprocessing
import select
import socket
import threading
import time

from .bindings import ContractError, _finite, _pairs
from .decision_bindings import canonical_json
from .decision_service import LearnedEvidence


def _encode(message, maximum):
    try:
        data = canonical_json(message).encode("utf-8") + b"\n"
    except (TypeError, ValueError, RecursionError) as error:
        raise ContractError("invalid process JSON") from error
    if len(data) > maximum:
        raise ContractError("oversized process JSON")
    return data


def _decode(data):
    try:
        result = json.loads(data, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite process JSON")))
        _finite(result)
        return result
    except (ValueError, TypeError, RecursionError) as error:
        raise ContractError("invalid process JSON") from error


def _serve_predictor(channel, provider, maximum):
    """Trusted explicit provider; bounded framing on both sides of the process."""
    try:
        channel.setblocking(True)
        channel.sendall(_encode({"ready": True}, maximum))
        with channel.makefile("rb") as stream:
            while True:
                data = stream.readline(maximum + 1)
                if not data:
                    return
                if len(data) > maximum or not data.endswith(b"\n"):
                    return
                message = _decode(data)
                generation, request = message["generation"], message["request"]
                try:
                    result = provider(request)
                    if not isinstance(result, LearnedEvidence):
                        raise TypeError("LearnedEvidence required")
                    reply = {"generation": generation, "success": True, "reply": result.reply,
                             "in_domain": result.in_domain, "domain_id": result.domain_id}
                    wire = _encode(reply, maximum)
                except Exception as error:
                    wire = _encode({"generation": generation, "success": False,
                                    "error": type(error).__name__}, maximum)
                channel.sendall(wire)
    except (EOFError, BrokenPipeError, OSError, ContractError, KeyError, TypeError):
        return
    finally:
        channel.close()


class ProcessPredictor:
    """One spawned provider process and one caller; no blocking Pipe.send/recv.

    IPC writes, reads, cancellation, and startup have wall-clock deadlines.
    Timeout/crash cancels this child and requires explicit restart outside the
    request deadline. Child cleanup has a separate <=100ms join/kill budget.
    The provider is an explicitly supplied trusted picklable module function.
    No model is loaded by default. OS memory limits and process-tree supervision
    remain prerequisites for later hardware qualification.
    """
    def __init__(self, provider, *, timeout_s=.02, startup_timeout_s=3,
                 max_bytes=1_000_000, context=None):
        if (not callable(provider) or type(timeout_s) not in (int, float)
                or not math.isfinite(timeout_s) or not 0 < timeout_s <= 1
                or type(startup_timeout_s) not in (int, float)
                or not math.isfinite(startup_timeout_s) or not 0 < startup_timeout_s <= 5
                or type(max_bytes) is not int or not 256 <= max_bytes <= 1_000_000):
            raise ValueError("bounded explicit process predictor configuration required")
        self.provider, self.timeout_s, self.startup_timeout_s = provider, timeout_s, startup_timeout_s
        self.max_bytes = max_bytes
        self._context = context or multiprocessing.get_context("spawn")
        self._slot, self._state_lock = threading.Lock(), threading.RLock()
        self._process, self._channel, self._generation = None, None, 0
        self._closed = False
        self._counts = {"calls": 0, "timeouts": 0, "cancelled": 0, "crashes": 0, "overload": 0}
        self._start()

    def _receive(self, channel, deadline, cancellation=None):
        data = bytearray()
        while True:
            if cancellation is not None and cancellation.is_set():
                raise TimeoutError("process predictor cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("process predictor deadline")
            if not select.select([channel], [], [], min(.005, remaining))[0]:
                continue
            chunk = channel.recv(min(65536, self.max_bytes + 1 - len(data)))
            if not chunk:
                raise EOFError("process predictor disconnected")
            data.extend(chunk)
            if len(data) > self.max_bytes:
                raise ContractError("oversized process reply")
            if b"\n" in data:
                if not data.endswith(b"\n") or data.count(b"\n") != 1:
                    raise ContractError("invalid process reply frame")
                return _decode(data[:-1])

    def _send(self, channel, data, deadline, cancellation):
        offset = 0
        while offset < len(data):
            if cancellation.is_set():
                raise TimeoutError("process predictor cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("process predictor deadline")
            if not select.select([], [channel], [], min(.005, remaining))[1]:
                continue
            try:
                written = channel.send(memoryview(data)[offset:])
            except BlockingIOError:
                continue
            if written <= 0:
                raise EOFError("process predictor disconnected")
            offset += written

    def _start(self):
        with self._state_lock:
            if self._closed:
                raise RuntimeError("process predictor stopped")
            parent, child = socket.socketpair()
            process = self._context.Process(target=_serve_predictor,
                args=(child, self.provider, self.max_bytes), daemon=True, name="optional-shadow-predictor")
            try:
                process.start()
            except Exception:
                parent.close()
                child.close()
                raise
            child.close()
            parent.setblocking(False)
            self._process, self._channel = process, parent
        try:
            if self._receive(parent, time.monotonic() + self.startup_timeout_s) != {"ready": True}:
                raise RuntimeError("process predictor startup protocol")
        except Exception:
            self._terminate()
            raise RuntimeError("process predictor startup failed") from None

    def _terminate(self):
        with self._state_lock:
            process, channel = self._process, self._channel
            self._process = self._channel = None
        if channel is not None:
            try:
                channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            channel.close()
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=.05)
            if process.is_alive():
                process.kill()
                process.join(timeout=.05)
            if process.is_alive():
                raise RuntimeError("process predictor could not be reaped")
            process.close()

    def __call__(self, request, cancellation):
        if not self._slot.acquire(blocking=False):
            with self._state_lock:
                self._counts["overload"] += 1
            raise RuntimeError("process predictor overload")
        try:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("process predictor stopped")
                process, channel = self._process, self._channel
                if process is None or not process.is_alive():
                    self._counts["crashes"] += 1
                    self._terminate()
                    raise RuntimeError("process predictor unavailable; explicit restart required")
                self._generation += 1
                generation = self._generation
                self._counts["calls"] += 1
            deadline = time.monotonic() + self.timeout_s
            data = _encode({"generation": generation, "request": request}, self.max_bytes)
            self._send(channel, data, deadline, cancellation)
            result = self._receive(channel, deadline, cancellation)
            if (type(result) is not dict or type(result.get("generation")) is not int
                    or result["generation"] != generation or cancellation.is_set()):
                raise ContractError("process predictor generation mismatch")
            if result.get("success") is not True:
                raise RuntimeError("process predictor provider failure")
            if (set(result) != {"generation", "success", "reply", "in_domain", "domain_id"}
                    or type(result["reply"]) is not dict or type(result["in_domain"]) is not bool
                    or type(result["domain_id"]) is not str or len(result["domain_id"]) > 256):
                raise ContractError("invalid process evidence envelope")
            return LearnedEvidence(result["reply"], result["in_domain"], result["domain_id"])
        except TimeoutError:
            with self._state_lock:
                self._counts["cancelled" if cancellation.is_set() else "timeouts"] += 1
            self._terminate()
            raise
        except (EOFError, BrokenPipeError, OSError, ValueError):
            with self._state_lock:
                self._counts["crashes"] += 1
            self._terminate()
            raise RuntimeError("process predictor crashed or invalid IPC") from None
        finally:
            self._slot.release()

    def restart(self):
        if not self._slot.acquire(blocking=False):
            raise RuntimeError("cannot restart active inference")
        try:
            self._terminate()
            self._start()
        finally:
            self._slot.release()

    def diagnostics(self):
        with self._state_lock:
            return {"alive": self._process is not None and self._process.is_alive(),
                    "pid": self._process.pid if self._process is not None else None,
                    "generation": self._generation, "counters": dict(self._counts)}

    def close(self):
        with self._state_lock:
            self._closed = True
        self._terminate()
