"""Bounded localhost NDJSON transport for offline/shadow checks.

One serialized daemon serves bounded frames; socket work uses one total exchange
budget. The owner supplies a trusted bounded handler. An uncooperative handler
cannot be forcibly stopped in a thread; actual inference belongs in the killable
process supervisor. Closing never waits indefinitely on a handler or peer.
"""
from contextlib import contextmanager
import json
import math
import socket
import threading
import time


class NetworkFailure(ValueError):
    pass


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise NetworkFailure("exchange budget exhausted")
    return remaining


def _read_frame(channel, maximum, deadline):
    data = bytearray()
    while True:
        channel.settimeout(_remaining(deadline))
        chunk = channel.recv(min(65536, maximum + 1 - len(data)))
        if not chunk:
            raise NetworkFailure("missing or truncated frame")
        data.extend(chunk)
        if len(data) > maximum:
            raise NetworkFailure("oversized frame")
        if b"\n" in data:
            if not data.endswith(b"\n") or data.count(b"\n") != 1:
                raise NetworkFailure("invalid NDJSON frame")
            return data[:-1]


def _write_frame(channel, data, deadline):
    channel.settimeout(_remaining(deadline))
    channel.sendall(data)


def _limits(max_bytes, timeout_s):
    if (type(max_bytes) is not int or not 1 <= max_bytes <= 8_000_000
            or type(timeout_s) not in (int, float) or not math.isfinite(timeout_s)
            or not 0 < timeout_s <= 2):
        raise ValueError("unsupported transport limits")


@contextmanager
def loopback_server(handler, *, max_bytes=1_000_000, socket_timeout_s=.5):
    _limits(max_bytes, socket_timeout_s)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    listener.settimeout(.05)
    address = listener.getsockname()
    stop, lock, active = threading.Event(), threading.Lock(), [None]

    def serve():
        while not stop.is_set():
            try:
                channel, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with channel:
                with lock:
                    if stop.is_set():
                        return
                    active[0] = channel
                deadline = time.monotonic() + socket_timeout_s
                try:
                    request = _read_frame(channel, max_bytes, deadline)
                    reply = handler(request.decode("utf-8"))
                    data = reply.encode("utf-8")
                    if len(data) + 1 > max_bytes or b"\n" in data:
                        raise NetworkFailure("invalid response frame")
                    _write_frame(channel, data + b"\n", deadline)
                except Exception as error:
                    # Stable transport failure, never a fabricated forecast.
                    data = json.dumps({"transport_error": type(error).__name__}, separators=(",", ":")).encode()
                    try:
                        _write_frame(channel, data + b"\n", deadline)
                    except (OSError, NetworkFailure):
                        pass
                finally:
                    with lock:
                        active[0] = None
    thread = threading.Thread(target=serve, daemon=True, name="bounded-shadow-loopback")
    thread.start()
    try:
        yield address
    finally:
        stop.set()
        listener.close()
        with lock:
            if active[0] is not None:
                try:
                    active[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                active[0].close()
        thread.join(timeout=.2)


def roundtrip(address, request: str, *, timeout_s=.5, max_bytes=1_000_000):
    _limits(max_bytes, timeout_s)
    if (type(address) not in (tuple, list) or len(address) != 2 or address[0] != "127.0.0.1"
            or type(address[1]) is not int or not 1 <= address[1] <= 65535):
        raise ValueError("offline transport requires numeric localhost and bounded port")
    deadline = time.monotonic() + timeout_s
    try:
        data = request.encode("utf-8")
        if len(data) + 1 > max_bytes or b"\n" in data:
            raise NetworkFailure("invalid request frame")
        # A numeric IPv4 socket avoids hostname/DNS work outside the budget.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as channel:
            channel.settimeout(_remaining(deadline))
            channel.connect(tuple(address))
            _write_frame(channel, data + b"\n", deadline)
            result = _read_frame(channel, max_bytes, deadline).decode("utf-8")
        parsed = json.loads(result)
        if isinstance(parsed, dict) and "transport_error" in parsed:
            raise NetworkFailure("remote loopback transport rejected request")
        return result
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NetworkFailure(type(error).__name__) from error
