"""Bounded localhost NDJSON transport for offline/shadow loopback checks.

No controller endpoint/topic is inferred. One serialized worker avoids
unbounded optional inference. The owner supplies the exact wire handler.
"""
from contextlib import contextmanager
import json
import socket
import socketserver
import threading


class NetworkFailure(ValueError):
    pass


def _read_frame(stream, maximum):
    frame = stream.readline(maximum + 1)
    if not frame or len(frame) > maximum or not frame.endswith(b"\n"):
        raise NetworkFailure("missing, truncated or oversized frame")
    return frame[:-1]


@contextmanager
def loopback_server(handler, *, max_bytes=1_000_000, socket_timeout_s=.5):
    if not 1 <= max_bytes <= 8_000_000 or not 0 < socket_timeout_s <= 2:
        raise ValueError("unsupported transport limits")
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.request.settimeout(socket_timeout_s)
            try:
                request = _read_frame(self.rfile,max_bytes)
                reply = handler(request.decode("utf-8"))
                data = reply.encode("utf-8")
                if len(data)+1 > max_bytes or b"\n" in data:
                    raise NetworkFailure("invalid response frame")
                self.wfile.write(data+b"\n")
            except Exception as error:
                # Stable transport failure, never a fabricated forecast.
                data = json.dumps({"transport_error":type(error).__name__},separators=(",",":")).encode()
                try:
                    self.wfile.write(data+b"\n")
                except OSError:
                    pass
    class Server(socketserver.TCPServer):
        allow_reuse_address = False
        request_queue_size = 2
    server = Server(("127.0.0.1",0),Handler)
    thread = threading.Thread(target=server.serve_forever,kwargs={"poll_interval":.01},daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def roundtrip(address, request: str, *, timeout_s=.5, max_bytes=1_000_000):
    if address[0] != "127.0.0.1" or not 0 < timeout_s <= 2:
        raise ValueError("offline transport requires localhost and bounded timeout")
    data = request.encode("utf-8")
    if len(data)+1 > max_bytes or b"\n" in data:
        raise NetworkFailure("invalid request frame")
    try:
        with socket.create_connection(address,timeout=timeout_s) as sock:
            sock.sendall(data+b"\n")
            with sock.makefile("rb") as stream:
                result = _read_frame(stream,max_bytes).decode("utf-8")
        parsed = json.loads(result)
        if isinstance(parsed,dict) and "transport_error" in parsed:
            raise NetworkFailure("remote loopback transport rejected request")
        return result
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NetworkFailure(type(error).__name__) from error
