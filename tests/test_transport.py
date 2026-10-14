import json
import socket
import time

import pytest
from frc_world_model.transport import NetworkFailure, loopback_server, roundtrip


def test_loopback_serialization_server_restart_and_disconnect():
    def handler(payload):
        return json.dumps(json.loads(payload),sort_keys=True,separators=(",",":"))
    with loopback_server(handler) as address:
        assert json.loads(roundtrip(address,'{"epoch":1,"forecast":false}')) == {"epoch":1,"forecast":False}
    with pytest.raises(NetworkFailure):
        roundtrip(address,'{}',timeout_s=.1)
    with loopback_server(handler) as new_address:
        assert json.loads(roundtrip(new_address,'{"epoch":2}'))["epoch"] == 2


def test_network_timeout_and_failed_handler_are_missing_predictions():
    def crash(_):
        raise ValueError("private-input")
    with loopback_server(crash) as address:
        with pytest.raises(NetworkFailure):
            roundtrip(address,'{}')
    def slow(_):
        time.sleep(.03)
        return '{}'
    with loopback_server(slow) as address:
        with pytest.raises(NetworkFailure):
            roundtrip(address,'{}',timeout_s=.005)


def test_malformed_oversized_and_nonlocal_frames_rejected():
    with loopback_server(lambda x:x,max_bytes=40) as address:
        with pytest.raises(NetworkFailure):
            roundtrip(address,'x'*100,max_bytes=40)
        with pytest.raises(NetworkFailure):
            roundtrip(address,'{}\n{}',max_bytes=40)
        with pytest.raises(NetworkFailure):
            roundtrip(address,'invalid-json',max_bytes=40)
    with pytest.raises(ValueError):
        roundtrip(('192.0.2.1',1),'{}')


def test_exact_wire_loopback_stale_epoch_and_abstention_preserve_baseline():
    from frc_world_model.bindings import CONTRACT,ContractError,load_document,make_masked_reply,parse_request,validate_reply
    from tests.test_bindings import policy
    text=(CONTRACT/"request-v1.json").read_text().strip()
    r=parse_request(text,now_us=1010000,max_sync_error_us=1000)
    def handler(payload):
        req=parse_request(payload,now_us=1010000,max_sync_error_us=1000)
        return json.dumps(make_masked_reply(req,1010000),separators=(",",":"))
    with loopback_server(handler) as address:
        reply=load_document(roundtrip(address,text),"reply")
    with pytest.raises(ContractError,match="uncalibrated"):
        validate_reply(r,reply,now_us=1030000,policy=policy())
    reply["epoch"]+=1
    with pytest.raises(ContractError,match="mismatch"):
        validate_reply(r,reply,now_us=1030000,policy=policy())


def test_slow_drip_peer_has_total_read_budget_and_next_request_is_served():
    import threading
    sent = threading.Event()
    with loopback_server(lambda payload: payload, socket_timeout_s=.03) as address:
        def drip():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as peer:
                peer.connect(address)
                for byte in b'{"slow":true}\n':
                    try:
                        peer.sendall(bytes([byte]))
                        sent.set()
                        time.sleep(.015)
                    except OSError:
                        return
        sender = threading.Thread(target=drip, daemon=True)
        sender.start()
        assert sent.wait(1)
        time.sleep(.05)
        assert roundtrip(address, '{}', timeout_s=.1) == '{}'
        sender.join(.2)


def test_roundtrip_slow_drip_reply_cannot_refresh_total_exchange_budget():
    import threading
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0)); listener.listen(1)
        def drip():
            peer, _ = listener.accept()
            with peer:
                peer.recv(100)
                for byte in b'{"v":1}\n':
                    try:
                        peer.sendall(bytes([byte])); time.sleep(.015)
                    except OSError:
                        return
        sender = threading.Thread(target=drip, daemon=True)
        sender.start()
        start = time.monotonic()
        with pytest.raises(NetworkFailure):
            roundtrip(listener.getsockname(), '{}', timeout_s=.035)
        assert time.monotonic() - start < .2
        sender.join(.2)


def test_server_shutdown_is_bounded_for_stalled_peer_and_uncooperative_handler():
    import threading
    entered, release = threading.Event(), threading.Event()
    def uncooperative(payload):
        entered.set()
        release.wait(2)
        return payload
    peer = None
    try:
        start = time.monotonic()
        with loopback_server(uncooperative) as address:
            peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer.connect(address); peer.sendall(b'{}\n')
            assert entered.wait(.5)
        assert time.monotonic() - start < .5
    finally:
        release.set()
        if peer is not None:
            peer.close()
    with loopback_server(lambda payload: payload, socket_timeout_s=1) as address:
        peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        peer.connect(address); peer.sendall(b'{')
        time.sleep(.01)
        start = time.monotonic()
    peer.close()
    assert time.monotonic() - start < .4
