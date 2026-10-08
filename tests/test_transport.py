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
