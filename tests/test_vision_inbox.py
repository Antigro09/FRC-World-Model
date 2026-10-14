"""Synthetic coherent inputs; no camera/controller or pose-fusion I/O."""
import copy
import json
import threading

import pytest

from frc_world_model.bindings import ContractError
from frc_world_model.vision_inbox import (CONTRACT, VisionInbox, VisionReceipt, VisionSource,
                                         decode_coherent)
from tools.benchmark_decisions import fixture_request

SOURCE = VisionSource("/CustomVision", "front_tags", "apriltag")


class Clock:
    def __init__(self):
        self.value = 1_000_000_000
    def __call__(self):
        return self.value
    def advance(self, ns=1_000_000):
        self.value += ns
        return self.value


def packet(sequence=1, frame=None, boot="boot-a", **updates):
    result = json.loads((CONTRACT / "multitag.json").read_text())
    result.update(packet_seq=sequence, frame_id=sequence if frame is None else frame,
                  boot_id=boot, capture_monotonic_us=100_000 + sequence,
                  capture_server_us=1_000_000 + sequence)
    result.update(updates)
    return result


def offer(inbox, clock, payload, *, source=SOURCE, epoch=1, stamp=None):
    clock.advance()
    receipt = VisionReceipt(source, json.dumps(payload) if isinstance(payload, dict) else payload,
        clock(), payload.get("packet_seq", 1) * 10 if stamp is None and isinstance(payload, dict) else (stamp or 10), epoch)
    assert inbox.offer(receipt)
    return inbox.process()


def ready(inbox, clock, *, source=SOURCE):
    assert offer(inbox, clock, packet(1), source=source) == ()
    assert len(offer(inbox, clock, packet(2), source=source)) == 1


def inbox_clock(**kwargs):
    clock = Clock()
    inbox = VisionInbox([SOURCE], now_ns=clock, **kwargs)
    inbox.connection(True, 1)
    return inbox, clock


def trace(inbox):
    request = fixture_request()
    request["prediction_request"]["history_cutoff_us"] = 1_000_010
    return inbox.trace_for_request(request)["observations"][0]


def test_packaged_schema_pin_strict_coherence_and_original_clocks():
    receipt = VisionReceipt(SOURCE, json.dumps(packet()), 10, 20, 1)
    assert decode_coherent(receipt)["capture_server_us"] == 1_000_001
    for payload in ('{"schema_version":2,"schema_version":2}', 'NaN', json.dumps(packet(pipeline="other")),
                    json.dumps(packet(packet_seq=1 << 63)), json.dumps(packet(time_sync_valid=True, capture_server_us=None))):
        with pytest.raises(ContractError):
            decode_coherent(VisionReceipt(SOURCE, payload, 10, 20, 1))
    with pytest.raises(ContractError, match="oversized"):
        decode_coherent(receipt, max_bytes=10)


def test_two_advancing_publications_and_causal_trace_never_mutate_robot_request():
    inbox, clock = inbox_clock(map_capture=lambda source, capture: capture)
    ready(inbox, clock)
    request = fixture_request()
    request["prediction_request"]["history_cutoff_us"] = 1_000_010
    saved = copy.deepcopy(request)
    result = inbox.trace_for_request(request)
    observation = result["observations"][0]
    assert observation["eligible_at_robot_cutoff"] is True
    assert observation["capture_robot_us"] == 1_000_002
    assert observation["transport_timestamp_us"] == 20
    assert observation["receipt_monotonic_ns"] == clock()
    assert observation["authoritative_ego"] is False and result["influences_commands"] is False
    assert len(observation["packet_sha256"]) == 64 and "localization" in observation["families"]
    assert request == saved


@pytest.mark.parametrize("mapper", [None, lambda *_: None, lambda *_: True,
                                     lambda *_: 1_000_011, lambda *_: 0])
def test_missing_or_future_clock_mapping_is_observed_but_not_eligible(mapper):
    inbox, clock = inbox_clock(map_capture=mapper)
    ready(inbox, clock)
    assert trace(inbox)["observed"] is True
    assert trace(inbox)["eligible_at_robot_cutoff"] is False


def test_stale_duplicate_receipts_cannot_extend_lifetime():
    inbox, clock = inbox_clock()
    ready(inbox, clock)
    # Same publication with fresh receipt metadata is not fresh camera input.
    assert offer(inbox, clock, packet(2), stamp=30) == ()
    clock.advance(100_000_000)
    assert trace(inbox)["status"] == "receipt_expired"
    assert trace(inbox)["observed"] is False
    assert offer(inbox, clock, packet(3)) == ()
    assert len(offer(inbox, clock, packet(4))) == 1


def test_same_frame_invalidation_is_tombstoned_and_new_frame_recovers():
    inbox, clock = inbox_clock()
    ready(inbox, clock)
    assert offer(inbox, clock, packet(3, frame=2, detections=[])) == ()
    assert trace(inbox)["status"] == "invalidation"
    assert offer(inbox, clock, packet(4, frame=2)) == ()
    assert trace(inbox)["status"] == "tombstoned_frame"
    assert len(offer(inbox, clock, packet(5, frame=3))) == 1


def test_reconnect_fences_retained_duplicates_and_requires_two_advances():
    inbox, clock = inbox_clock()
    ready(inbox, clock)
    inbox.connection(False, 2)
    inbox.connection(True, 3)
    assert not inbox.offer(VisionReceipt(SOURCE, json.dumps(packet(3)), clock(), 30, 1))
    assert offer(inbox, clock, packet(2), epoch=3, stamp=20) == ()
    assert offer(inbox, clock, packet(3), epoch=3) == ()
    assert len(offer(inbox, clock, packet(4), epoch=3)) == 1
    inbox.connection(False, 2)
    assert inbox.diagnostics()["connected"] is True


def test_boot_retirement_revision_corruption_and_bounded_retirement():
    inbox, clock = inbox_clock(max_retired_boots=2)
    ready(inbox, clock)
    assert offer(inbox, clock, packet(3, mount_revision="sha256:" + "a" * 64)) == ()
    assert trace(inbox)["status"] == "revision_changed_without_boot"
    assert offer(inbox, clock, packet(4)) == ()  # reverting cannot revive the boot
    assert offer(inbox, clock, packet(1, boot="boot-b"), stamp=40) == ()
    assert len(offer(inbox, clock, packet(2, boot="boot-b"), stamp=50)) == 1
    assert offer(inbox, clock, packet(5), stamp=60) == ()
    assert offer(inbox, clock, packet(1, boot="boot-c"), stamp=70) == ()
    assert offer(inbox, clock, packet(1, boot="boot-d"), stamp=80) == ()
    assert inbox.diagnostics()["sources"][SOURCE.identity]["status"] == "retirement_limit"


def test_saturated_queue_is_bounded_and_loss_requires_fresh_adoption():
    inbox, clock = inbox_clock(capacity=2)
    for seq in (1, 2, 3):
        clock.advance()
        assert inbox.offer(VisionReceipt(SOURCE, json.dumps(packet(seq)), clock(), seq * 10, 1))
        assert inbox.diagnostics()["queue_depth"] <= 2
    assert inbox.process() == ()
    assert inbox.diagnostics()["counters"]["queue_overload"] == 1
    assert offer(inbox, clock, packet(4)) == ()
    assert len(offer(inbox, clock, packet(5))) == 1


def test_malformed_packet_fences_source_and_safe_drain_handles_removed_queue():
    inbox, clock = inbox_clock()
    for text in ("{bad", json.dumps(packet(2))):
        clock.advance()
        assert inbox.offer(VisionReceipt(SOURCE, text, clock(), clock() // 1000, 1))
    assert inbox.process(16) == ()
    assert inbox.diagnostics()["queue_depth"] == 0
    assert trace(inbox)["status"] == "malformed_packet"


def test_loss_is_per_source_and_slow_mapping_never_holds_vision_inbox_lock():
    other = VisionSource("/OtherCustomVision", "front_tags", "apriltag")
    clock = Clock()
    entered, release = threading.Event(), threading.Event()
    def mapping(source, capture):
        entered.set()
        assert release.wait(1)
        return capture
    inbox = VisionInbox([SOURCE, other], now_ns=clock, map_capture=mapping)
    inbox.connection(True, 1)
    ready(inbox, clock)
    ready(inbox, clock, source=other)
    worker = threading.Thread(target=inbox.trace_for_request, args=(fixture_request(),), daemon=True)
    worker.start()
    try:
        assert entered.wait(1)
        inbox.loss(SOURCE, "synthetic_loss")
        assert len(offer(inbox, clock, packet(3), source=other)) == 1
    finally:
        release.set()
        worker.join(1)
    assert inbox.diagnostics()["sources"][other.identity]["status"] == "observed"
    inbox.close()
    inbox.connection(True, 2)
    assert not inbox.offer(VisionReceipt(other, json.dumps(packet(4)), clock(), 40, 2))


@pytest.mark.parametrize("fence", ["disconnect", "loss", "expiry", "new_frame"])
def test_slow_mapping_trace_rechecks_every_source_fence(fence):
    entered, release = threading.Event(), threading.Event()
    def mapping(source, capture):
        entered.set()
        assert release.wait(1)
        return capture
    inbox, clock = inbox_clock(map_capture=mapping)
    ready(inbox, clock)
    results = []
    worker = threading.Thread(target=lambda: results.append(inbox.trace_for_request(fixture_request())), daemon=True)
    worker.start()
    try:
        assert entered.wait(1)
        if fence == "disconnect":
            inbox.connection(False, 2)
        elif fence == "loss":
            inbox.loss(SOURCE)
        elif fence == "expiry":
            clock.advance(100_000_000)
        else:
            assert len(offer(inbox, clock, packet(3))) == 1
    finally:
        release.set(); worker.join(1)
    entry = results[0]["observations"][0]
    assert entry["observed"] is False and entry["eligible_at_robot_cutoff"] is False
    assert entry["reason"] == "superseded_during_trace"
    assert entry["connection_epoch"] == inbox.diagnostics()["connection_epoch"]


@pytest.mark.parametrize("corruption", ["selected", "target_capture", "envelope_capture", "detection_reference", "detection_index"])
def test_object_capture_selection_and_compact_references_are_coherent(corruption):
    payload = json.loads((CONTRACT / "objects_compact_covariance_selection.json").read_text())
    source = VisionSource("/CustomVision", payload["pipeline"], "object")
    receipt = lambda: VisionReceipt(source, json.dumps(payload), 10, 20, 1)
    assert decode_coherent(receipt())["objects"]["valid"] is True
    if corruption == "selected":
        payload["objects"]["selected_track_id"] = 999
    elif corruption == "target_capture":
        payload["objects"]["targets"][0]["capture_monotonic_us"] += 1
    elif corruption == "envelope_capture":
        payload["objects"]["capture_monotonic_us"] += 1
    elif corruption == "detection_reference":
        payload["detections"][0]["robot_relative"]["track_id"] = 999
    else:
        payload["objects"]["targets"][0]["detection_index"] = 999
    with pytest.raises(ContractError):
        decode_coherent(receipt())
