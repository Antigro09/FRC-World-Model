"""Independent, bounded CustomVision coherent observations for shadow review.

This owns no pose estimator, persistent world tracks, action authority or actuator.
Receipt time, producer capture time and mapped robot time remain distinct.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import copy
import json
from pathlib import Path
import threading
import time

from .bindings import ContractError, _finite, _pairs

PROFILE = "custom-vision-schema2-2026.1"
CONTRACT = Path(__file__).with_name("_contracts") / "custom-vision-schema2"
SCHEMA_HASHES = {
    "schema2.json": "6565ccd1984efb47579b7f97940fe7d818289eac969ce4ed44b72bc10d85a966",
    "custom-vision-schema2-2026.1.json": "0d5edf4f3eb7c6091d343fe4229147d7fac0c2aa210bf602365c841c259b36e5",
}


@dataclass(frozen=True, order=True)
class VisionSource:
    root: str
    pipeline: str
    kind: str

    def __post_init__(self):
        if (not isinstance(self.root, str) or not self.root.startswith("/") or self.root.endswith("/")
                or "//" in self.root or len(self.root) > 256
                or not isinstance(self.pipeline, str) or not self.pipeline or "/" in self.pipeline
                or len(self.pipeline) > 128 or self.kind not in ("apriltag", "object")):
            raise ValueError("explicit bounded vision root/pipeline/type required")

    @property
    def topic(self):
        return self.root + "/" + self.pipeline + "/result"

    @property
    def identity(self):
        return self.topic + ":" + self.kind


@dataclass(frozen=True)
class VisionReceipt:
    source: VisionSource
    coherent_json: str
    receipt_monotonic_ns: int
    transport_timestamp_us: int
    connection_epoch: int


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012
    schemas = {}
    for name, digest in SCHEMA_HASHES.items():
        content = (CONTRACT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ContractError("CustomVision schema pin mismatch")
        schemas[name] = json.loads(content)
    registry = Registry().with_resource("urn:custom-vision:schema2",
        Resource.from_contents(schemas["schema2.json"], default_specification=DRAFT202012))
    return Draft202012Validator(schemas[PROFILE + ".json"], registry=registry)


def decode_coherent(receipt, max_bytes=262_144):
    if type(receipt) is not VisionReceipt or type(receipt.source) is not VisionSource:
        raise ContractError("typed vision receipt required")
    if not isinstance(receipt.coherent_json, str) or len(receipt.coherent_json.encode("utf-8")) > max_bytes:
        raise ContractError("oversized coherent vision packet")
    for value in (receipt.receipt_monotonic_ns, receipt.transport_timestamp_us, receipt.connection_epoch):
        if type(value) is not int or value < 0:
            raise ContractError("explicit integer vision receipt clocks/epoch required")
    try:
        packet = json.loads(receipt.coherent_json, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite vision JSON")))
        _finite(packet)
        _validator().validate(packet)
        if packet["pipeline"] != receipt.source.pipeline or packet["type"] != receipt.source.kind:
            raise ContractError("vision source/pipeline/type mismatch")
        if (len(packet["boot_id"]) > 256 or len(packet["detections"]) > 256
                or any(type(packet[key]) is not int or not 0 <= packet[key] <= (1 << 63) - 1
                       for key in ("packet_seq", "frame_id", "capture_monotonic_us"))):
            raise ContractError("vision identity/detection capacity")
        objects = packet.get("objects", {})
        targets = objects.get("targets", [])
        if len(targets) > 256:
            raise ContractError("vision target capacity")
        ids = [t["track_id"] for t in targets]
        if (len(set(ids)) != len(ids)
                or any(t["capture_monotonic_us"] != packet["capture_monotonic_us"] for t in targets)
                or (objects and objects.get("capture_monotonic_us") != packet["capture_monotonic_us"])):
            raise ContractError("vision target identity/capture mismatch")
        selected = objects.get("selected_track_id")
        if selected is not None and selected not in ids:
            raise ContractError("vision selected reference missing")
        for detection in packet["detections"]:
            relative = detection.get("robot_relative", {})
            if relative.get("valid") is True and relative.get("track_id") not in ids:
                raise ContractError("vision compact target reference missing")
        for target in targets:
            index = target["detection_index"]
            if (index >= len(packet["detections"])
                    or packet["detections"][index].get("robot_relative", {}).get("track_id") != target["track_id"]):
                raise ContractError("vision target/detection pairing mismatch")
        if packet["time_sync_valid"] is True and (type(packet["capture_server_us"]) is not int or packet["capture_server_us"] <= 0):
            raise ContractError("vision synchronized capture unavailable")
        return packet
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("invalid coherent vision: " + type(error).__name__) from error


def _families(packet):
    objects, localization, poi = (packet.get(name, {}) for name in ("objects", "localization", "poi"))
    return frozenset(name for name, valid in (
        ("detections", bool(packet["detections"])),
        ("objects", objects.get("valid") is True and bool(objects.get("targets"))),
        ("localization", localization.get("valid") is True),
        ("poi", poi.get("valid") is True)) if valid)


@dataclass
class _State:
    boot: str | None = None
    retired: set = field(default_factory=set)
    sequence: int = -1
    frame: int = -1
    nt_stamp: int = -1
    capture: int = -1
    revisions: tuple | None = None
    advancement: int = 0
    tombstones: set = field(default_factory=set)
    families: frozenset = frozenset()
    observation: tuple | None = None
    last_activity_ns: int | None = None
    last_accepted_ns: int | None = None
    last_usable_ns: int | None = None
    loss_barrier_ns: int = -1
    status: str = "awaiting_connection"
    failed: bool = False


class VisionInbox:
    """A bounded independent handoff; every source expires and fences separately.

    map_capture(source, nt_server_us) must explicitly return robot microseconds,
    or None. No arrival/capture/wall-clock substitution is performed. Two advancing
    publications (including original transport metadata) establish each session.
    """
    def __init__(self, sources, *, capacity=32, max_bytes=262_144,
                 receipt_timeout_ns=100_000_000, max_capture_age_us=100_000,
                 max_retired_boots=16, now_ns=time.monotonic_ns, map_capture=None):
        self.sources = tuple(sources)
        if not 1 <= len(self.sources) <= 16 or len(set(self.sources)) != len(self.sources):
            raise ValueError("one to sixteen unique vision sources required")
        if any(type(source) is not VisionSource for source in self.sources):
            raise ValueError("typed configured sources required")
        if (type(capacity) is not int or not 1 <= capacity <= 128
                or type(max_bytes) is not int or not 1 <= max_bytes <= 1_000_000
                or type(max_retired_boots) is not int or not 1 <= max_retired_boots <= 64):
            raise ValueError("bounded vision inbox policy required")
        if type(receipt_timeout_ns) is not int or not 0 < receipt_timeout_ns <= 2_000_000_000 or type(max_capture_age_us) is not int or not 0 < max_capture_age_us <= 2_000_000:
            raise ValueError("bounded independent receipt/capture policy required")
        _validator()  # Schema load happens at startup, never inside model work.
        self.capacity, self.max_bytes = capacity, max_bytes
        self.receipt_timeout_ns, self.max_capture_age_us = receipt_timeout_ns, max_capture_age_us
        self.max_retired_boots, self.now_ns = max_retired_boots, now_ns
        self.map_capture = map_capture or (lambda source, capture: None)
        self._states = {source: _State() for source in self.sources}
        self._queue, self._counts, self._lock = deque(), Counter(), threading.RLock()
        self._connected, self._epoch, self._closed = False, -1, False

    def _clear(self, state, reason, *, barrier=None, adopt=False):
        if state.frame >= 0:
            state.tombstones.add(state.frame)
        state.observation, state.families, state.status = None, frozenset(), reason
        if barrier is not None:
            state.loss_barrier_ns = max(state.loss_barrier_ns, barrier)
        if adopt:
            state.advancement = 0

    def connection(self, connected, epoch, *, now_ns=None):
        if type(connected) is not bool or type(epoch) is not int or epoch < 0:
            raise ValueError("verified connection state and monotonic epoch required")
        now = self.now_ns() if now_ns is None else now_ns
        if type(now) is not int or now < 0:
            raise ValueError("explicit monotonic receipt clock required")
        with self._lock:
            if self._closed or epoch < self._epoch:
                self._counts["old_connection_epoch"] += 1
                return
            if connected != self._connected or epoch != self._epoch:
                self._queue.clear()
                for state in self._states.values():
                    self._clear(state, "reconnect_adoption" if connected else "disconnected", barrier=now, adopt=True)
            self._connected, self._epoch = connected, epoch

    def loss(self, source, reason="queue_overload", *, now_ns=None):
        now = self.now_ns() if now_ns is None else now_ns
        with self._lock:
            self._clear(self._states[source], reason, barrier=now, adopt=True)
            self._queue = deque(item for item in self._queue if item.source != source)
            self._counts[reason] += 1

    def offer(self, receipt):
        with self._lock:
            if (self._closed or not self._connected or type(receipt) is not VisionReceipt
                    or type(receipt.source) is not VisionSource or receipt.source not in self._states
                    or receipt.connection_epoch != self._epoch):
                self._counts["ineligible_receipt"] += 1
                return False
            if not isinstance(receipt.coherent_json, str) or len(receipt.coherent_json.encode("utf-8")) > self.max_bytes:
                self.loss(receipt.source, "oversized_packet")
                return False
            if len(self._queue) >= self.capacity:
                lost = self._queue[0].source
                self.loss(lost, "queue_overload")
                # Overflow is a loss barrier; the current publication cannot heal it.
            self._queue.append(receipt)
            self._counts["offered"] += 1
            return True

    def process(self, maximum=16):
        if type(maximum) is not int or not 1 <= maximum <= 128:
            raise ValueError("bounded vision processing quantum required")
        accepted = []
        with self._lock:
            now = self.now_ns()
            for _ in range(maximum):
                if not self._queue:
                    break
                receipt = self._queue.popleft()
                state = self._states[receipt.source]
                if (receipt.connection_epoch != self._epoch or not self._connected
                        or type(receipt.receipt_monotonic_ns) is not int
                        or not 0 <= now - receipt.receipt_monotonic_ns < self.receipt_timeout_ns
                        or receipt.receipt_monotonic_ns <= state.loss_barrier_ns):
                    self._counts["stale_or_fenced_receipt"] += 1
                    continue
                state.last_activity_ns = receipt.receipt_monotonic_ns
                try:
                    packet = decode_coherent(receipt, self.max_bytes)
                except ContractError:
                    self.loss(receipt.source, "malformed_packet", now_ns=now)
                    continue
                if state.failed or packet["boot_id"] in state.retired:
                    self._counts["retired_or_failed_boot"] += 1
                    continue
                if state.boot != packet["boot_id"]:
                    if state.boot is not None and state.boot not in state.retired:
                        if len(state.retired) >= self.max_retired_boots:
                            state.failed = True
                            self._clear(state, "retirement_limit")
                            continue
                        state.retired.add(state.boot)
                    retired, barrier = state.retired, state.loss_barrier_ns
                    state = self._states[receipt.source] = _State(boot=packet["boot_id"], retired=retired, loss_barrier_ns=barrier)
                revisions = tuple(packet[k] for k in ("calibration_revision", "mount_revision", "field_layout_revision"))
                if state.revisions is not None and revisions != state.revisions:
                    self._clear(state, "revision_changed_without_boot", adopt=True)
                    # A corrupted boot cannot resume by reverting its revisions.
                    if len(state.retired) >= self.max_retired_boots:
                        state.failed = True
                    else:
                        state.retired.add(state.boot)
                    self._counts["revision_changed_without_boot"] += 1
                    continue
                if packet["packet_seq"] <= state.sequence or packet["frame_id"] < state.frame or receipt.transport_timestamp_us <= state.nt_stamp or receipt.transport_timestamp_us <= 0:
                    self._counts["duplicate_or_reordered"] += 1
                    continue
                prior_frame, prior_capture, prior_families = state.frame, state.capture, state.families
                state.sequence, state.frame, state.nt_stamp = packet["packet_seq"], packet["frame_id"], receipt.transport_timestamp_us
                state.revisions, state.last_accepted_ns = revisions, receipt.receipt_monotonic_ns
                state.advancement = min(2, state.advancement + 1)
                state.tombstones = {frame for frame in state.tombstones if frame >= state.frame}
                families = _families(packet)
                invalid = (not packet["connected"] or packet["error"] is not None or not packet["detections"]
                    or (prior_frame == state.frame and not prior_families.issubset(families)))
                if invalid:
                    self._clear(state, "invalidation")
                    self._counts["invalidation"] += 1
                    continue
                if state.frame in state.tombstones:
                    self._clear(state, "tombstoned_frame")
                    self._counts["tombstoned_frame"] += 1
                    continue
                if packet["capture_monotonic_us"] <= prior_capture or (prior_frame == state.frame and state.capture >= 0):
                    self._counts["duplicate_capture"] += 1
                    continue
                state.capture = packet["capture_monotonic_us"]
                if state.advancement < 2:
                    state.status = "pending_advancing_publication"
                    continue
                state.observation, state.families = (receipt, packet), families
                state.last_usable_ns, state.status = receipt.receipt_monotonic_ns, "observed"
                self._counts["observed"] += 1
                accepted.append(receipt)
            self.expire(now)
        return tuple(accepted)

    def expire(self, now_ns=None):
        now = self.now_ns() if now_ns is None else now_ns
        with self._lock:
            for state in self._states.values():
                if state.observation and not 0 <= now - state.last_usable_ns < self.receipt_timeout_ns:
                    self._clear(state, "receipt_expired", barrier=now, adopt=True)
                    self._counts["receipt_expired"] += 1

    def trace_for_request(self, request):
        r = request["prediction_request"]
        cutoff = r["history_cutoff_us"]
        with self._lock:
            self.expire()
            # Mapping is owner-supplied work. Copy the small latest-state view,
            # then release the subscription lock before invoking the callback.
            snapshots = [(source, state.status, state.observation, state.families)
                         for source, state in self._states.items()]
            epoch = self._epoch
        entries = []
        for source, status, observation, families in snapshots:
            entry = {"source": source.identity, "topic": source.topic, "connection_epoch": epoch,
                "status": status, "eligible_at_robot_cutoff": False, "observed": observation is not None,
                "authoritative_ego": False, "reason": status}
            if observation:
                receipt, packet = observation
                capture = packet["capture_server_us"]
                try:
                    mapped = self.map_capture(source, capture) if packet["time_sync_valid"] else None
                except Exception:
                    mapped = None
                eligible = type(mapped) is int and mapped >= 0 and 0 <= cutoff - mapped <= self.max_capture_age_us
                entry.update(boot_id=packet["boot_id"], frame_id=packet["frame_id"], packet_seq=packet["packet_seq"],
                    capture_monotonic_us=packet["capture_monotonic_us"], capture_server_us=capture,
                    capture_robot_us=mapped if type(mapped) is int and mapped >= 0 else None,
                    receipt_monotonic_ns=receipt.receipt_monotonic_ns, transport_timestamp_us=receipt.transport_timestamp_us,
                    packet_sha256=hashlib.sha256(receipt.coherent_json.encode()).hexdigest(),
                    eligible_at_robot_cutoff=eligible, families=sorted(families),
                    frame="producer_capture_relative; no projection or fusion",
                    reason="matched_observed_input" if eligible else "unmapped_stale_or_future_capture")
            entries.append(entry)
        # A mapping callback may overlap a reconnect, invalidation, expiry or
        # newer frame. Recheck every captured source after all callback work;
        # an old observation cannot cross those fences in the returned trace.
        with self._lock:
            self.expire()
            for (source, status, observation, _), entry in zip(snapshots, entries):
                state = self._states[source]
                if (state.observation is not observation or state.status != status or self._epoch != epoch
                        or (observation is not None and (self._closed or not self._connected))):
                    entry.update(connection_epoch=self._epoch, status=state.status, observed=False,
                                 eligible_at_robot_cutoff=False, reason="superseded_during_trace")
        return {"schema_version": "frc-shadow-vision-trace/1", "robot_request_id": r["request_id"],
            "robot_snapshot_id": r["snapshot_id"], "robot_epoch": r["epoch"], "robot_cutoff_us": cutoff,
            "robot_source": copy.deepcopy(r["source"]), "observations": entries, "influences_commands": False}

    def diagnostics(self):
        with self._lock:
            self.expire()
            return {"connected": self._connected, "connection_epoch": self._epoch,
                "queue_depth": len(self._queue), "queue_capacity": self.capacity, "counters": dict(self._counts),
                "sources": {s.identity: {"status": state.status, "boot_id": state.boot, "packet_seq": state.sequence,
                    "frame_id": state.frame, "retired_boots": len(state.retired)} for s, state in self._states.items()}}

    def close(self):
        with self._lock:
            self._closed = True
            self._queue.clear()
            for state in self._states.values():
                self._clear(state, "stopped")
