"""Frozen, bounded prediction service for replay/shadow only.

All dataclasses here are provisional internal representations. External wire
bindings must preserve the exact exported World-State schema; live promotion
is unavailable in this module. Forecasts are never fresh measurements.
"""
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Callable, Dict, Mapping, Optional, Tuple
import uuid

from .analytical import (AnalyticalDynamics, EgoMechanismState, LAYOUT_ID,
                         LAYOUT_STATUS, PhysicalCommand, Rollout, STATE_LAYOUT)


@dataclass(frozen=True)
class RequestIdentity:
    schema_version: str
    robot_id: str
    map_id: str
    frame_id: str
    calibration_id: str
    source_id: str
    source_epoch: str
    session_id: str
    reset_epoch: str
    clock_id: str
    service_epoch: str = ""


@dataclass(frozen=True)
class StateSample:
    timestamp_ns: int
    state: EgoMechanismState
    valid: Tuple[bool, ...] = (True,) * len(STATE_LAYOUT)
    provenance: str = "causal-fused-estimate"


@dataclass(frozen=True)
class CandidateRollout:
    candidate_id: str
    commands: Tuple[PhysicalCommand, ...]


@dataclass(frozen=True)
class PredictionRequest:
    identity: RequestIdentity
    snapshot_id: str
    request_id: str
    model_id: str
    capture_time_ns: int
    state_time_ns: int
    history_cutoff_ns: int
    issued_ns: int
    expiry_ns: int
    sync_valid: bool
    initial_state: EgoMechanismState
    candidates: Tuple[CandidateRollout, ...]
    horizon_steps: int
    history: Tuple[StateSample, ...] = ()
    state_valid: Tuple[bool, ...] = (True,) * len(STATE_LAYOUT)
    sync_id: str = "provisional-clock-sync"
    layout_id: str = LAYOUT_ID
    layout_status: str = LAYOUT_STATUS
    mode: str = "replay"

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str, max_bytes: int = 8_000_000) -> "PredictionRequest":
        if len(value.encode("utf-8")) > max_bytes:
            raise ValueError("request exceeds serialization limit")
        def reject_constant(token):
            raise ValueError("nonfinite JSON constant: " + token)
        raw = json.loads(value, parse_constant=reject_constant)
        raw["identity"] = RequestIdentity(**raw["identity"])
        raw["initial_state"] = EgoMechanismState(**raw["initial_state"])
        raw["state_valid"] = tuple(raw["state_valid"])
        candidates = []
        for candidate in raw["candidates"]:
            commands = []
            for command in candidate["commands"]:
                if command["execution_deviation"] is not None:
                    command["execution_deviation"] = tuple(command["execution_deviation"])
                commands.append(PhysicalCommand(**command))
            candidates.append(CandidateRollout(candidate["candidate_id"], tuple(commands)))
        raw["candidates"] = tuple(candidates)
        history = []
        for sample in raw["history"]:
            sample["state"] = EgoMechanismState(**sample["state"])
            sample["valid"] = tuple(sample["valid"])
            history.append(StateSample(**sample))
        raw["history"] = tuple(history)
        return cls(**raw)


@dataclass(frozen=True)
class CandidateForecast:
    candidate_id: str
    rollout: Rollout
    predictor: str
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class PredictionReply:
    identity: RequestIdentity
    snapshot_id: str
    request_id: str
    model_id: str
    capture_time_ns: int
    state_time_ns: int
    history_cutoff_ns: int
    issued_ns: int
    expiry_ns: int
    horizon_steps: int
    layout_id: str
    layout_status: str
    mode: str
    sync_id: str
    status: str
    forecasts: Tuple[CandidateForecast, ...]
    rejection_reason: Optional[str] = None
    measured: bool = False
    influences_commands: bool = False
    uncertainty_status: str = "uncalibrated-no-physical-covariance"
    state_valid: Tuple[bool, ...] = (True,) * len(STATE_LAYOUT)
    sync_valid: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str, max_bytes: int = 8_000_000) -> "PredictionReply":
        if len(value.encode("utf-8")) > max_bytes:
            raise ValueError("reply exceeds serialization limit")
        def reject_constant(token):
            raise ValueError("nonfinite JSON constant: " + token)
        raw = json.loads(value, parse_constant=reject_constant)
        raw["identity"] = RequestIdentity(**raw["identity"])
        forecasts = []
        for f in raw["forecasts"]:
            r = f["rollout"]
            if r["kind"] not in ("analytical-forecast", "learned-forecast") or r["measured"] is not False:
                raise ValueError("forecast cannot be reclassified as measured state")
            r["states"] = tuple(EgoMechanismState(**s) for s in r["states"])
            r["times_ns"] = tuple(r["times_ns"])
            r["saturated"] = tuple(tuple(v) for v in r["saturated"])
            f["rollout"] = Rollout(**r)
            forecasts.append(CandidateForecast(**f))
        raw["forecasts"] = tuple(forecasts)
        raw["state_valid"] = tuple(raw["state_valid"])
        reply = cls(**raw)
        if reply.measured is not False or reply.influences_commands is not False:
            raise ValueError("replay/shadow reply cannot influence commands or become measured")
        return reply


@dataclass(frozen=True)
class ServiceLimits:
    max_candidates: int = 8
    max_horizon_steps: int = 100
    max_history_steps: int = 256
    max_pending: int = 2
    max_age_ns: int = 500_000_000
    max_capture_state_skew_ns: int = 500_000_000
    workers: int = 1

    def __post_init__(self) -> None:
        if min(asdict(self).values()) <= 0:
            raise ValueError("all service limits must be positive")


@dataclass(frozen=True)
class FrozenPackage:
    root: Path
    manifest: Mapping[str, object]
    checksum: str
    metadata: Mapping[str, object]

    @classmethod
    def load(cls, path, max_artifact_bytes: int = 64_000_000) -> "FrozenPackage":
        """Validate checksums and JSON metadata, never unpickle or download weights.

        Manifest artifacts are keyed by weights/schema/normalization/calibration.
        Every artifact requires {path, sha256}. Symlinks and escaping paths fail.
        This validates packaging only; it does not claim trained/calibrated FRC
        weights exist or establish on-device compatibility.
        """
        root = Path(path).resolve()
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("manifest symlink is disallowed")
        if manifest_path.stat().st_size > 1_000_000:
            raise ValueError("manifest too large")
        source = manifest_path.read_bytes()
        manifest = json.loads(source)
        if manifest.get("format") != "frc-world-model-frozen-v1":
            raise ValueError("unknown package format")
        for key in ("model_id", "schema_version", "layout_id", "model_kind"):
            if not isinstance(manifest.get(key), str) or not manifest[key]:
                raise ValueError("missing manifest identity: " + key)
        if manifest["model_kind"] not in ("rwm-u", "analytical-fixture"):
            raise ValueError("unsupported frozen model kind")
        artifacts = manifest.get("artifacts", {})
        if not {"weights", "schema", "normalization", "calibration"}.issubset(artifacts):
            raise ValueError("package requires weights/schema/normalization/calibration artifacts")
        metadata: Dict[str, object] = {}
        for name, record in artifacts.items():
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("artifact path escapes package")
            candidate = root / relative
            if any(parent.is_symlink() for parent in (candidate,) + tuple(candidate.parents) if parent != root.parent):
                raise ValueError("artifact symlink is disallowed")
            candidate = candidate.resolve()
            if root not in candidate.parents or not candidate.is_file():
                raise ValueError("artifact missing or outside package")
            if candidate.stat().st_size > max_artifact_bytes:
                raise ValueError("artifact exceeds configured size limit")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError("artifact checksum mismatch: " + name)
            if name != "weights":
                metadata[name] = json.loads(candidate.read_bytes())
        schema = metadata["schema"]
        if schema.get("version") != manifest["schema_version"] or schema.get("layout_id") != manifest["layout_id"]:
            raise ValueError("manifest/schema identity mismatch")
        norm = metadata["normalization"]
        if norm.get("fit_split") != "train" or not norm.get("dataset_sha256"):
            raise ValueError("normalization requires train-only provenance")
        mean, scale = norm.get("mean", []), norm.get("scale", [])
        if not 1 <= len(mean) <= 256 or len(scale) != len(mean):
            raise ValueError("normalization dimensions mismatch")
        layout = norm.get("layout")
        names = layout.get("names") if isinstance(layout, dict) else layout
        if not isinstance(names, list) or len(names) != len(mean) or len(set(names)) != len(names):
            raise ValueError("normalization requires explicit unique feature layout")
        if isinstance(layout, dict):
            units = layout.get("units")
            if not isinstance(units, list) or len(units) != len(names) or any(not isinstance(v,str) or not v for v in units):
                raise ValueError("normalization requires explicit units")
        if "config" in metadata and metadata["config"].get("state_dim") != len(mean):
            raise ValueError("normalization/config state dimension mismatch")
        if not all(math.isfinite(v) for v in mean + scale) or not all(v > 0 for v in scale):
            raise ValueError("invalid normalization")
        calibration = metadata["calibration"]
        if calibration.get("status") not in ("uncalibrated", "held-out-session-calibrated"):
            raise ValueError("calibration status missing")
        if calibration.get("status") == "held-out-session-calibrated" and not calibration.get("session_ids"):
            raise ValueError("calibration requires held-out session provenance")
        return cls(root, manifest, hashlib.sha256(source).hexdigest(), metadata)


class PredictionService:
    """No control interfaces. Baseline survives absent/failed optional inference.

    submit() bounds pending optional work. Overload sheds learned inference and
    returns the already available analytical forecast. cancel() and reset()
    prevent publishing late learned results; uncooperative adapters may continue
    CPU work until they return. Deployed adapters must have bounded execution.
    """
    def __init__(self, identity: RequestIdentity, baseline: Optional[AnalyticalDynamics] = None,
                 predictor: Optional[Callable[[PredictionRequest], Mapping[str, Rollout]]] = None,
                 model_id: str = "analytical-unfitted", limits: Optional[ServiceLimits] = None,
                 clock: Callable[[], int] = time.monotonic_ns):
        self.identity = replace(identity, service_epoch=uuid.uuid4().hex)
        self.baseline = baseline or AnalyticalDynamics()
        self.predictor, self.model_id = predictor, model_id
        self.limits, self.clock = limits or ServiceLimits(), clock
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=self.limits.workers,
                                        thread_name_prefix="optional-prediction")
        self._pending = 0
        self._closed = False
        self._cancelled = set()
        self._seen = {}
        self._generation = 0
        self._last_now = None

    def _reject_reason(self, request: PredictionRequest, now_ns: int) -> Optional[str]:
        if self._closed:
            return "service-stopped"
        if request.identity != self.identity:
            return "identity-or-epoch-mismatch"
        if any(not str(value) for value in asdict(request.identity).values()):
            return "missing-identity-prerequisite"
        if request.model_id != self.model_id:
            return "model-mismatch"
        if request.layout_id != LAYOUT_ID or request.layout_status != LAYOUT_STATUS:
            return "schema-or-layout-mismatch"
        if request.mode not in ("replay", "shadow"):
            return "live-promotion-requires-separate-approval-and-evidence"
        if not request.sync_valid or not request.sync_id:
            return "clock-sync-invalid"
        times = (request.capture_time_ns, request.state_time_ns, request.history_cutoff_ns,
                 request.issued_ns, request.expiry_ns, now_ns)
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in times):
            return "invalid-time-units"
        if self._last_now is not None and now_ns < self._last_now:
            return "clock-rollback"
        self._last_now = now_ns
        if not request.capture_time_ns <= request.state_time_ns == request.history_cutoff_ns <= request.issued_ns <= now_ns:
            return "noncausal-time-alignment"
        if now_ns >= request.expiry_ns:
            return "expired"
        if now_ns - request.state_time_ns > self.limits.max_age_ns:
            return "stale-state"
        if request.state_time_ns - request.capture_time_ns > self.limits.max_capture_state_skew_ns:
            return "capture-state-skew"
        if len(request.state_valid) != len(STATE_LAYOUT) or not all(v is True for v in request.state_valid):
            return "missing-state-prerequisite"
        if not request.snapshot_id or not request.request_id:
            return "missing-request-identity"
        if not 1 <= request.horizon_steps <= self.limits.max_horizon_steps:
            return "horizon-limit"
        if not 1 <= len(request.candidates) <= self.limits.max_candidates:
            return "candidate-limit"
        if len(request.history) > self.limits.max_history_steps:
            return "history-limit"
        previous = -1
        for sample in request.history:
            if sample.timestamp_ns <= previous or sample.timestamp_ns > request.history_cutoff_ns:
                return "history-order-or-cutoff"
            if sample.provenance != "causal-fused-estimate":
                return "future-smoothed-or-unknown-input-provenance"
            if len(sample.valid) != len(STATE_LAYOUT) or not all(v is True for v in sample.valid):
                return "missing-history-prerequisite"
            previous = sample.timestamp_ns
        candidate_ids = [c.candidate_id for c in request.candidates]
        if any(not c for c in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
            return "candidate-id-invalid"
        for candidate in request.candidates:
            if len(candidate.commands) != request.horizon_steps:
                return "candidate-horizon-mismatch"
            for i, command in enumerate(candidate.commands):
                if command.timestamp_ns != request.history_cutoff_ns + i * self.baseline.config.dt_ns:
                    return "command-time-gap-or-order"
                if command.provenance != "expected":
                    return "future-command-must-be-expected"
        return None

    def _reply(self, request, status, forecasts=(), reason=None):
        return PredictionReply(request.identity, request.snapshot_id, request.request_id,
            request.model_id, request.capture_time_ns, request.state_time_ns,
            request.history_cutoff_ns, request.issued_ns, request.expiry_ns,
            request.horizon_steps, request.layout_id, request.layout_status,
            request.mode, request.sync_id, status, tuple(forecasts), reason,
            state_valid=request.state_valid, sync_valid=request.sync_valid)

    def _baseline(self, request, reason):
        forecasts = tuple(CandidateForecast(candidate.candidate_id,
            self.baseline.rollout(request.initial_state, candidate.commands, request.history_cutoff_ns),
            "analytical", reason) for candidate in request.candidates)
        return self._reply(request, "baseline", forecasts)

    def _process(self, request, generation):
        try:
            learned = None
            fallback_reason = "learned-prediction-unavailable"
            with self._lock:
                if request.request_id in self._cancelled:
                    fallback_reason = "cancelled"
                elif self.predictor is not None:
                    fallback_reason = None
            if fallback_reason is None:
                try:
                    learned = self.predictor(request)
                except Exception as error:
                    fallback_reason = "predictor-failure:" + type(error).__name__
            with self._lock:
                if generation != self._generation or request.identity != self.identity or self._closed:
                    return self._reply(request, "rejected", reason="late-reply-across-epoch")
                reason = self._reject_reason(request, self.clock())
                if reason:
                    return self._reply(request, "rejected", reason=reason)
                if request.request_id in self._cancelled:
                    return self._baseline(request, "cancelled")
                if fallback_reason is not None:
                    return self._baseline(request, fallback_reason)
                expected_ids = tuple(c.candidate_id for c in request.candidates)
                try:
                    if set(learned) != set(expected_ids):
                        return self._baseline(request, "learned-candidate-mismatch")
                    if any(not self._valid_learned(learned[key], request) for key in expected_ids):
                        return self._baseline(request, "learned-shape-time-or-kind-mismatch")
                except (TypeError, ValueError, AttributeError, KeyError):
                    return self._baseline(request, "learned-output-invalid")
                return self._reply(request, "forecast", tuple(CandidateForecast(key, learned[key], "learned")
                                                              for key in expected_ids))
        finally:
            with self._lock:
                self._pending -= 1
                self._cancelled.discard(request.request_id)

    def _valid_learned(self, rollout, request):
        return (self._valid_output(rollout, request) and rollout.kind == "learned-forecast"
                and len(rollout.saturated) == request.horizon_steps)

    def submit(self, request: PredictionRequest) -> Future:
        with self._lock:
            reason = self._reject_reason(request, self.clock())
            if reason:
                return _ready(self._reply(request, "rejected", reason=reason))
            self._seen = {key: expiry for key, expiry in self._seen.items() if expiry > self._last_now}
            if request.request_id in self._seen:
                return _ready(self._reply(request, "rejected", reason="duplicate-request"))
            if len(self._seen) >= 10_000:
                return _ready(self._reply(request, "rejected", reason="request-ledger-limit"))
            self._seen[request.request_id] = request.expiry_ns
            if self._pending >= self.limits.max_pending:
                return _ready(self._baseline(request, "overload-optional-work-shed"))
            self._pending += 1
            return self._pool.submit(self._process, request, self._generation)

    def predict(self, request: PredictionRequest, timeout_s: Optional[float] = None) -> PredictionReply:
        try:
            return self.submit(request).result(timeout=timeout_s)
        except TimeoutError:
            self.cancel(request.request_id)
            with self._lock:
                reason = self._reject_reason(request, self.clock())
                if reason:
                    return self._reply(request, "rejected", reason=reason)
                return self._baseline(request, "optional-inference-timeout")

    def cancel(self, request_id: str) -> None:
        with self._lock:
            if request_id in self._seen:
                self._cancelled.add(request_id)

    def reset(self, identity: RequestIdentity) -> None:
        with self._lock:
            self.identity = replace(identity, service_epoch=uuid.uuid4().hex)
            self._generation += 1
            self._seen.clear()
            self._cancelled.clear()
            self._last_now = None

    def close(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            self._generation += 1
        self._pool.shutdown(wait=wait)

    def reply_matches(self, reply: PredictionReply, request: PredictionRequest, now_ns: int) -> bool:
        """Consumer guard required even after serialization/network/restart."""
        return (not self._closed and reply.status in ("baseline", "forecast")
            and reply.identity == request.identity == self.identity
            and reply.request_id == request.request_id and reply.snapshot_id == request.snapshot_id
            and reply.model_id == request.model_id == self.model_id
            and reply.layout_id == request.layout_id and reply.layout_status == request.layout_status
            and reply.mode == request.mode and reply.sync_id == request.sync_id
            and reply.capture_time_ns == request.capture_time_ns
            and reply.state_time_ns == request.state_time_ns
            and reply.history_cutoff_ns == request.history_cutoff_ns
            and reply.issued_ns == request.issued_ns and reply.expiry_ns == request.expiry_ns
            and reply.horizon_steps == request.horizon_steps
            and reply.state_valid == request.state_valid and reply.sync_valid == request.sync_valid
            and now_ns >= request.issued_ns and now_ns < reply.expiry_ns
            and tuple(f.candidate_id for f in reply.forecasts) == tuple(c.candidate_id for c in request.candidates)
            and all(self._valid_output(f.rollout, request) for f in reply.forecasts)
            and reply.measured is False and reply.influences_commands is False)

    def _valid_output(self, rollout, request):
        return (isinstance(rollout, Rollout)
            and all(isinstance(s, EgoMechanismState) and all(math.isfinite(v) for v in s.vector())
                    for s in rollout.states)
            and rollout.measured is False and rollout.kind in ("analytical-forecast", "learned-forecast")
            and len(rollout.states) == request.horizon_steps
            and rollout.times_ns == tuple(request.history_cutoff_ns + (i + 1) * self.baseline.config.dt_ns
                                           for i in range(request.horizon_steps)))


def _ready(result) -> Future:
    future = Future()
    future.set_result(result)
    return future
