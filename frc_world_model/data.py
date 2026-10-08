"""Deterministic offline data helpers. This local envelope is NOT World-State's binding.

Records carry capture/accepted times and immutable provenance. Strict windows exclude
missing labels; adding masked losses is a separately named method change.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOCAL_SCHEMA_VERSION = "frc-world-model-local-log-v1-pending-binding"
SPLIT_NAMES = ("train", "validation", "calibration", "test")
IDENTITY_KEYS = ("robot_id", "map_id", "frame_id", "calibration_id", "source_id", "source_epoch", "reset_epoch",
                 "geometry_revision", "localization_revision", "configuration_revision", "sync_mapping_revision", "clock_domain")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class FeatureLayout:
    names: tuple[str, ...]
    units: tuple[str, ...]
    angle_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.units) or len(set(self.names)) != len(self.names):
            raise ValueError("layout requires unique names and one explicit unit per feature")
        if any(not name or not unit for name, unit in zip(self.names, self.units)):
            raise ValueError("empty feature name/unit")
        if len(set(self.angle_indices)) != len(self.angle_indices) or any(i < 0 or i >= len(self.names) for i in self.angle_indices):
            raise ValueError("invalid angle index")
        if any(self.units[i] != "rad" for i in self.angle_indices):
            raise ValueError("angle features must use radians")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureLayout":
        return cls(tuple(value["names"]), tuple(value["units"]), tuple(value.get("angle_indices", ())))


@dataclass(frozen=True)
class LogRecord:
    record_id: str
    kind: str
    session_id: str
    day: str
    identity: tuple[tuple[str, str], ...]
    timestamp_ns: int
    values: tuple[float | None, ...]
    valid: tuple[bool, ...]
    capture_time_ns: int | None = None
    interval_end_ns: int | None = None
    label_provenance: str | None = None
    execution: tuple[tuple[str, Any], ...] = ()
    sync_error_ns: int | None = None
    sync_limit_ns: int | None = None
    wire_provenance_json: str = ""

    @property
    def episode_key(self) -> tuple[Any, ...]:
        # Epoch or calibration/frame/robot changes can never share recurrent carry.
        return (self.session_id, self.day, self.identity)

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionReport:
    records: tuple[LogRecord, ...]
    original_path: str
    original_sha256: str
    original_bytes: int
    line_count: int
    exclusions: dict[str, int]
    detections: dict[str, int]
    layout_hash: str
    dataset_hash: str

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": LOCAL_SCHEMA_VERSION,
            "production_binding": "pending",
            "original": {"path": self.original_path, "sha256": self.original_sha256, "bytes": self.original_bytes, "lines": self.line_count},
            "accepted_records": len(self.records), "exclusions": self.exclusions,
            "detections": self.detections, "layout_hash": self.layout_hash,
            "dataset_hash": self.dataset_hash,
            "label_provenance": dict(Counter(r.label_provenance for r in self.records if r.kind == "state")),
        }


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _parse_record(raw: Mapping[str, Any], layout: FeatureLayout, action_layout: FeatureLayout) -> LogRecord:
    if raw.get("schema_version") != LOCAL_SCHEMA_VERSION:
        raise ValueError("schema_version")
    kind = raw.get("kind")
    if kind not in ("state", "action"):
        raise ValueError("kind")
    record_id, session_id, day = raw.get("record_id"), raw.get("session_id"), raw.get("day")
    if not isinstance(record_id, str) or not record_id or not isinstance(session_id, str) or not session_id:
        raise ValueError("record/session identity")
    if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
        raise ValueError("day must be ISO date")
    identity = raw.get("identity", {})
    if any(k not in identity or not isinstance(identity[k], str) or not identity[k] for k in IDENTITY_KEYS):
        raise ValueError("missing identity prerequisite")
    if raw.get("time_sync_valid") is not True:
        raise ValueError("invalid_time_sync")
    sync_error = _integer(raw.get("time_sync_error_ns"), "time_sync_error_ns")
    sync_limit = _integer(raw.get("time_sync_limit_ns"), "time_sync_limit_ns")
    if sync_error > sync_limit:
        raise ValueError("time_sync_error_exceeds_limit")
    expected = layout if kind == "state" else action_layout
    if raw.get("layout_hash") != canonical_hash(asdict(expected)):
        raise ValueError("layout_hash")
    values, valid = raw.get("values"), raw.get("valid")
    if not isinstance(values, list) or not isinstance(valid, list) or len(values) != len(expected.names) or len(valid) != len(values):
        raise ValueError("vector_shape")
    parsed: list[float | None] = []
    for value, is_valid in zip(values, valid):
        if type(is_valid) is not bool:
            raise ValueError("mask_type")
        if is_valid:
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("nonfinite_valid_value")
            parsed.append(float(value))
        else:
            # Invalid numeric values are discarded, never treated as a training label.
            parsed.append(None)
    if kind == "state":
        timestamp = _integer(raw.get("state_time_ns"), "state_time_ns")
        capture = _integer(raw.get("capture_time_ns"), "capture_time_ns")
        if capture > timestamp:
            raise ValueError("capture_after_state")
        provenance = raw.get("label_provenance")
        if provenance == "future_smoothed" or raw.get("input_causal") is not True:
            raise ValueError("future_smoothed_input")
        if provenance not in ("measured", "fused_estimate", "independent_ground_truth", "synthetic_fixture"):
            raise ValueError("label_provenance")
        end = None
        execution = ()
    else:
        if raw.get("accepted") is not True:
            raise ValueError("action_not_accepted")
        timestamp = _integer(raw.get("accepted_time_ns"), "accepted_time_ns")
        end = _integer(raw.get("interval_end_ns"), "interval_end_ns")
        if end <= timestamp:
            raise ValueError("empty_action_interval")
        capture, provenance = None, None
        details = raw.get("execution", {})
        if not isinstance(details, dict):
            raise ValueError("execution_metadata")
        # Optional actual clipping/saturation/deviation metadata is preserved verbatim.
        canonical_hash(details)
        execution = tuple(sorted(details.items()))
    return LogRecord(record_id, kind, session_id, day, tuple((k, identity[k]) for k in IDENTITY_KEYS), timestamp,
                     tuple(parsed), tuple(valid), capture, end, provenance, execution,
                     sync_error, sync_limit,
                     json.dumps(raw.get("wire_provenance",{}),sort_keys=True,separators=(",",":"),allow_nan=False))


def load_jsonl(path: str | Path, layout: FeatureLayout, action_layout: FeatureLayout) -> IngestionReport:
    """Read-only ingestion. Detect arrival-order defects, then sort by causal time.

    Conflicting same-time records are all excluded. Exact duplicates are collapsed.
    An overlapping action interval is excluded, rather than guessed/averaged.
    """
    source = Path(path).resolve()
    payload = source.read_bytes()
    lines = payload.decode("utf-8").splitlines()
    exclusions: Counter[str] = Counter()
    detections: Counter[str] = Counter()
    candidates: list[LogRecord] = []
    seen_ids: dict[str, str] = {}
    bad_ids: set[str] = set()
    latest: dict[tuple[Any, ...], int] = {}
    epochs: dict[tuple[str, str], set[tuple[tuple[str, str], ...]]] = defaultdict(set)
    for line in lines:
        if not line.strip():
            exclusions["blank_line"] += 1
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("record_object")
            record = _parse_record(raw, layout, action_layout)
        except (ValueError, TypeError, KeyError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            exclusions[reason] += 1
            continue
        digest = canonical_hash(record.serializable())
        if record.record_id in seen_ids:
            if seen_ids[record.record_id] == digest:
                detections["duplicate_record_id"] += 1
                exclusions["exact_duplicate"] += 1
            else:
                bad_ids.add(record.record_id)
                detections["conflicting_record_id"] += 1
                exclusions["conflicting_record_id"] += 1
            continue
        seen_ids[record.record_id] = digest
        arrival_key = (record.episode_key, record.kind)
        if record.timestamp_ns < latest.get(arrival_key, -1):
            detections["out_of_order"] += 1
        latest[arrival_key] = max(record.timestamp_ns, latest.get(arrival_key, -1))
        epochs[(record.session_id, record.day)].add(record.identity)
        candidates.append(record)
    detections["epoch_or_identity_changes"] = sum(max(0, len(v) - 1) for v in epochs.values())
    buckets: dict[tuple[Any, ...], list[LogRecord]] = defaultdict(list)
    for record in candidates:
        if record.record_id in bad_ids:
            exclusions["conflicting_record_id_original"] += 1
        else:
            buckets[(record.episode_key, record.kind, record.timestamp_ns)].append(record)
    accepted: list[LogRecord] = []
    for bucket in buckets.values():
        if len(bucket) == 1:
            accepted.extend(bucket)
        elif len({canonical_hash({k: v for k, v in r.serializable().items() if k != "record_id"}) for r in bucket}) == 1:
            accepted.append(min(bucket, key=lambda r: r.record_id))
            exclusions["duplicate_timestamp"] += len(bucket) - 1
            detections["duplicate_timestamp"] += len(bucket) - 1
        else:
            exclusions["conflicting_timestamp"] += len(bucket)
            detections["conflicting_timestamp"] += 1
    ordered = sorted(accepted, key=lambda r: (r.episode_key, r.timestamp_ns, r.kind, r.record_id))
    overlapping_ids: set[str] = set()
    previous: dict[tuple[Any, ...], LogRecord] = {}
    for record in ordered:
        if record.kind != "action":
            continue
        prior = previous.get(record.episode_key)
        if prior and record.timestamp_ns < prior.interval_end_ns:
            overlapping_ids.update((prior.record_id, record.record_id))
            detections["overlapping_action_intervals"] += 1
        if prior is None or record.interval_end_ns > prior.interval_end_ns:
            previous[record.episode_key] = record
    exclusions["overlapping_action_intervals"] += len(overlapping_ids)
    records = tuple(r for r in ordered if r.record_id not in overlapping_ids)
    return IngestionReport(records, str(source), hashlib.sha256(payload).hexdigest(), len(payload), len(lines),
                           dict(sorted(exclusions.items())), dict(sorted(detections.items())),
                           canonical_hash({"state": asdict(layout), "action": asdict(action_layout)}),
                           canonical_hash([r.serializable() for r in records]))


@dataclass(frozen=True)
class SplitConfig:
    ratios: tuple[float, ...] = (0.6, 0.15, 0.15, 0.1)
    seed: str = "frc-world-model-v1"
    group_by: str = "day"

    def __post_init__(self) -> None:
        if len(self.ratios) != 4 or any(not math.isfinite(r) or r < 0 for r in self.ratios) or not math.isclose(sum(self.ratios), 1.0):
            raise ValueError("four finite nonnegative split ratios must sum to one")
        if self.group_by not in ("day", "session"):
            raise ValueError("group_by must be day or session")


@dataclass(frozen=True)
class DatasetSplit:
    records: Mapping[str, tuple[LogRecord, ...]]
    groups: Mapping[str, tuple[str, ...]]
    config: SplitConfig

    def manifest(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "config_hash": canonical_hash(asdict(self.config)),
                "groups": dict(self.groups), "record_counts": {k: len(v) for k, v in self.records.items()}}


def split_sessions(records: Iterable[LogRecord] | IngestionReport, config: SplitConfig = SplitConfig()) -> DatasetSplit:
    """Split complete days (default) or sessions BEFORE window construction."""
    if isinstance(records, IngestionReport):
        records = records.records
    records = tuple(records)
    groups: dict[str, list[LogRecord]] = defaultdict(list)
    session_days: dict[str, set[str]] = defaultdict(set)
    for record in records:
        session_days[record.session_id].add(record.day)
    # Sessions spanning midnight connect those days, so neither day nor session leaks.
    parent = {r.day: r.day for r in records}
    def root(day: str) -> str:
        while parent[day] != day:
            parent[day] = parent[parent[day]]
            day = parent[day]
        return day
    if config.group_by == "day":
        for days in session_days.values():
            first = min(days)
            for day in days:
                a, b = root(first), root(day)
                parent[max(a, b)] = min(a, b)
    for record in records:
        key = root(record.day) if config.group_by == "day" else record.session_id
        groups[key].append(record)
    keys = sorted(groups, key=lambda key: (canonical_hash([config.seed, key]), key))
    exact = [len(keys) * r for r in config.ratios]
    counts = [int(v) for v in exact]
    for i in sorted(range(4), key=lambda i: (-(exact[i] - counts[i]), i))[:len(keys) - sum(counts)]:
        counts[i] += 1
    output: dict[str, tuple[LogRecord, ...]] = {}
    membership: dict[str, tuple[str, ...]] = {}
    offset = 0
    for name, count in zip(SPLIT_NAMES, counts):
        selected = keys[offset:offset + count]
        membership[name] = tuple(selected)
        output[name] = tuple(r for key in selected for r in groups[key])
        offset += count
    return DatasetSplit(output, membership, config)


@dataclass(frozen=True)
class WindowConfig:
    timestep_ns: int
    history_steps: int
    horizon_steps: int
    max_state_age_ns: int = 0
    max_action_age_ns: int | None = None

    def __post_init__(self) -> None:
        if type(self.timestep_ns) is not int or self.timestep_ns <= 0 or type(self.history_steps) is not int or self.history_steps <= 0 or type(self.horizon_steps) is not int or self.horizon_steps <= 0:
            raise ValueError("positive integer timestep/history/horizon required")
        if type(self.max_state_age_ns) is not int or not 0 <= self.max_state_age_ns < self.timestep_ns:
            raise ValueError("state age tolerance must be below one timestep")
        if self.max_action_age_ns is not None and (type(self.max_action_age_ns) is not int or self.max_action_age_ns < 0):
            raise ValueError("invalid action age limit")


@dataclass(frozen=True)
class Window:
    window_id: str
    episode_key: tuple[Any, ...]
    cutoff_ns: int
    timestep_ns: int
    history: tuple[LogRecord, ...]
    history_actions: tuple[LogRecord, ...]
    actions: tuple[LogRecord, ...]
    targets: tuple[LogRecord, ...]

    @property
    def initial_state(self) -> tuple[float, ...]:
        return self.history[-1].values


@dataclass(frozen=True)
class WindowReport:
    windows: tuple[Window, ...]
    exclusions: Mapping[str, int]
    detections: Mapping[str, int]
    config: WindowConfig

    def manifest(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "config_hash": canonical_hash(asdict(self.config)),
                "windows": len(self.windows), "exclusions": dict(self.exclusions), "detections": dict(self.detections),
                "window_hash": canonical_hash([w.window_id for w in self.windows])}


def build_windows(records: Iterable[LogRecord], config: WindowConfig) -> WindowReport:
    """Strict full-label windows with ZOH causal states and whole-interval actions.

    Commands must cover an entire grid interval; a mid-interval command transition
    causes exclusion. No averaging of mode setpoints or future-state interpolation.
    """
    episodes: dict[tuple[Any, ...], list[LogRecord]] = defaultdict(list)
    for record in records:
        episodes[record.episode_key].append(record)
    exclusions: Counter[str] = Counter()
    detections: Counter[str] = Counter()
    windows: list[Window] = []
    dt = config.timestep_ns
    for key, episode in sorted(episodes.items()):
        states = sorted((r for r in episode if r.kind == "state"), key=lambda r: r.timestamp_ns)
        actions = sorted((r for r in episode if r.kind == "action"), key=lambda r: r.timestamp_ns)
        detections["episodes"] += 1
        detections["state_gaps"] += sum(b.timestamp_ns - a.timestamp_ns > dt + config.max_state_age_ns for a, b in zip(states, states[1:]))
        detections["action_gaps"] += sum(a.interval_end_ns < b.timestamp_ns for a, b in zip(actions, actions[1:]))
        if not states or not actions:
            exclusions["episode_missing_state_or_action"] += 1
            continue
        state_times, action_times = [r.timestamp_ns for r in states], [r.timestamp_ns for r in actions]
        first = ((states[0].timestamp_ns + dt - 1) // dt) * dt + (config.history_steps - 1) * dt
        last = states[-1].timestamp_ns - config.horizon_steps * dt
        for cutoff in range(first, last + 1, dt):
            def state_at(timestamp: int) -> LogRecord:
                i = bisect_right(state_times, timestamp) - 1
                if i < 0 or timestamp - states[i].timestamp_ns > config.max_state_age_ns:
                    raise ValueError("state_gap_or_alignment")
                record = states[i]
                if not all(record.valid):
                    raise ValueError("unknown_state_or_target")
                return record
            def action_at(timestamp: int) -> LogRecord:
                i = bisect_right(action_times, timestamp) - 1
                if i < 0 or actions[i].interval_end_ns < timestamp + dt:
                    raise ValueError("action_gap_or_midstep_transition")
                record = actions[i]
                if config.max_action_age_ns is not None and timestamp - record.timestamp_ns > config.max_action_age_ns:
                    raise ValueError("stale_action")
                if not all(record.valid):
                    raise ValueError("unknown_action")
                return record
            try:
                history = tuple(state_at(cutoff - i * dt) for i in range(config.history_steps - 1, -1, -1))
                targets = tuple(state_at(cutoff + i * dt) for i in range(1, config.horizon_steps + 1))
                history_actions = tuple(action_at(cutoff - i * dt) for i in range(config.history_steps - 1, 0, -1))
                future_actions = tuple(action_at(cutoff + i * dt) for i in range(config.horizon_steps))
            except ValueError as exc:
                exclusions[str(exc)] += 1
                continue
            identifier = canonical_hash({"episode": key, "cutoff_ns": cutoff, "config": asdict(config),
                                         "record_ids": [r.record_id for r in history + history_actions + future_actions + targets]})
            windows.append(Window(identifier, key, cutoff, dt, history, history_actions, future_actions, targets))
    return WindowReport(tuple(windows), dict(exclusions), dict(detections), config)


@dataclass(frozen=True)
class ReversibleNormalizer:
    layout: FeatureLayout
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    fit_count: int
    training_values_hash: str
    angle_method: str = "wrapped_delta_about_train_circular_mean"

    @classmethod
    def fit(cls, values: Iterable[Sequence[float | None]], layout: FeatureLayout, *, split: str, min_scale: float = 1e-6) -> "ReversibleNormalizer":
        if split != "train":
            raise ValueError("normalization may be fitted on train only")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("positive finite min_scale required")
        rows = tuple(tuple(v) for v in values)
        if not rows:
            raise ValueError("no train values")
        if any(len(row) != len(layout.names) for row in rows):
            raise ValueError("normalizer vector_shape")
        centers, scales = [], []
        for i in range(len(layout.names)):
            column = [float(row[i]) for row in rows if row[i] is not None]
            if not column or any(not math.isfinite(v) for v in column):
                raise ValueError("no finite train labels for feature")
            if i in layout.angle_indices:
                s, c = sum(math.sin(v) for v in column), sum(math.cos(v) for v in column)
                if math.hypot(s, c) / len(column) < 1e-8:
                    raise ValueError("ambiguous circular mean; choose a documented angle representation")
                center = math.atan2(s, c)
                differences = [wrap_angle(v - center) for v in column]
            else:
                center = sum(column) / len(column)
                differences = [v - center for v in column]
            centers.append(center)
            scales.append(max(min_scale, math.sqrt(sum(d * d for d in differences) / len(differences))))
        return cls(layout, tuple(centers), tuple(scales), len(rows), canonical_hash(rows))

    def transform(self, values: Sequence[float | None]) -> tuple[float | None, ...]:
        if len(values) != len(self.centers):
            raise ValueError("normalizer vector_shape")
        output = []
        for i, value in enumerate(values):
            if value is None:
                output.append(None)
                continue
            if not math.isfinite(value):
                raise ValueError("nonfinite normalization input")
            delta = value - self.centers[i]
            output.append((wrap_angle(delta) if i in self.layout.angle_indices else delta) / self.scales[i])
        return tuple(output)

    def inverse(self, values: Sequence[float | None]) -> tuple[float | None, ...]:
        if len(values) != len(self.centers):
            raise ValueError("normalizer vector_shape")
        output = []
        for i, value in enumerate(values):
            if value is None:
                output.append(None)
                continue
            if not math.isfinite(value):
                raise ValueError("nonfinite normalized value")
            raw = value * self.scales[i] + self.centers[i]
            output.append(wrap_angle(raw) if i in self.layout.angle_indices else raw)
        return tuple(output)

    def metadata(self) -> dict[str, Any]:
        return {**asdict(self), "fit_split": "train", "method_change": "external_train_only_preprocessing"}


def prepare_offline(report: IngestionReport, state_layout: FeatureLayout, action_layout: FeatureLayout,
                    window_config: WindowConfig, split_config: SplitConfig = SplitConfig()) -> tuple[DatasetSplit, dict[str, WindowReport], ReversibleNormalizer, ReversibleNormalizer]:
    """Lightweight orchestration only; never starts training or allocates a device."""
    splits = split_sessions(report, split_config)
    windows = {name: build_windows(records, window_config) for name, records in splits.records.items()}
    # Fit raw train records once, not repeated window occurrences.
    state_norm = ReversibleNormalizer.fit((r.values for r in splits.records["train"] if r.kind == "state"), state_layout, split="train")
    action_norm = ReversibleNormalizer.fit((r.values for r in splits.records["train"] if r.kind == "action"), action_layout, split="train")
    return splits, windows, state_norm, action_norm
