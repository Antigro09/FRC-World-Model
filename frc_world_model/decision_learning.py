"""Bounded offline task-outcome plumbing, independent of live decision authority.

Candidate commands available at the cutoff are prediction inputs. Commands
accepted during later execution and outcome labels are deliberately separate.
The fitted table below is an empirical comparison baseline, never a qualified
learned forecast, calibrated covariance, or a counterfactual policy evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

from .data import IDENTITY_KEYS, SPLIT_NAMES, SplitConfig, canonical_hash

OUTCOME_SCHEMA = "frc-decision-outcome-log/1"
MAX_RECORDS = 2048
MAX_BYTES = 16 * 1024 * 1024
MAX_SAMPLES = 64
MAX_HORIZON_US = 30_000_000
TASK_KINDS = ("PICKUP", "SCORE", "TRANSIT", "HOLD")
PROVENANCES = ("measured", "independent_ground_truth", "synthetic_fixture")


def _keys(raw: Any, required: Iterable[str], optional: Iterable[str] = ()) -> None:
    if not isinstance(raw, dict) or not set(required) <= set(raw) or set(raw) - set(required) - set(optional):
        raise ValueError("missing or unknown fields")


def _id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("bounded nonempty identity required")
    return value


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be a valid integer")
    return value


def _number(value: Any, name: str, minimum: float = -1e6, maximum: float = 1e6) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(name + " must be finite and bounded")
    return float(value)


def _vector(raw: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or len(raw) != size:
        raise ValueError(name + " shape")
    return tuple(_number(v, name) for v in raw)


def _identity(raw: Any) -> tuple[tuple[str, str], ...]:
    _keys(raw, IDENTITY_KEYS)
    result = tuple((key, _id(raw[key])) for key in IDENTITY_KEYS)
    if raw["frame_id"] != "BLUE_FIELD" or raw["clock_domain"] != "ROBOT_MONOTONIC_US":
        raise ValueError("explicit field frame and robot clock required")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


@dataclass(frozen=True)
class CommandSample:
    offset_us: int
    body_twist: tuple[float, float, float]
    mechanisms: tuple[tuple[str, str | bool | float], ...]
    accepted_us: int | None = None
    applied_body_twist: tuple[float, float, float] | None = None


def _commands(raw: Any, step_us: int, horizon_us: int, cutoff_us: int, *, executed: bool) -> tuple[CommandSample, ...]:
    if not isinstance(raw, list) or len(raw) != horizon_us // step_us or not 1 <= len(raw) <= MAX_SAMPLES:
        raise ValueError("complete bounded command grid required")
    result = []
    for index, sample in enumerate(raw):
        _keys(sample, ("offset_us", "body_twist", "mechanisms", "accepted_us") if executed else
              ("offset_us", "body_twist", "mechanisms"), ("applied_body_twist",) if executed else ())
        offset = _int(sample["offset_us"], "offset_us")
        if offset != index * step_us:
            raise ValueError("command offset grid")
        mechanisms = sample["mechanisms"]
        if not isinstance(mechanisms, dict) or len(mechanisms) > 16:
            raise ValueError("bounded mechanism commands required")
        parsed = []
        for name, value in sorted(mechanisms.items()):
            _id(name)
            if type(value) is bool:
                parsed.append((name, value))
            elif isinstance(value, str):
                parsed.append((name, _id(value)))
            else:
                parsed.append((name, _number(value, "mechanism command")))
        accepted = _int(sample["accepted_us"], "accepted_us") if executed else None
        # Acceptance can precede this interval, but cannot be recorded afterward.
        if executed and accepted > cutoff_us + offset:
            raise ValueError("noncausal accepted command")
        applied = _vector(sample["applied_body_twist"], 3, "applied twist") if "applied_body_twist" in sample else None
        result.append(CommandSample(offset, _vector(sample["body_twist"], 3, "body twist"), tuple(parsed), accepted, applied))
    return tuple(result)


@dataclass(frozen=True)
class CausalTaskInput:
    """No labels, later accepted commands, or label-source identity are exposed."""
    record_id: str
    session_id: str
    day: str
    identity: tuple[tuple[str, str], ...]
    request_id: str
    snapshot_id: int
    candidate_id: str
    task_id: str
    task_kind: str
    cutoff_us: int
    horizon_us: int
    step_us: int
    observed_us: int
    ego_state: tuple[float, ...]  # x, y, heading, body vx, body vy, omega; SI
    possession: str
    possession_observed_us: int
    mechanism_ready: bool | None
    mechanism_observed_us: int
    goal_xy_m: tuple[float, float]
    target_ids: tuple[str, ...]
    path_length_m: float
    minimum_clearance_m: float
    time_remaining_s: float
    commands: tuple[CommandSample, ...]


@dataclass(frozen=True)
class TaskOutcome:
    observed_us: int
    success: bool
    risk_event: bool
    utility: float
    provenance: str
    label_source_id: str
    utility_definition_id: str
    risk_definition_id: str


@dataclass(frozen=True)
class DecisionRecord:
    inputs: CausalTaskInput
    executed_commands: tuple[CommandSample, ...]
    outcome: TaskOutcome

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DecisionRecord":
        _keys(raw, ("schema_version", "record_id", "session_id", "day", "identity", "request_id", "snapshot_id",
                    "candidate", "cutoff_us", "horizon_us", "step_us", "context", "commands_created_us",
                    "candidate_commands", "executed_commands", "outcome", "time_sync_valid",
                    "time_sync_error_us", "time_sync_limit_us"))
        if raw["schema_version"] != OUTCOME_SCHEMA or raw["time_sync_valid"] is not True:
            raise ValueError("schema or synchronization invalid")
        if _int(raw["time_sync_error_us"], "sync error") > _int(raw["time_sync_limit_us"], "sync limit"):
            raise ValueError("synchronization limit exceeded")
        day = raw["day"]
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
            raise ValueError("ISO day required")
        identity = _identity(raw["identity"])
        cutoff = _int(raw["cutoff_us"], "cutoff_us")
        horizon, step = _int(raw["horizon_us"], "horizon_us", 1), _int(raw["step_us"], "step_us", 1)
        if horizon > MAX_HORIZON_US or horizon % step:
            raise ValueError("bounded integral horizon required")
        if _int(raw["commands_created_us"], "commands_created_us") > cutoff:
            raise ValueError("future candidate command leakage")
        candidate, context, outcome = raw["candidate"], raw["context"], raw["outcome"]
        _keys(candidate, ("candidate_id", "task_id", "task_kind", "goal_xy_m", "target_ids", "path_length_m"))
        if candidate["task_kind"] not in TASK_KINDS:
            raise ValueError("unsupported task kind")
        targets = candidate["target_ids"]
        if not isinstance(targets, list) or len(targets) > 16 or len(set(targets)) != len(targets):
            raise ValueError("bounded unique target identities required")
        targets = tuple(_id(t) for t in targets)
        _keys(context, ("input_causal", "observed_us", "ego_state", "possession", "possession_observed_us",
                        "mechanism_ready", "mechanism_observed_us", "minimum_clearance_m", "time_remaining_s"))
        if context["input_causal"] is not True or context["possession"] not in ("EMPTY", "HELD", "UNKNOWN"):
            raise ValueError("causal measured possession context required")
        if context["mechanism_ready"] is not None and type(context["mechanism_ready"]) is not bool:
            raise ValueError("readiness must be bool or unknown")
        observed = tuple(_int(context[k], k) for k in ("observed_us", "possession_observed_us", "mechanism_observed_us"))
        if any(t > cutoff for t in observed):
            raise ValueError("future context leakage")
        commands = _commands(raw["candidate_commands"], step, horizon, cutoff, executed=False)
        executed = _commands(raw["executed_commands"], step, horizon, cutoff, executed=True)
        _keys(outcome, ("observed_us", "identity", "success", "risk_event", "utility", "provenance",
                        "label_source_id", "utility_definition_id", "risk_definition_id"))
        if _identity(outcome["identity"]) != identity:
            raise ValueError("outcome crossed source/reset/map identity")
        outcome_time = _int(outcome["observed_us"], "outcome observed_us")
        if outcome_time < cutoff + horizon:
            raise ValueError("outcome does not cover the forecast horizon")
        if type(outcome["success"]) is not bool or type(outcome["risk_event"]) is not bool:
            raise ValueError("observed task labels required; unknown is not false")
        if outcome["provenance"] not in PROVENANCES:
            raise ValueError("explicit measured/ground-truth/synthetic provenance required")
        inputs = CausalTaskInput(_id(raw["record_id"]), _id(raw["session_id"]), day, identity, _id(raw["request_id"]),
                                _int(raw["snapshot_id"], "snapshot_id"), _id(candidate["candidate_id"]),
                                _id(candidate["task_id"]), candidate["task_kind"], cutoff, horizon, step,
                                observed[0], _vector(context["ego_state"], 6, "ego state"), context["possession"],
                                observed[1], context["mechanism_ready"], observed[2],
                                _vector(candidate["goal_xy_m"], 2, "goal xy"), targets,
                                _number(candidate["path_length_m"], "path length", 0),
                                _number(context["minimum_clearance_m"], "clearance", 0),
                                _number(context["time_remaining_s"], "time remaining", 0), commands)
        labels = TaskOutcome(outcome_time, outcome["success"], outcome["risk_event"],
                             _number(outcome["utility"], "utility"), outcome["provenance"],
                             _id(outcome["label_source_id"]), _id(outcome["utility_definition_id"]),
                             _id(outcome["risk_definition_id"]))
        return cls(inputs, executed, labels)


def _bounded(records: Iterable[DecisionRecord], maximum: int = MAX_RECORDS) -> tuple[DecisionRecord, ...]:
    if type(maximum) is not int or not 1 <= maximum <= MAX_RECORDS:
        raise ValueError("record bound exceeds offline ceiling")
    result, seen, decisions = [], {}, {}
    for source_index, record in enumerate(records):
        if not isinstance(record, DecisionRecord):
            raise ValueError("validated DecisionRecord required")
        if source_index >= maximum:
            raise ValueError("record capacity exceeded")
        digest = canonical_hash(asdict(record))
        identifier = record.inputs.record_id
        key = (record.inputs.session_id, record.inputs.identity, record.inputs.cutoff_us, record.inputs.candidate_id)
        if (identifier in seen and seen[identifier] != digest) or (key in decisions and decisions[key] != digest):
            raise ValueError("conflicting record or decision identity")
        if identifier in seen:
            continue
        seen[identifier], decisions[key] = digest, digest
        result.append(record)
    return tuple(sorted(result, key=lambda row: (row.inputs.session_id, row.inputs.day,
                                                row.inputs.cutoff_us, row.inputs.record_id)))


def load_decision_jsonl(path: str | Path, *, max_records: int = MAX_RECORDS, max_bytes: int = MAX_BYTES) -> tuple[tuple[DecisionRecord, ...], dict[str, Any]]:
    """Strict read-only ingestion. Any invalid line rejects the dataset."""
    path = Path(path)
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES or path.stat().st_size > max_bytes:
        raise ValueError("bounded input file required")
    with path.open("rb") as source:
        payload = source.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("input grew past file limit")
    lines = payload.decode("utf-8").splitlines()
    if len(lines) > max_records:
        raise ValueError("record capacity exceeded")
    parsed = []
    for number, line in enumerate(lines, 1):
        try:
            raw = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
            parsed.append(DecisionRecord.from_dict(raw))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"invalid outcome line {number}: {exc}") from exc
    records = _bounded(parsed, max_records)
    import hashlib
    return records, {"schema_version": OUTCOME_SCHEMA, "original_path": str(path.resolve()),
                     "original_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                     "lines": len(lines), "accepted_records": len(records), "duplicates": len(lines) - len(records),
                     "dataset_hash": canonical_hash([asdict(r) for r in records]),
                     "label_provenance": dict(Counter(r.outcome.provenance for r in records))}


@dataclass(frozen=True)
class OutcomeSplit:
    records: Mapping[str, tuple[DecisionRecord, ...]]
    group_membership: Mapping[str, tuple[str, ...]]
    config: SplitConfig

    def manifest(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "groups": dict(self.group_membership),
                "counts": {name: len(rows) for name, rows in self.records.items()},
                "split_hash": canonical_hash({name: [canonical_hash(asdict(r)) for r in rows] for name, rows in self.records.items()})}


def split_decision_sessions(records: Iterable[DecisionRecord], config: SplitConfig = SplitConfig()) -> OutcomeSplit:
    """Day grouping joins sessions crossing midnight before any fitting."""
    rows = _bounded(records)
    session_days = defaultdict(set)
    for row in rows:
        session_days[row.inputs.session_id].add(row.inputs.day)
    parent = {r.inputs.day: r.inputs.day for r in rows}

    def root(day):
        while parent[day] != day:
            parent[day] = parent[parent[day]]
            day = parent[day]
        return day

    if config.group_by == "day":
        for days in session_days.values():
            for day in days:
                a, b = root(min(days)), root(day)
                parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for row in rows:
        groups[root(row.inputs.day) if config.group_by == "day" else row.inputs.session_id].append(row)
    keys = sorted(groups, key=lambda k: (canonical_hash([config.seed, k]), k))
    exact = [len(keys) * ratio for ratio in config.ratios]
    counts = [int(x) for x in exact]
    for index in sorted(range(4), key=lambda i: (-(exact[i] - counts[i]), i))[:len(keys) - sum(counts)]:
        counts[index] += 1
    output, membership, offset = {}, {}, 0
    for name, count in zip(SPLIT_NAMES, counts):
        selected = keys[offset:offset + count]
        membership[name] = tuple(selected)
        output[name] = tuple(sorted((r for key in selected for r in groups[key]), key=lambda r: r.inputs.record_id))
        offset += count
    return OutcomeSplit(output, membership, config)


@dataclass(frozen=True)
class OutcomePrediction:
    utility: float
    success_probability: float
    risk_probability: float
    provenance: str = "offline_empirical_table"
    support_examples: int = 0

    def __post_init__(self):
        _number(self.utility, "predicted utility")
        _number(self.success_probability, "success probability", 0, 1)
        _number(self.risk_probability, "risk probability", 0, 1)
        _id(self.provenance)
        _int(self.support_examples, "support_examples")


def _cell(inputs: CausalTaskInput) -> tuple[Any, ...]:
    # Explicit coarse command conditioning; this cannot represent dynamics.
    speed = sum(math.hypot(*s.body_twist[:2]) for s in inputs.commands) / len(inputs.commands)
    turn = sum(abs(s.body_twist[2]) for s in inputs.commands) / len(inputs.commands)
    mechanism_modes = tuple(s.mechanisms for s in inputs.commands)
    domain = tuple((k, v) for k, v in inputs.identity if k not in ("source_epoch", "reset_epoch"))
    return (canonical_hash(domain), inputs.task_kind, inputs.possession, inputs.mechanism_ready, inputs.horizon_us,
            math.floor(inputs.path_length_m), math.floor(speed), math.floor(turn), canonical_hash(mechanism_modes))


def _execution_matches(row: DecisionRecord) -> bool:
    """Labels describe candidate commands only when accepted/applied grids agree."""
    return all((a.body_twist, a.mechanisms) == (b.body_twist, b.mechanisms) and
               (b.applied_body_twist is None or b.applied_body_twist == a.body_twist)
               for a, b in zip(row.inputs.commands, row.executed_commands))


@dataclass(frozen=True)
class EmpiricalOutcomeBaseline:
    cells: tuple[tuple[tuple[Any, ...], OutcomePrediction], ...]
    train_sessions: tuple[str, ...]
    train_days: tuple[str, ...]
    train_record_ids: tuple[str, ...]
    training_hash: str
    utility_definition_id: str
    risk_definition_id: str
    model_id: str
    report: Mapping[str, Any]

    def predict(self, inputs: CausalTaskInput) -> OutcomePrediction | None:
        if inputs.possession == "UNKNOWN" or inputs.mechanism_ready is None:
            return None
        return dict(self.cells).get(_cell(inputs))  # Unseen bins abstain.


def fit_outcome_baseline(records: Iterable[DecisionRecord], *, split: str, min_examples: int = 2) -> EmpiricalOutcomeBaseline:
    """Train-only bounded empirical table; no optimizer, NN package or promotion."""
    if split != "train" or type(min_examples) is not int or not 1 <= min_examples <= MAX_RECORDS:
        raise ValueError("bounded train-only fit required")
    started = time.perf_counter()
    rows = _bounded(records)
    if not rows:
        raise ValueError("no training outcomes")
    definitions = {(r.outcome.utility_definition_id, r.outcome.risk_definition_id) for r in rows}
    if len(definitions) != 1:
        raise ValueError("utility/risk definitions must not be mixed")
    buckets, exclusions = defaultdict(list), Counter()
    for row in rows:
        if row.inputs.possession == "UNKNOWN" or row.inputs.mechanism_ready is None:
            exclusions["unknown_measured_context"] += 1
            continue
        if not _execution_matches(row):
            # No label is attributed to a command sequence the robot did not accept/apply.
            exclusions["candidate_execution_deviation"] += 1
            continue
        buckets[_cell(row.inputs)].append(row.outcome)
    cells = []
    for key, labels in sorted(buckets.items(), key=lambda item: repr(item[0])):
        if len(labels) < min_examples:
            exclusions["insufficient_bin_support"] += len(labels)
            continue
        cells.append((key, OutcomePrediction(sum(x.utility for x in labels) / len(labels),
                     sum(x.success for x in labels) / len(labels), sum(x.risk_event for x in labels) / len(labels),
                     support_examples=len(labels))))
    digest = canonical_hash([asdict(r) for r in rows])
    utility_id, risk_id = next(iter(definitions))
    model_id = "offline-task-table-" + canonical_hash([digest, [(key, asdict(value)) for key, value in cells], min_examples])[:16]
    report = {"training_records": len(rows), "fitted_bins": len(cells), "min_examples": min_examples,
              "exclusions": dict(exclusions), "label_provenance": dict(Counter(r.outcome.provenance for r in rows)),
              "duration_s": time.perf_counter() - started, "device": "cpu", "optimizer_updates": 0,
              "advisory_only": True, "qualified_for_live": False,
              "claims": "empirical comparison baseline; no physical uncertainty or counterfactual ranking evidence"}
    return EmpiricalOutcomeBaseline(tuple(cells), tuple(sorted({r.inputs.session_id for r in rows})),
             tuple(sorted({r.inputs.day for r in rows})), tuple(sorted(r.inputs.record_id for r in rows)),
             digest, utility_id, risk_id, model_id, report)


def evaluate_outcomes(records: Iterable[DecisionRecord], predictors: Mapping[str, Callable[[CausalTaskInput], OutcomePrediction | None]],
                      *, split: str, trained: EmpiricalOutcomeBaseline,
                      frozen_test_policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Identical held-out records for each predictor; abstention lowers coverage.

    Executed labels score factual outcome predictions. They cannot score regret
    for unexecuted candidate goals or justify counterfactual task ranking.
    """
    if split not in ("validation", "calibration", "test") or not predictors or len(predictors) > 8:
        raise ValueError("bounded held-out evaluation required")
    rows = _bounded(records)
    if any(r.inputs.session_id in trained.train_sessions or r.inputs.day in trained.train_days or
           r.inputs.record_id in trained.train_record_ids for r in rows):
        raise ValueError("training session/day/record leaked into evaluation")
    if any((r.outcome.utility_definition_id, r.outcome.risk_definition_id) !=
           (trained.utility_definition_id, trained.risk_definition_id) for r in rows):
        raise ValueError("evaluation utility/risk definition mismatch")
    eligible = tuple(row for row in rows if _execution_matches(row))
    exclusions = {"candidate_execution_deviation": len(rows) - len(eligible)} if len(eligible) != len(rows) else {}
    if split == "test":
        if not frozen_test_policy or frozen_test_policy.get("model_id") != trained.model_id:
            raise ValueError("final test requires a model-bound frozen calibration policy")
        policy = dict(frozen_test_policy)
        policy_hash = policy.pop("policy_hash", None)
        if canonical_hash(policy) != policy_hash or policy.get("calibration_split") != "calibration":
            raise ValueError("invalid frozen calibration policy")
        if sorted(predictors) != policy.get("predictor_names"):
            raise ValueError("final-test predictors differ from frozen calibration")
        if any(r.inputs.session_id in policy["calibration_sessions"] or r.inputs.day in policy["calibration_days"] for r in rows):
            raise ValueError("calibration session/day leaked into final test")
    report = {"split": split, "records": len(rows), "eligible_records": len(eligible), "exclusions": exclusions,
              "model_id": trained.model_id,
              "record_hash": canonical_hash([asdict(r) for r in rows]), "predictors": {},
              "sessions": sorted({r.inputs.session_id for r in rows}), "days": sorted({r.inputs.day for r in rows}),
              "label_provenance": dict(Counter(r.outcome.provenance for r in rows)),
              "ranking_evidence": "unavailable: executed logs contain no unexecuted candidate outcomes",
              "uncertainty_semantics": "empirical label scores; no calibrated physical covariance",
              "qualified_for_live": False}
    for name, predict in predictors.items():
        _id(name)
        groups, missing, latencies = defaultdict(list), Counter(exclusions), []
        predictions = {}
        for row in eligible:
            started = time.perf_counter_ns()
            try:
                value = predict(row.inputs)
                if value is None:
                    missing["abstained"] += 1
                    continue
                if not isinstance(value, OutcomePrediction):
                    raise ValueError("validated OutcomePrediction required")
                value.__post_init__()
                errors = (abs(value.utility - row.outcome.utility), (value.success_probability - row.outcome.success) ** 2,
                          (value.risk_probability - row.outcome.risk_event) ** 2)
                for key in ("all", "task:" + row.inputs.task_kind, "session:" + row.inputs.session_id):
                    groups[key].append(errors)
                predictions[row.inputs.record_id] = asdict(value)
            except Exception as exc:
                missing[type(exc).__name__ + ":" + str(exc)] += 1
            finally:
                latencies.append((time.perf_counter_ns() - started) / 1e6)
        scores = {key: {"count": len(v), "utility_mae": sum(x[0] for x in v) / len(v),
                       "success_brier": sum(x[1] for x in v) / len(v), "risk_brier": sum(x[2] for x in v) / len(v)}
                  for key, v in sorted(groups.items())}
        report["predictors"][name] = {"metrics": scores, "missing": dict(missing),
                                      "coverage": len(predictions) / len(rows) if rows else 0., "predictions": predictions,
                                      "latency_ms": {"samples": len(latencies), "maximum": max(latencies, default=0.),
                                                     "mean": sum(latencies) / len(latencies) if latencies else 0.}}
        if split == "test":
            limits = frozen_test_policy["limits"]
            overall = scores.get("all")
            report["predictors"][name]["frozen_limits_passed"] = bool(overall) and all(
                overall[key] <= limits[key] for key in ("utility_mae", "success_brier", "risk_brier")) and \
                report["predictors"][name]["coverage"] >= limits["minimum_coverage"]
    report["evaluation_hash"] = canonical_hash(report)
    return report


def freeze_outcome_test_policy(calibration_report: Mapping[str, Any], *, model_id: str,
                               limits: Mapping[str, float]) -> dict[str, Any]:
    """Freeze descriptive offline score limits before final-test inspection."""
    required = {"utility_mae", "success_brier", "risk_brier", "minimum_coverage"}
    if calibration_report.get("split") != "calibration" or calibration_report.get("model_id") != model_id:
        raise ValueError("matching held-out calibration required")
    report = dict(calibration_report)
    digest = report.pop("evaluation_hash", None)
    if canonical_hash(report) != digest or not report.get("records") or not report.get("sessions"):
        raise ValueError("complete hashed calibration evidence required")
    if not report.get("predictors") or any(not value["metrics"].get("all") for value in report["predictors"].values()):
        raise ValueError("calibration predictions required")
    if set(limits) != required:
        raise ValueError("all explicit score and coverage limits required")
    parsed = {key: _number(value, key, 0, 1 if key != "utility_mae" else 1e6) for key, value in limits.items()}
    policy = {"model_id": model_id, "calibration_split": "calibration", "calibration_hash": digest,
              "calibration_sessions": report["sessions"], "calibration_days": report["days"], "limits": parsed,
              "predictor_names": sorted(report["predictors"]),
              "qualified_for_live": False, "claims": "offline score limits only; no physical safety qualification"}
    policy["policy_hash"] = canonical_hash(policy)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default="decision-outcome-v1")
    args = parser.parse_args()
    records, ingestion = load_decision_jsonl(args.log)
    splits = split_decision_sessions(records, SplitConfig(seed=args.seed))
    model = fit_outcome_baseline(splits.records["train"], split="train")
    result = {"ingestion": ingestion, "split": splits.manifest(), "model": asdict(model),
              "evaluation": {name: evaluate_outcomes(splits.records[name], {"empirical_table": model.predict},
                             split=name, trained=model) for name in ("validation", "calibration")},
              "final_test": "uninspected", "qualified_for_live": False}
    with args.output.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "model_id": model.model_id, "qualified_for_live": False}))


if __name__ == "__main__":
    main()
