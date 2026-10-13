from copy import deepcopy
from dataclasses import asdict
import json
from itertools import repeat
import subprocess
import sys

import pytest

from frc_world_model.data import IDENTITY_KEYS, SplitConfig
from frc_world_model.decision_learning import (
    OUTCOME_SCHEMA, DecisionRecord, OutcomePrediction, evaluate_outcomes,
    fit_outcome_baseline, freeze_outcome_test_policy, load_decision_jsonl,
    split_decision_sessions,
)


def outcome_fixture(index=0, *, session=None, day=None, speed=1., utility=4., success=True,
                    task="PICKUP", possession="EMPTY", ready=True):
    identity = {key: "fixture-" + key for key in IDENTITY_KEYS}
    identity.update(frame_id="BLUE_FIELD", clock_domain="ROBOT_MONOTONIC_US")
    cutoff, step, horizon = 1_000_000 + index * 1_000_000, 100_000, 200_000
    commands = [{"offset_us": offset, "body_twist": [speed, 0., 0.], "mechanisms": {"intake": "ACQUIRE"}}
                for offset in (0, step)]
    return {
        "schema_version": OUTCOME_SCHEMA, "record_id": f"record-{index}",
        "session_id": session or f"session-{index}", "day": day or f"2026-10-{index + 1:02d}",
        "identity": identity, "request_id": f"request-{index}", "snapshot_id": index,
        "candidate": {"candidate_id": f"candidate-{index}", "task_id": f"task-{index}", "task_kind": task,
                      "goal_xy_m": [2., 1.], "target_ids": ["track-1"], "path_length_m": 2.5},
        "cutoff_us": cutoff, "horizon_us": horizon, "step_us": step,
        "context": {"input_causal": True, "observed_us": cutoff, "ego_state": [0., 0., 0., 0., 0., 0.],
                    "possession": possession, "possession_observed_us": cutoff, "mechanism_ready": ready,
                    "mechanism_observed_us": cutoff, "minimum_clearance_m": .5, "time_remaining_s": 100.},
        "commands_created_us": cutoff, "candidate_commands": commands,
        "executed_commands": [dict(sample, accepted_us=cutoff + sample["offset_us"]) for sample in commands],
        "outcome": {"observed_us": cutoff + horizon, "identity": deepcopy(identity), "success": success,
                    "risk_event": not success, "utility": utility, "provenance": "synthetic_fixture",
                    "label_source_id": "toy-labeler-v1", "utility_definition_id": "toy-utility-v1",
                    "risk_definition_id": "toy-risk-v1"},
        "time_sync_valid": True, "time_sync_error_us": 10, "time_sync_limit_us": 100,
    }


def record(index=0, **kwargs):
    return DecisionRecord.from_dict(outcome_fixture(index, **kwargs))


def test_causal_inputs_do_not_expose_future_commands_or_labels():
    row = record()
    assert row.executed_commands[1].accepted_us > row.inputs.cutoff_us
    assert row.inputs.commands[1].accepted_us is None
    assert not hasattr(row.inputs, "outcome") and not hasattr(row.inputs, "executed_commands")
    assert not hasattr(row.inputs, "utility") and not hasattr(row.inputs, "label_source_id")
    called = []
    trained = fit_outcome_baseline([record(2), record(3)], split="train")

    def predictor(inputs):
        called.append(asdict(inputs))
        return OutcomePrediction(4., 1., 0., "deterministic_fixture")

    report = evaluate_outcomes([row], {"deterministic": predictor}, split="validation", trained=trained)
    assert report["predictors"]["deterministic"]["metrics"]["all"]["utility_mae"] == 0
    assert "outcome" not in called[0] and "executed_commands" not in called[0]
    assert report["qualified_for_live"] is False


@pytest.mark.parametrize("change", [
    lambda raw: raw.update(schema_version="frc-prediction/1"),
    lambda raw: raw.update(commands_created_us=raw["cutoff_us"] + 1),
    lambda raw: raw["context"].update(observed_us=raw["cutoff_us"] + 1),
    lambda raw: raw["context"].update(possession_observed_us=raw["cutoff_us"] + 1),
    lambda raw: raw["context"].update(input_causal=False),
    lambda raw: raw["context"].update(future_success=True),
    lambda raw: raw["context"].update(ego_state=[0., 0., 0., float("nan"), 0., 0.]),
    lambda raw: raw["context"].update(mechanism_ready=1),
    lambda raw: raw["candidate"].update(task_kind="DRIVE_MOTORS"),
    lambda raw: raw.update(time_sync_error_us=101),
    lambda raw: raw.update(time_sync_valid=False),
    lambda raw: raw["identity"].update(clock_domain="UNDECLARED"),
    lambda raw: raw["candidate_commands"][1].update(offset_us=100_001),
    lambda raw: raw["candidate_commands"][0].update(accepted_us=10),
    lambda raw: raw["executed_commands"][1].update(accepted_us=raw["cutoff_us"] + 100_001),
    lambda raw: raw["outcome"].update(observed_us=raw["cutoff_us"] + raw["horizon_us"] - 1),
    lambda raw: raw["outcome"].update(success=None),
    lambda raw: raw["outcome"].update(risk_event=0),
    lambda raw: raw["outcome"]["identity"].update(reset_epoch="reboot"),
    lambda raw: raw.update(horizon_us=30_100_000),
    lambda raw: raw.update(step_us=True),
])
def test_malformed_leaking_missing_and_epoch_crossing_records_rejected(change):
    raw = outcome_fixture()
    change(raw)
    with pytest.raises((ValueError, TypeError)):
        DecisionRecord.from_dict(raw)


def test_jsonl_is_read_only_strict_deduplicated_and_bounded(tmp_path):
    path = tmp_path / "input.jsonl"
    original = (json.dumps(outcome_fixture()) + "\n") * 2
    path.write_text(original)
    rows, report = load_decision_jsonl(path)
    assert len(rows) == 1 and report["duplicates"] == 1
    assert report["label_provenance"] == {"synthetic_fixture": 1}
    assert path.read_text() == original
    with pytest.raises(ValueError, match="capacity"):
        load_decision_jsonl(path, max_records=1)
    with pytest.raises(ValueError, match="bounded input"):
        load_decision_jsonl(path, max_bytes=10)
    conflicting = outcome_fixture()
    conflicting["outcome"]["success"] = False
    path.write_text(json.dumps(outcome_fixture()) + "\n" + json.dumps(conflicting) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_decision_jsonl(path)
    path.write_text(json.dumps(outcome_fixture()) + "\n{}\n")
    with pytest.raises(ValueError, match="line 2"):
        load_decision_jsonl(path)


@pytest.mark.parametrize("field", ["commands_created_us", "success", "offset_us"])
def test_jsonl_duplicate_keys_are_rejected_at_every_nesting_level(tmp_path, field):
    raw = json.dumps(outcome_fixture())
    marker = '"' + field + '": '
    duplicate = raw.replace(marker, marker + '999999999, ' + marker, 1)
    path = tmp_path / "duplicate-key.jsonl"
    path.write_text(duplicate + "\n")
    with pytest.raises(ValueError, match="duplicate JSON key: " + field):
        load_decision_jsonl(path)


def test_python_api_bounds_source_iteration_even_for_an_infinite_duplicate_stream():
    with pytest.raises(ValueError, match="capacity"):
        fit_outcome_baseline(repeat(record()), split="train")


def test_day_and_midnight_session_groups_never_leak_and_shuffle_is_deterministic():
    rows = [record(index) for index in range(12)]
    rows[1] = record(1, session="session-0", day="2026-10-02")
    config = SplitConfig((.25, .25, .25, .25), "fixture-split", "day")
    a, b = split_decision_sessions(rows, config), split_decision_sessions(reversed(rows), config)
    assert a.manifest() == b.manifest()
    membership = {}
    for split, records in a.records.items():
        for row in records:
            for key in ("session:" + row.inputs.session_id, "day:" + row.inputs.day):
                assert key not in membership or membership[key] == split
                membership[key] = split
    assert set(a.records) == {"train", "validation", "calibration", "test"}


def test_baseline_is_action_and_task_conditioned_and_abstains_without_support():
    rows = [record(0, speed=0., utility=1., success=False), record(1, speed=0., utility=1., success=False),
            record(2, speed=2., utility=8.), record(3, speed=2., utility=8.)]
    trained = fit_outcome_baseline(rows, split="train")
    assert trained.predict(record(4, speed=0.).inputs).utility == 1
    assert trained.predict(record(4, speed=2.).inputs).utility == 8
    assert trained.predict(record(4, speed=3.).inputs) is None
    assert trained.predict(record(4, possession="UNKNOWN").inputs) is None
    assert trained.predict(record(4, ready=None).inputs) is None
    assert trained.predict(record(4, task="SCORE").inputs) is None
    different_map = outcome_fixture(4, speed=2.)
    different_map["identity"]["map_id"] = "new-map"
    different_map["outcome"]["identity"] = deepcopy(different_map["identity"])
    assert trained.predict(DecisionRecord.from_dict(different_map).inputs) is None
    assert trained.report["qualified_for_live"] is False and trained.report["optimizer_updates"] == 0
    assert fit_outcome_baseline(rows, split="train").model_id == trained.model_id
    with pytest.raises(ValueError, match="train-only"):
        fit_outcome_baseline(rows, split="validation")


def test_execution_deviation_is_preserved_and_excluded_from_expected_command_labels():
    raw = outcome_fixture()
    raw["executed_commands"][1]["body_twist"] = [0., 0., 0.]
    deviated = DecisionRecord.from_dict(raw)
    trained = fit_outcome_baseline([deviated, record(1)], split="train")
    assert trained.report["exclusions"]["candidate_execution_deviation"] == 1
    assert trained.report["exclusions"]["insufficient_bin_support"] == 1
    assert trained.cells == ()
    raw = outcome_fixture(2)
    raw["executed_commands"][0]["applied_body_twist"] = [0.5, 0., 0.]
    clipped = DecisionRecord.from_dict(raw)
    assert fit_outcome_baseline([clipped], split="train").report["exclusions"]["candidate_execution_deviation"] == 1


@pytest.mark.parametrize("deviation", ["accepted", "applied", "mechanism"])
def test_held_out_execution_deviations_are_excluded_without_scoring_counterfactual_labels(deviation):
    trained = fit_outcome_baseline([record(0), record(1)], split="train")
    raw = outcome_fixture(4)
    if deviation == "accepted":
        raw["executed_commands"][1]["body_twist"] = [0., 0., 0.]
    elif deviation == "applied":
        raw["executed_commands"][1]["applied_body_twist"] = [0.5, 0., 0.]
    else:
        raw["executed_commands"][1]["mechanisms"] = {"intake": "STOPPED"}
    called = []

    def predictor(inputs):
        called.append(inputs.record_id)
        return trained.predict(inputs)

    report = evaluate_outcomes([DecisionRecord.from_dict(raw), record(5)], {"table": predictor},
                               split="validation", trained=trained)
    assert called == ["record-5"]
    assert report["records"] == 2 and report["eligible_records"] == 1
    assert report["exclusions"] == {"candidate_execution_deviation": 1}
    assert report["predictors"]["table"]["metrics"]["all"]["count"] == 1
    assert report["predictors"]["table"]["coverage"] == .5
    assert report["predictors"]["table"]["missing"] == {"candidate_execution_deviation": 1}


def test_evaluation_compares_identical_factual_labels_and_counts_failures():
    trained = fit_outcome_baseline([record(0), record(1)], split="train")
    rows = [record(4), record(5, utility=0., success=False)]

    def unavailable(inputs):
        raise TimeoutError("offline fixture")

    report = evaluate_outcomes(rows, {
        "table": trained.predict, "deterministic": lambda inputs: OutcomePrediction(0., .5, .5, "deterministic_fixture"),
        "abstained": lambda inputs: None, "unavailable": unavailable,
    }, split="validation", trained=trained)
    metrics = report["predictors"]["table"]["metrics"]["all"]
    assert metrics == {"count": 2, "utility_mae": 2., "success_brier": .5, "risk_brier": .5}
    assert report["predictors"]["deterministic"]["metrics"]["all"]["success_brier"] == .25
    assert report["predictors"]["abstained"]["coverage"] == 0
    assert report["predictors"]["unavailable"]["missing"] == {"TimeoutError:offline fixture": 2}
    assert report["predictors"]["table"]["latency_ms"]["samples"] == 2
    assert "unexecuted" in report["ranking_evidence"]
    changed = evaluate_outcomes([record(4, utility=9.)], {"table": trained.predict}, split="validation", trained=trained)
    assert changed["evaluation_hash"] != report["evaluation_hash"]


def test_train_session_day_and_label_definition_leakage_rejected():
    trained = fit_outcome_baseline([record(0), record(1)], split="train")
    for row in (record(0), record(4, session="session-0"), record(4, day="2026-10-01")):
        with pytest.raises(ValueError, match="leaked"):
            evaluate_outcomes([row], {"table": trained.predict}, split="validation", trained=trained)
    raw = outcome_fixture(4)
    raw["outcome"]["utility_definition_id"] = "different-utility"
    with pytest.raises(ValueError, match="definition mismatch"):
        evaluate_outcomes([DecisionRecord.from_dict(raw)], {"table": trained.predict}, split="validation", trained=trained)
    with pytest.raises(ValueError, match="must not be mixed"):
        fit_outcome_baseline([record(0), DecisionRecord.from_dict(raw)], split="train")


def test_final_test_requires_frozen_model_bound_policy_and_reports_failed_limits():
    trained = fit_outcome_baseline([record(0), record(1)], split="train")
    predictors = {"table": trained.predict}
    with pytest.raises(ValueError, match="frozen calibration"):
        evaluate_outcomes([record(6)], predictors, split="test", trained=trained)
    calibration = evaluate_outcomes([record(4), record(5)], predictors, split="calibration", trained=trained)
    limits = {"utility_mae": .1, "success_brier": .1, "risk_brier": .1, "minimum_coverage": 1.}
    policy = freeze_outcome_test_policy(calibration, model_id=trained.model_id, limits=limits)
    passed = evaluate_outcomes([record(6)], predictors, split="test", trained=trained, frozen_test_policy=policy)
    assert passed["predictors"]["table"]["frozen_limits_passed"]
    failed = evaluate_outcomes([record(6, utility=0., success=False)], predictors,
                               split="test", trained=trained, frozen_test_policy=policy)
    assert not failed["predictors"]["table"]["frozen_limits_passed"] and failed["qualified_for_live"] is False
    with pytest.raises(ValueError, match="calibration session/day"):
        evaluate_outcomes([record(5)], predictors, split="test", trained=trained, frozen_test_policy=policy)
    corrupt = dict(policy, limits=dict(limits, utility_mae=100.))
    with pytest.raises(ValueError, match="invalid frozen"):
        evaluate_outcomes([record(6)], predictors, split="test", trained=trained, frozen_test_policy=corrupt)
    with pytest.raises(ValueError, match="predictors differ"):
        evaluate_outcomes([record(6)], {"other": trained.predict}, split="test", trained=trained, frozen_test_policy=policy)
    calibration["records"] = 20
    with pytest.raises(ValueError, match="hashed calibration"):
        freeze_outcome_test_policy(calibration, model_id=trained.model_id, limits=limits)


def test_small_dataset_reports_empty_held_out_splits_without_invented_scores():
    split = split_decision_sessions([record(0)])
    assert split.manifest()["counts"] == {"train": 1, "validation": 0, "calibration": 0, "test": 0}
    trained = fit_outcome_baseline(split.records["train"], split="train")
    report = evaluate_outcomes([], {"table": trained.predict}, split="validation", trained=trained)
    assert report["records"] == 0 and report["predictors"]["table"]["metrics"] == {}


def test_cli_is_bounded_exclusive_and_leaves_final_test_uninspected(tmp_path):
    source, target = tmp_path / "fixtures.jsonl", tmp_path / "report.json"
    source.write_text("".join(json.dumps(outcome_fixture(index)) + "\n" for index in range(12)))
    command = [sys.executable, "-m", "frc_world_model.decision_learning", str(source), "--output", str(target)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(target.read_text())
    assert report["qualified_for_live"] is False and report["final_test"] == "uninspected"
    assert set(report["evaluation"]) == {"validation", "calibration"}
    assert json.loads(completed.stdout)["qualified_for_live"] is False
    original = target.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert target.read_bytes() == original
