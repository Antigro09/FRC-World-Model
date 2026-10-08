from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from frc_world_model.data import (
    LOCAL_SCHEMA_VERSION, FeatureLayout, ReversibleNormalizer, SplitConfig, WindowConfig,
    build_windows, canonical_hash, load_jsonl, prepare_offline, split_sessions, wrap_angle,
)
from frc_world_model.evaluation import (
    AnalyticalConfig, EgoLayout, MechanismResponse, PredictorInput, ablate_inputs,
    analytical_response_forecast, calibrate_session_bounds, constant_velocity_forecast,
    evaluate_windows, persistence_forecast,
)

STATE = FeatureLayout(("x", "y", "heading", "vx_robot", "vy_robot", "omega", "lift"),
                      ("m", "m", "rad", "m/s", "m/s", "rad/s", "m"), (2,))
ACTION = FeatureLayout(("vx_robot", "vy_robot", "omega", "lift_setpoint"), ("m/s", "m/s", "rad/s", "m"))
EGO = EgoLayout(0, 1, 2, 3, 4, 5)
DT = 100_000_000


def local_record(kind, tick, *, session="session-a", day="2026-10-01", epoch="0", values=None, **overrides):
    layout = STATE if kind == "state" else ACTION
    record = {
        "schema_version": LOCAL_SCHEMA_VERSION, "record_id": f"{session}-{day}-{epoch}-{kind}-{tick}",
        "kind": kind, "session_id": session, "day": day,
        "identity": {"robot_id": "synthetic-robot", "map_id": "synthetic-map", "frame_id": "field-test",
                     "calibration_id": "synthetic-cal", "source_id": "synthetic-source", "source_epoch": "0", "reset_epoch": epoch,
                     "geometry_revision":"synthetic-0","localization_revision":"0","configuration_revision":"0",
                     "sync_mapping_revision":"0","clock_domain":"synthetic-monotonic-ns"},
        "time_sync_valid": True,"time_sync_error_ns":0,"time_sync_limit_ns":1000,
        "layout_hash": canonical_hash(asdict(layout)),
        "values": values or ([tick * 0.1, 0., 0., 1., 0., 0., 0.] if kind == "state" else [1., 0., 0., 0.]),
        "valid": [True] * len(layout.names),
    }
    if kind == "state":
        record.update(state_time_ns=tick * DT, capture_time_ns=tick * DT, input_causal=True, label_provenance="fused_estimate")
    else:
        record.update(accepted_time_ns=tick * DT, accepted=True, interval_end_ns=(tick + 1) * DT, execution={"saturated": False})
    record.update(overrides)
    return record


def session_records(session="session-a", day="2026-10-01", epoch="0"):
    return [local_record(kind, tick, session=session, day=day, epoch=epoch)
            for tick in range(8) for kind in ("state", "action")]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "original.jsonl"

    def load(self, rows):
        self.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return load_jsonl(self.path, STATE, ACTION)

    def test_original_hash_and_determinism(self):
        rows = session_records()
        report = self.load(rows)
        original = self.path.read_bytes()
        repeat = load_jsonl(self.path, STATE, ACTION)
        self.assertEqual(report.dataset_hash, repeat.dataset_hash)
        self.assertEqual(report.original_sha256, repeat.original_sha256)
        self.assertEqual(original, self.path.read_bytes())
        self.assertEqual(report.manifest()["production_binding"], "pending")
        self.assertEqual(report.manifest()["label_provenance"], {"fused_estimate": 8})

    def test_arrival_order_recovered_by_capture_state_time(self):
        rows = list(reversed(session_records()))
        report = self.load(rows)
        self.assertGreater(report.detections["out_of_order"], 0)
        windows = build_windows(report.records, WindowConfig(DT, 2, 2))
        self.assertEqual([w.cutoff_ns for w in windows.windows], [DT * t for t in range(1, 6)])
        for window in windows.windows:
            self.assertTrue(all(r.timestamp_ns <= window.cutoff_ns for r in window.history))
            self.assertTrue(all(r.timestamp_ns > window.cutoff_ns for r in window.targets))

    def test_duplicate_and_conflicting_record_id(self):
        rows = session_records()
        rows.append(dict(rows[0]))
        rows.append({**rows[2], "values": [99., 0., 0., 1., 0., 0., 0.]})
        report = self.load(rows)
        self.assertEqual(report.exclusions["exact_duplicate"], 1)
        self.assertEqual(report.exclusions["conflicting_record_id_original"], 1)
        self.assertNotIn(rows[2]["record_id"], [r.record_id for r in report.records])

    def test_conflicting_same_timestamp_fails_closed(self):
        rows = session_records()
        rows.append({**rows[0], "record_id": "different", "values": [3., 0., 0., 1., 0., 0., 0.]})
        report = self.load(rows)
        self.assertEqual(report.exclusions["conflicting_timestamp"], 2)
        self.assertFalse(any(r.kind == "state" and r.timestamp_ns == 0 for r in report.records))

    def test_future_smoothing_sync_frame_and_schema_rejected(self):
        rows = [local_record("state", 0, input_causal=False),
                local_record("state", 1, label_provenance="future_smoothed"),
                local_record("state", 2, time_sync_valid=False),
                local_record("state", 3, identity={}),
                local_record("state", 4, schema_version="unknown"),
                local_record("action", 0, accepted=False)]
        report = self.load(rows)
        self.assertFalse(report.records)
        self.assertEqual(report.exclusions["future_smoothed_input"], 2)
        self.assertEqual(report.exclusions["invalid_time_sync"], 1)
        self.assertEqual(report.exclusions["missing identity prerequisite"], 1)
        self.assertEqual(report.exclusions["schema_version"], 1)
        self.assertEqual(report.exclusions["action_not_accepted"], 1)

    def test_unknown_values_preserved_and_strict_window_excludes(self):
        rows = session_records()
        rows[6]["values"][0] = 9000.
        rows[6]["valid"][0] = False
        report = self.load(rows)
        unknown = next(r for r in report.records if r.record_id == rows[6]["record_id"])
        self.assertIsNone(unknown.values[0])
        windows = build_windows(report.records, WindowConfig(DT, 2, 2))
        self.assertGreater(windows.exclusions["unknown_state_or_target"], 0)
        self.assertTrue(all(all(r.valid) for w in windows.windows for r in w.history + w.targets))
        normalizer = ReversibleNormalizer.fit([unknown.values, [1., 0., 0., 1., 0., 0., 0.]], STATE, split="train")
        self.assertIsNone(normalizer.transform(unknown.values)[0])
        self.assertEqual(normalizer.centers[0], 1.)

    def test_nan_does_not_become_ground_truth(self):
        row = local_record("state", 0, values=[float("nan"), 0., 0., 1., 0., 0., 0.])
        report = self.load([row])
        self.assertEqual(report.exclusions["nonfinite_valid_value"], 1)
        self.assertFalse(report.records)

    def test_epochs_and_calibration_changes_never_share_windows(self):
        rows = session_records(epoch="0") + session_records(epoch="1")
        report = self.load(rows)
        self.assertEqual(report.detections["epoch_or_identity_changes"], 1)
        windows = build_windows(report.records, WindowConfig(DT, 2, 2))
        self.assertEqual(windows.detections["episodes"], 2)
        for w in windows.windows:
            self.assertEqual(len({r.episode_key for r in w.history + w.actions + w.targets}), 1)

    def test_gaps_and_midstep_actions_are_not_interpolated(self):
        rows = session_records()
        rows = [r for r in rows if not (r["kind"] == "state" and r["state_time_ns"] == 3 * DT)]
        report = self.load(rows)
        result = build_windows(report.records, WindowConfig(DT, 2, 2))
        self.assertGreater(result.exclusions["state_gap_or_alignment"], 0)
        self.assertEqual(result.detections["state_gaps"], 1)
        rows = session_records()
        action = rows[3]
        action["accepted_time_ns"] += DT // 2
        report = self.load(rows)
        result = build_windows(report.records, WindowConfig(DT, 2, 2))
        self.assertGreater(result.exclusions["action_gap_or_midstep_transition"], 0)

    def test_overlapping_actions_excluded_and_clipping_preserved(self):
        rows = session_records()
        rows[1]["interval_end_ns"] = 2 * DT
        rows[5]["execution"] = {"requested": [2., 0., 0., 0.], "saturated": True, "actual": [1., 0., 0., 0.]}
        report = self.load(rows)
        self.assertEqual(report.exclusions["overlapping_action_intervals"], 2)
        action = next(r for r in report.records if r.record_id == rows[5]["record_id"])
        self.assertTrue(dict(action.execution)["saturated"])

    def test_session_and_day_splits_precede_windows(self):
        rows = []
        for i in range(8):
            rows += session_records(session=f"session-{i}", day=f"2026-10-{i // 2 + 1:02}")
        report = self.load(rows)
        config = SplitConfig((.25, .25, .25, .25), "fixture", "day")
        split, windows, sn, an = prepare_offline(report, STATE, ACTION, WindowConfig(DT, 2, 2), config)
        self.assertEqual({len(g) for g in split.groups.values()}, {1})
        sessions = {name: {r.session_id for r in records} for name, records in split.records.items()}
        for a in sessions:
            for b in sessions:
                if a != b:
                    self.assertFalse(sessions[a] & sessions[b])
        self.assertEqual(split.manifest(), split_sessions(report, config).manifest())
        self.assertTrue(all(len(r.windows) == 10 for r in windows.values()))
        self.assertEqual(sn.fit_count, 16)
        self.assertEqual(an.fit_count, 16)

    def test_midnight_session_connects_day_groups(self):
        report = self.load(session_records("same", "2026-10-01") + session_records("same", "2026-10-02") + session_records("other", "2026-10-02"))
        split = split_sessions(report)
        populated = [name for name, records in split.records.items() if records]
        self.assertEqual(len(populated), 1)
        self.assertEqual(len(split.records[populated[0]]), 48)

    def test_train_only_circular_normalization_round_trip(self):
        rows = [[0., 0., math.pi - .01, 1., 0., 0., 0.], [2., 0., -math.pi + .01, 1., 0., 0., 0.]]
        with self.assertRaisesRegex(ValueError, "train only"):
            ReversibleNormalizer.fit(rows, STATE, split="test")
        normalizer = ReversibleNormalizer.fit(rows, STATE, split="train")
        self.assertLess(normalizer.scales[2], .011)
        for row in rows:
            result = normalizer.inverse(normalizer.transform(row))
            for i, (actual, expected) in enumerate(zip(result, row)):
                self.assertAlmostEqual(wrap_angle(actual - expected) if i == 2 else actual - expected, 0.)
        self.assertEqual(normalizer.centers[0], 1.)
        self.assertEqual(normalizer.metadata()["fit_split"], "train")

    def test_state_age_is_causal_and_bounded(self):
        rows = session_records()
        for row in rows:
            if row["kind"] == "state" and row["state_time_ns"] > 0:
                row["state_time_ns"] -= 10
                row["capture_time_ns"] -= 10
                row["capture_time_ns"] -= 10
        report = self.load(rows)
        self.assertFalse(build_windows(report.records, WindowConfig(DT, 2, 2)).windows)
        windows = build_windows(report.records, WindowConfig(DT, 2, 2, 10)).windows
        self.assertTrue(windows)
        for w in windows:
            self.assertLessEqual(w.history[-1].timestamp_ns, w.cutoff_ns)

    def test_constant_velocity_exact_body_twist_and_wrapped_heading(self):
        inputs = PredictorInput(0, 1_000_000_000, ((0., 0., 0., 1., 0., math.pi / 2, 0.),), (), ((0., 0., 0., 0.),), (), "test")
        state = constant_velocity_forecast(inputs, EGO)[0]
        self.assertAlmostEqual(state[0], 2 / math.pi)
        self.assertAlmostEqual(state[1], 2 / math.pi)
        self.assertAlmostEqual(state[2], math.pi / 2)
        self.assertEqual(persistence_forecast(inputs)[0], inputs.history[0])

    def test_analytical_clipping_and_mechanism_response(self):
        config = AnalyticalConfig(EGO, (0, 1, 2), (1., 1., 1.), (.5, .5, 1.), (MechanismResponse(6, 3, 1., 0., 1.),))
        config.validate(STATE, ACTION)
        inputs = PredictorInput(0, 1_000_000_000, ((0., 0., 0., 0., 0., 0., 0.),), (), ((2., 0., 0., 3.),), (), "test")
        state = analytical_response_forecast(inputs, config)[0]
        self.assertAlmostEqual(state[3], .5 * (1 - math.exp(-1)))
        self.assertAlmostEqual(state[0], .5 * math.exp(-1))
        self.assertAlmostEqual(state[6], 1 - math.exp(-1))

    def test_evaluation_callback_has_no_future_labels_and_same_horizons(self):
        report = self.load(session_records())
        windows = build_windows(report.records, WindowConfig(DT, 2, 2)).windows
        seen = []
        def predictor(inputs):
            self.assertFalse(hasattr(inputs, "targets"))
            seen.append(inputs)
            return constant_velocity_forecast(inputs, EGO)
        result = evaluate_windows(windows, predictor, STATE, ego=EGO, name="constant_velocity")
        self.assertEqual(result["successful_windows"], 5)
        self.assertEqual(set(result["per_horizon_ns"]), {str(DT), str(2 * DT)})
        self.assertLess(result["per_horizon_ns"][str(2 * DT)]["position_error_m"]["max"], 1e-12)
        self.assertEqual(result["label_provenance"], {"fused_estimate": 10})
        one = evaluate_windows(windows, predictor, STATE, ego=EGO, name="constant_velocity", mode="oneStep" if False else "one_step")
        self.assertEqual(set(one["per_horizon_ns"]), {str(DT)})
        with self.assertRaisesRegex(ValueError, "calibration-only"):
            calibrate_session_bounds(result, split="test")
        bounds = calibrate_session_bounds(result, split="calibration")
        self.assertTrue(bounds["advisory_only"])
        self.assertEqual(bounds["coverage_claim"].split(";")[0], "none")

    def test_prediction_failures_count_and_ablation_requires_neutral_action(self):
        report = self.load(session_records())
        windows = build_windows(report.records, WindowConfig(DT, 2, 2)).windows
        result = evaluate_windows(windows, lambda inputs: [[float("nan")]], STATE, name="broken")
        self.assertEqual(result["successful_windows"], 0)
        self.assertEqual(sum(result["missing_predictions"].values()), 5)
        with self.assertRaisesRegex(ValueError, "neutral action"):
            ablate_inputs(persistence_forecast, actions=True)
        inputs = PredictorInput.from_window(windows[0])
        seen = []
        predictor = ablate_inputs(lambda i: seen.append(i) or persistence_forecast(i), history=True, actions=True, neutral_action=[0., 0., 0., 0.])
        predictor(inputs)
        self.assertEqual(len(seen[0].history), 1)
        self.assertFalse(seen[0].history_actions)
        self.assertEqual(len(seen[0].actions), 2)

    def test_cli_exclusive_manifest_and_final_test_not_inspected(self):
        rows = []
        for i in range(4):
            rows += session_records(f"session-{i}", f"2026-10-{i+1:02}")
        self.load(rows)
        original = self.path.read_bytes()
        config = Path(self.temp.name) / "config.json"
        config.write_text(json.dumps({"state_layout": asdict(STATE), "action_layout": asdict(ACTION),
                                      "window": {"timestep_ns": DT, "history_steps": 2, "horizon_steps": 2},
                                      "split": {"ratios": [.25, .25, .25, .25]}, "ego": asdict(EGO)}))
        output = Path(self.temp.name) / "manifest.json"
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "tools" / "data_cli.py"), str(self.path), "--config", str(config), "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        artifact = json.loads(output.read_text())
        self.assertNotIn("test", artifact["evaluation"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()


def test_evaluation_checksum_binds_actual_error_values(tmp_path):
    path=tmp_path/"log.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in session_records()))
    records=load_jsonl(path,STATE,ACTION)
    windows=build_windows(records.records,WindowConfig(DT,2,2)).windows
    def biased(amount):
        def predict(inputs):
            return [tuple(v+(amount if i==0 else 0) for i,v in enumerate(row)) for row in persistence_forecast(inputs)]
        return predict
    first=evaluate_windows(windows,biased(1.),STATE,ego=EGO,name="rwm")
    second=evaluate_windows(windows,biased(100.),STATE,ego=EGO,name="rwm")
    assert first["evaluated_window_ids"]==second["evaluated_window_ids"]
    assert first["evaluation_hash"]!=second["evaluation_hash"]
