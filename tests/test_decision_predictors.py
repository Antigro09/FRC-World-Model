from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from frc_world_model import decision_predictors as identities
from frc_world_model.data import IDENTITY_KEYS
from frc_world_model.decision_learning import (
    CausalTaskInput, CommandSample, EmpiricalOutcomeBaseline, OutcomePrediction, _cell,
)
from frc_world_model.decision_predictors import PredictorSpec, empirical_predictor_spec


def _fixture_predict(inputs, config, artifacts):
    return inputs + config["offset"] + len(artifacts["checkpoint"].get("weights", b""))


def _other_predict(inputs, config, artifacts):
    return inputs - config["offset"]


def _with_default(inputs, config, artifacts=None):
    return inputs


def _load_module(path, *, name="bounded_predictor_fixture"):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture
def module_fixture(tmp_path):
    path = tmp_path / "predictor.py"
    path.write_text("def predict(inputs, config, artifacts):\n    return inputs + config['offset']\n")
    module = _load_module(path)
    yield path, module
    sys.modules.pop(module.__name__, None)


def test_configuration_is_canonical_copied_immutable_and_identity_is_json_safe():
    original = {"offset": 2, "nested": {"samples": [1, 2]}}
    spec = PredictorSpec.function(_fixture_predict, config=original)
    equal = PredictorSpec.function(_fixture_predict, config={"nested": {"samples": [1, 2]}, "offset": 2})
    assert spec.identity == equal.identity
    assert json.loads(json.dumps(spec.identity)) == spec.identity
    original["offset"] = 200
    original["nested"]["samples"].append(3)
    assert spec(1) == 3 and spec.config["nested"]["samples"] == (1, 2)
    with pytest.raises(TypeError):
        spec.config["offset"] = 3
    with pytest.raises(TypeError):
        spec.config["nested"]["samples"][0] = 3
    exposed_identity = spec.identity
    exposed_identity["config_sha256"] = "forged"
    assert spec.verify() == equal.identity
    changed = PredictorSpec.function(_fixture_predict, config=original)
    assert changed.identity != spec.identity


def test_same_display_name_cannot_hide_replaced_runtime_implementation(module_fixture):
    path, module = module_fixture
    first = PredictorSpec.function(module.predict, config={"offset": 2})
    # Replace the function under exactly the same module/entrypoint name, with
    # no source-file modification: a name-only policy would miss this change.
    exec(compile("def predict(inputs, config, artifacts):\n    return inputs - config['offset']\n",
                 str(path), "exec"), module.__dict__)
    replacement = PredictorSpec.function(module.predict, config={"offset": 2})
    assert first.identity["entrypoint"] == replacement.identity["entrypoint"]
    assert first.identity["artifacts"] == replacement.identity["artifacts"]
    assert first.identity["runtime_code_sha256"] != replacement.identity["runtime_code_sha256"]
    with pytest.raises(ValueError, match="registered module-level"):
        first.verify()
    assert {"same-display-name": first.identity} != {"same-display-name": replacement.identity}


def test_runtime_code_mutation_is_rejected_before_invocation(monkeypatch):
    spec = PredictorSpec.function(_fixture_predict, config={"offset": 2})
    monkeypatch.setattr(_fixture_predict, "__code__", _other_predict.__code__)
    with pytest.raises(ValueError, match="implementation identity changed"):
        spec(1)


@pytest.mark.parametrize("role", ["implementation", "checkpoint", "data"])
def test_changed_files_are_rejected_and_artifact_bytes_are_frozen(tmp_path, role):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"before")
    kwargs = {role + "_files": {"weights": artifact}}
    spec = PredictorSpec.function(_fixture_predict, config={"offset": 2}, **kwargs)
    expected = 9 if role == "checkpoint" else 3
    assert spec(1) == expected
    artifact.write_bytes(b"after")
    with pytest.raises(ValueError, match=role + " artifact identity changed"):
        spec.verify()
    # An evaluation verifies files up front; invocation consumes the unchanged
    # bytes it registered, never a newly loaded checkpoint or mutable path.
    assert spec(1) == expected
    rebuilt = PredictorSpec.function(_fixture_predict, config={"offset": 2}, **kwargs)
    assert spec.identity != rebuilt.identity


def test_callback_source_changes_even_when_loaded_runtime_is_unchanged(module_fixture):
    path, module = module_fixture
    spec = PredictorSpec.function(module.predict, config={"offset": 2})
    path.write_text(path.read_text() + "# changed implementation artifact\n")
    with pytest.raises(ValueError, match="implementation artifact identity changed"):
        spec.verify()
    path.unlink()
    with pytest.raises(ValueError, match="predictor artifact"):
        spec.verify()


def test_same_content_identity_survives_relocated_checkout(module_fixture, tmp_path):
    path, module = module_fixture
    first = PredictorSpec.function(module.predict, config={"offset": 2})
    other = tmp_path / "other" / "predictor.py"
    other.parent.mkdir()
    other.write_bytes(path.read_bytes())
    relocated = _load_module(other, name=module.__name__)
    second = PredictorSpec.function(relocated.predict, config={"offset": 2})
    assert first.identity == second.identity
    assert str(tmp_path) not in json.dumps(second.identity)


def test_direct_dataclass_replacement_cannot_retain_identity_with_changed_snapshot(tmp_path):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"before")
    spec = PredictorSpec.function(_fixture_predict, config={"offset": 2}, checkpoint_files={"weights": checkpoint})
    modified = tuple(replace(artifact, content=b"after") if artifact.role == "checkpoint" else artifact
                     for artifact in spec._artifacts)
    checkpoint.write_bytes(b"after")
    with pytest.raises(ValueError, match="snapshot identity changed"):
        replace(spec, _artifacts=modified)
    with pytest.raises(ValueError, match="configuration identity changed"):
        replace(spec, _config={"offset": 20})
    with pytest.raises(ValueError, match="snapshot identity changed"):
        replace(spec, _identity_json=spec._identity_json.replace(b'"identity_hash":"', b'"identity_hash":"corrupt'))
    with pytest.raises(ValueError, match="bounded predictor artifact set"):
        replace(spec, _artifacts=())


def test_verify_recomputes_full_manifest_even_after_unsupported_frozen_field_corruption(tmp_path):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"before")
    spec = PredictorSpec.function(_fixture_predict, config={"offset": 2}, checkpoint_files={"weights": checkpoint})
    modified = tuple(replace(artifact, content=b"after") if artifact.role == "checkpoint" else artifact
                     for artifact in spec._artifacts)
    checkpoint.write_bytes(b"after")
    # Explicit frozen-field corruption is unsupported, but evaluation must
    # still reject the old manifest rather than trusting its hash metadata.
    object.__setattr__(spec, "_artifacts", modified)
    with pytest.raises(ValueError, match="snapshot identity changed"):
        spec.verify()


def test_stateful_callback_shapes_are_rejected_instead_of_guessed():
    captured = 1

    def closure(inputs, config, artifacts):
        return inputs + captured

    class Stateful:
        def predict(self, inputs, config, artifacts):
            return self

    for callback in (closure, Stateful().predict, _with_default, lambda inputs: inputs):
        with pytest.raises(ValueError, match="module function"):
            PredictorSpec.function(callback, config={})


@pytest.mark.parametrize("config", [{"value": float("nan")}, {"value": object()}, {1: "bad-key"},
                                     {"value": 1 << 4097}, {"value": "x" * (identities.MAX_CONFIG_BYTES + 1)}])
def test_non_json_nonfinite_or_oversized_configuration_rejected(config):
    with pytest.raises(ValueError, match="configuration"):
        PredictorSpec.function(_fixture_predict, config=config)


def test_nested_and_aggregate_configuration_bounds_apply_before_unbounded_serialization():
    nested = {}
    for _ in range(identities.MAX_CONFIG_DEPTH + 1):
        nested = {"child": nested}
    for config in (nested, {"values": ["x" * 200_000] * 10}):
        with pytest.raises(ValueError, match="bounded predictor configuration"):
            PredictorSpec.function(_fixture_predict, config=config)


def test_artifact_count_size_total_and_duplicate_identity_are_bounded(tmp_path, monkeypatch):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"weights")
    with pytest.raises(ValueError, match="bounded predictor artifact set"):
        PredictorSpec.function(_fixture_predict, config={}, checkpoint_files={str(i): path for i in range(32)})
    with pytest.raises(ValueError, match="reserved artifact"):
        PredictorSpec.function(_fixture_predict, config={}, implementation_files={"entrypoint_source": path})
    with pytest.raises(ValueError, match="implementation mapping"):
        PredictorSpec.function(_fixture_predict, config={}, implementation_files=[("source", path)])
    source_bytes = Path(__file__).stat().st_size
    with path.open("wb") as output:
        output.truncate(source_bytes + 2)
    monkeypatch.setattr(identities, "MAX_ARTIFACT_BYTES", source_bytes + 1)
    with pytest.raises(ValueError, match="bounded regular predictor artifact"):
        PredictorSpec.function(_fixture_predict, config={}, checkpoint_files={"weights": path})
    path.write_bytes(b"weights")
    monkeypatch.setattr(identities, "MAX_TOTAL_ARTIFACT_BYTES", source_bytes + 3)
    with pytest.raises(ValueError, match="bounded total predictor artifact bytes"):
        PredictorSpec.function(_fixture_predict, config={}, checkpoint_files={"weights": path})


def test_empirical_helper_binds_complete_prediction_state_without_mutable_bound_method():
    identity = {key: "fixture-" + key for key in IDENTITY_KEYS}
    identity.update(frame_id="BLUE_FIELD", clock_domain="ROBOT_MONOTONIC_US")
    inputs = CausalTaskInput(
        record_id="record", session_id="session", day="2026-10-12", identity=tuple(identity.items()),
        request_id="request", snapshot_id=0, candidate_id="candidate", task_id="task", task_kind="PICKUP",
        cutoff_us=1_000_000, horizon_us=200_000, step_us=100_000, observed_us=1_000_000,
        ego_state=(0.,) * 6, possession="EMPTY", possession_observed_us=1_000_000,
        mechanism_ready=True, mechanism_observed_us=1_000_000, goal_xy_m=(2., 1.), target_ids=("track",),
        path_length_m=2., minimum_clearance_m=.5, time_remaining_s=100.,
        commands=tuple(CommandSample(offset, (1., 0., 0.), (("intake", "ACQUIRE"),)) for offset in (0, 100_000)),
    )
    prediction = OutcomePrediction(4., .75, .1, support_examples=2)
    model = EmpiricalOutcomeBaseline(((_cell(inputs), prediction),), ("session",), ("day",), ("record",),
                                     "training-content-hash", "utility-v1", "risk-v1", "model-v1",
                                     {"min_examples": 2, "duration_s": 1.})
    first = empirical_predictor_spec(model)
    assert first(inputs) == model.predict(inputs) == prediction
    assert any(artifact["artifact_id"] == "decision_learning_source" for artifact in first.identity["artifacts"])
    model.report["duration_s"] = 500.
    assert empirical_predictor_spec(model).identity == first.identity
    model.report["min_examples"] = 3
    changed = replace(model, cells=((_cell(inputs), replace(prediction, utility=400.)),))
    second = empirical_predictor_spec(changed)
    assert first(inputs).utility == 4. and second(inputs).utility == 400.
    assert first.identity != second.identity
    with pytest.raises(ValueError, match="module function"):
        PredictorSpec.function(model.predict, config={})
