"""Content-bound identities for descriptive, frozen offline outcome evaluation.

This wrapper freezes explicit JSON configuration and artifact bytes. It binds a
plain module function's source file and runtime code, rather than trusting a
predictor display name. Callbacks receive ``(inputs, config, artifacts)`` where
the latter two arguments are deeply immutable snapshots. No checkpoint is
deserialized here.

This is reproducibility plumbing, not a Python sandbox or an attestation of a
callback's behavior. Undeclared dependencies, mutable globals, imported runtime
state, external I/O, native libraries, hardware and nondeterminism cannot be
proven by source/artifact hashes. Declare dependency sources and all relevant
configuration/data/checkpoints; review callbacks before use. Bare callbacks,
closures and arbitrary bound methods cannot qualify a frozen final test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import json
import math
import marshal
from pathlib import Path
import platform
import sys
from types import CodeType, FunctionType, MappingProxyType
from typing import Any, Callable, Mapping

IDENTITY_SCHEMA = "frc-offline-predictor-identity/1"
MAX_ARTIFACTS = 32
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_CONFIG_DEPTH = 16
MAX_CONFIG_NODES = 65_536
MAX_IDENTITY_BYTES = 64 * 1024


def _id(value: Any) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError("bounded artifact identity required")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_config(raw: Any) -> tuple[Mapping[str, Any], bytes]:
    """Copy only bounded JSON values; tuples support immutable model snapshots."""
    count = 0
    encoded_budget = 0

    def charge(size: int) -> None:
        nonlocal encoded_budget
        encoded_budget += size
        if encoded_budget > MAX_CONFIG_BYTES:
            raise ValueError("bounded predictor configuration required")

    def copy(value: Any, depth: int) -> Any:
        nonlocal count
        count += 1
        if depth > MAX_CONFIG_DEPTH or count > MAX_CONFIG_NODES:
            raise ValueError("bounded predictor configuration required")
        if value is None or type(value) is bool:
            charge(5)
            return value
        if type(value) in (str, int, float):
            if (type(value) is float and not math.isfinite(value)) or \
                    (type(value) is int and value.bit_length() > 4096) or \
                    (type(value) is str and len(value) > MAX_CONFIG_BYTES):
                raise ValueError("finite bounded predictor configuration required")
            charge(len(_canonical(value)) + 1)
            return value
        if isinstance(value, Mapping):
            if len(value) > MAX_CONFIG_NODES or any(type(key) is not str for key in value):
                raise ValueError("JSON predictor configuration keys required")
            for key in value:
                if len(key) > MAX_CONFIG_BYTES:
                    raise ValueError("bounded predictor configuration required")
                charge(len(_canonical(key)) + 2)
            charge(2)
            return {key: copy(item, depth + 1) for key, item in value.items()}
        if type(value) in (tuple, list):
            if len(value) > MAX_CONFIG_NODES:
                raise ValueError("bounded predictor configuration required")
            charge(len(value) + 2)
            return [copy(item, depth + 1) for item in value]
        raise ValueError("JSON predictor configuration values required")

    if not isinstance(raw, Mapping):
        raise ValueError("explicit JSON predictor configuration required")
    snapshot = copy(raw, 0)
    encoded = _canonical(snapshot)
    if len(encoded) > MAX_CONFIG_BYTES:
        raise ValueError("bounded predictor configuration required")

    def freeze(value: Any) -> Any:
        if type(value) is dict:
            return MappingProxyType({key: freeze(item) for key, item in value.items()})
        if type(value) is list:
            return tuple(freeze(item) for item in value)
        return value

    return freeze(snapshot), encoded


def _normalized_code(code: CodeType) -> CodeType:
    # Exclude checkout path and source line positions, while preserving behavior
    # (including nested code/constants and exception tables) for this Python ABI.
    return code.replace(co_filename="<predictor>", co_firstlineno=0, co_linetable=b"",
                        co_consts=tuple(_normalized_code(item) if isinstance(item, CodeType) else item
                                        for item in code.co_consts))


def _runtime_identity(callback: Callable) -> dict[str, Any]:
    if type(callback) is not FunctionType or callback.__qualname__ != callback.__name__ or \
            callback.__closure__ is not None or callback.__defaults__ is not None or callback.__kwdefaults__:
        raise ValueError("module function without closure/default state required; use an explicit snapshot helper")
    parameters = tuple(inspect.signature(callback).parameters.values())
    if len(parameters) != 3 or any(parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD) for parameter in parameters):
        raise ValueError("predictor function must accept inputs, config, artifacts")
    module = sys.modules.get(callback.__module__)
    if module is None or getattr(module, callback.__name__, None) is not callback:
        raise ValueError("registered module-level predictor function required")
    return {"entrypoint": callback.__module__ + ":" + callback.__qualname__,
            "runtime_code_sha256": _digest(marshal.dumps(_normalized_code(callback.__code__))),
            "python_runtime": platform.python_implementation() + "/" + platform.python_version()}


@dataclass(frozen=True)
class _Artifact:
    role: str
    artifact_id: str
    path: Path
    content: bytes = field(repr=False)

    def manifest(self) -> dict[str, Any]:
        return {"role": self.role, "artifact_id": self.artifact_id,
                "sha256": _digest(self.content), "bytes": len(self.content)}


def _identity_manifest(callback: Callable, config_json: bytes,
                       artifacts: tuple[_Artifact, ...]) -> dict[str, Any]:
    if type(artifacts) is not tuple or not 1 <= len(artifacts) <= MAX_ARTIFACTS:
        raise ValueError("bounded predictor artifact set required")
    source = inspect.getsourcefile(callback)
    if not source:
        raise ValueError("readable predictor implementation source required")
    keys, total_bytes, entrypoint_present = set(), 0, False
    for artifact in artifacts:
        if type(artifact) is not _Artifact or artifact.role not in ("implementation", "checkpoint", "data") or \
                type(artifact.content) is not bytes or not isinstance(artifact.path, Path):
            raise ValueError("validated predictor artifact snapshot required")
        _id(artifact.artifact_id)
        key = (artifact.role, artifact.artifact_id)
        if key in keys or len(artifact.content) > MAX_ARTIFACT_BYTES:
            raise ValueError("bounded unique predictor artifacts required")
        keys.add(key)
        total_bytes += len(artifact.content)
        if total_bytes > MAX_TOTAL_ARTIFACT_BYTES:
            raise ValueError("bounded total predictor artifact bytes required")
        if key == ("implementation", "entrypoint_source"):
            if artifact.path.resolve() != Path(source).resolve():
                raise ValueError("predictor entrypoint source identity changed")
            entrypoint_present = True
    if not entrypoint_present:
        raise ValueError("predictor entrypoint implementation artifact required")
    identity = {"schema_version": IDENTITY_SCHEMA, **_runtime_identity(callback),
                "config_sha256": _digest(config_json),
                "artifacts": [artifact.manifest() for artifact in artifacts],
                "qualification": "descriptive offline identity only; callback dependencies/state require review"}
    identity["identity_hash"] = _digest(_canonical(identity))
    return identity


def _read_artifact(path: Path) -> bytes:
    try:
        if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("bounded regular predictor artifact required")
        with path.open("rb") as stream:
            content = stream.read(MAX_ARTIFACT_BYTES + 1)
    except OSError as exc:
        raise ValueError("predictor artifact unavailable") from exc
    if len(content) > MAX_ARTIFACT_BYTES:
        raise ValueError("bounded predictor artifact required")
    return content


@dataclass(frozen=True)
class PredictorSpec:
    """Immutable explicit predictor inputs plus a verifiable durable identity.

    Register with :meth:`function`. ``verify()`` rehashes current files before
    an evaluation, while each invocation verifies runtime code and uses the
    already frozen bytes/config (avoiding repeated large artifact reads).
    ``identity`` is a fresh JSON-safe manifest; modifying it changes no state.
    """
    _callback: Callable = field(repr=False)
    _config: Mapping[str, Any] = field(repr=False)
    _config_json: bytes = field(repr=False)
    _artifacts: tuple[_Artifact, ...] = field(repr=False)
    _identity_json: bytes = field(repr=False)

    def __post_init__(self) -> None:
        # Direct construction/dataclasses.replace must have the same checks as
        # registration; old metadata cannot be retained with changed snapshots.
        if type(self._config_json) is not bytes or len(self._config_json) > MAX_CONFIG_BYTES or \
                type(self._identity_json) is not bytes or len(self._identity_json) > MAX_IDENTITY_BYTES:
            raise ValueError("bounded predictor identity snapshot required")
        frozen_config, config_json = _json_config(self._config)
        if config_json != self._config_json:
            raise ValueError("predictor configuration identity changed")
        identity = _identity_manifest(self._callback, config_json, self._artifacts)
        if self._identity_json != _canonical(identity):
            raise ValueError("predictor snapshot identity changed")
        object.__setattr__(self, "_config", frozen_config)

    @classmethod
    def function(cls, callback: Callable, *, config: Mapping[str, Any],
                 implementation_files: Mapping[str, str | Path] | None = None,
                 checkpoint_files: Mapping[str, str | Path] | None = None,
                 data_files: Mapping[str, str | Path] | None = None) -> PredictorSpec:
        """Bind a module function, explicit config, and bounded content files.

        Include any additional implementation/dependency files explicitly. The
        callback's own Python source file is always included automatically.
        Artifacts passed to callbacks are grouped by ``checkpoint`` and ``data``
        and contain immutable bytes, with the supplied logical IDs as keys.
        Paths are verification locations, not identity: hashes survive moving
        a checkout. No pickle, torch or executable format is loaded here.
        """
        _runtime_identity(callback)
        source = inspect.getsourcefile(callback)
        if not source:
            raise ValueError("readable predictor implementation source required")
        immutable_config, config_json = _json_config(config)
        artifacts = []
        if implementation_files is not None and not isinstance(implementation_files, Mapping):
            raise ValueError("explicit predictor implementation mapping required")
        sources = dict(implementation_files or {})
        if "entrypoint_source" in sources:
            raise ValueError("entrypoint_source is a reserved artifact identity")
        sources["entrypoint_source"] = source
        groups = (("implementation", sources), ("checkpoint", checkpoint_files or {}),
                  ("data", data_files or {}))
        if any(not isinstance(paths, Mapping) for _, paths in groups) or \
                sum(len(paths) for _, paths in groups) > MAX_ARTIFACTS:
            raise ValueError("bounded predictor artifact set required")
        total_bytes = 0
        for role, paths in groups:
            for artifact_id in paths:
                _id(artifact_id)
            for artifact_id, path in sorted(paths.items()):
                if not isinstance(path, (str, Path)):
                    raise ValueError("explicit predictor artifact path required")
                resolved = Path(path).resolve()
                content = _read_artifact(resolved)
                total_bytes += len(content)
                if total_bytes > MAX_TOTAL_ARTIFACT_BYTES:
                    raise ValueError("bounded total predictor artifact bytes required")
                artifacts.append(_Artifact(role, artifact_id, resolved, content))
        identity = _identity_manifest(callback, config_json, tuple(artifacts))
        return cls(callback, immutable_config, config_json, tuple(artifacts), _canonical(identity))

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self._identity_json)

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config

    def _verify_runtime(self) -> None:
        identity = self.identity
        runtime = _runtime_identity(self._callback)
        if any(runtime[key] != identity[key] for key in runtime):
            raise ValueError("predictor implementation identity changed")
        # The snapshot itself is immutable, but this also detects unsupported
        # manual corruption of the wrapper's frozen fields before evaluation.
        _, config_json = _json_config(self._config)
        if config_json != self._config_json or _digest(config_json) != identity["config_sha256"]:
            raise ValueError("predictor configuration identity changed")

    def verify(self) -> dict[str, Any]:
        """Fail closed on changed/missing source, dependency or artifact files."""
        self._verify_runtime()
        identity = _identity_manifest(self._callback, self._config_json, self._artifacts)
        if type(self._identity_json) is not bytes or len(self._identity_json) > MAX_IDENTITY_BYTES or \
                self._identity_json != _canonical(identity):
            raise ValueError("predictor snapshot identity changed")
        for artifact in self._artifacts:
            if _digest(_read_artifact(artifact.path)) != _digest(artifact.content):
                raise ValueError("predictor " + artifact.role + " artifact identity changed: " + artifact.artifact_id)
        return self.identity

    def __call__(self, inputs: Any) -> Any:
        self._verify_runtime()
        artifacts = MappingProxyType({role: MappingProxyType({artifact.artifact_id: artifact.content
                            for artifact in self._artifacts if artifact.role == role})
                            for role in ("checkpoint", "data")})
        return self._callback(inputs, self._config, artifacts)


def _empirical_predict(inputs: Any, config: Mapping[str, Any], artifacts: Mapping[str, Any]) -> Any:
    from .decision_learning import OutcomePrediction, _cell, _validate_causal_input

    _validate_causal_input(inputs)
    if inputs.possession == "UNKNOWN" or inputs.mechanism_ready is None:
        return None
    prediction = dict(config["cells"]).get(_cell(inputs))
    return None if prediction is None else OutcomePrediction(**dict(prediction))


def empirical_predictor_spec(model: Any) -> PredictorSpec:
    """Snapshot the built-in empirical table; never retain mutable model state.

    Cells, training identity, label definitions and support configuration are
    bound. Runtime measurement fields in ``model.report`` are excluded because
    they do not affect prediction. A changed model needs a new calibration.
    """
    from .decision_learning import EmpiricalOutcomeBaseline

    if type(model) is not EmpiricalOutcomeBaseline:
        raise ValueError("built-in empirical outcome baseline required")
    config = {"kind": "empirical_outcome_baseline", "model_id": model.model_id,
              "training_hash": model.training_hash, "train_sessions": model.train_sessions,
              "train_days": model.train_days, "train_record_ids": model.train_record_ids,
              "utility_definition_id": model.utility_definition_id, "risk_definition_id": model.risk_definition_id,
              "min_examples": model.report["min_examples"],
              "cells": [(key, asdict(prediction)) for key, prediction in model.cells]}
    source = inspect.getsourcefile(EmpiricalOutcomeBaseline)
    if not source:
        raise ValueError("empirical implementation source unavailable")
    return PredictorSpec.function(_empirical_predict, config=config,
                                  implementation_files={"decision_learning_source": source})
