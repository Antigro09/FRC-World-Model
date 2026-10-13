"""Reproducible, self-contained companion decision-support JSON schemas.

The embedded predictor schema retains the pinned World-State contract. A local
profile adds tighter engineering resource bounds without redefining that owner
contract. Schema generation performs no model loading, networking or training.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "frc-decision-support/1"
CONTRACT = Path(__file__).with_name("_contracts") / "decision-support-v1"
REQUEST_SCHEMA = CONTRACT / "decision-request-v1.schema.json"
REPLY_SCHEMA = CONTRACT / "decision-reply-v1.schema.json"
PREDICTION_SCHEMA = Path(__file__).with_name("_contracts") / "world-state-v1" / "prediction-request-v1.schema.json"
PREDICTION_SCHEMA_SHA256 = "4fe241f505bc6055d9526bfb7dda996a7838cfcbdc12f5c636989cc9c1d5f658"
PREDICTION_REPLY_SCHEMA = PREDICTION_SCHEMA.with_name("prediction-reply-v1.schema.json")
PREDICTION_REPLY_SCHEMA_SHA256 = "5d05c7fef9f08ed1d4196c64455c623b6cd1b489852fa609c65bccab03e6cc14"
MAX_WIRE_INTEGER = 9_007_199_254_740_991
IDENTITY_KEYS = ("schema_version", "model_id", "request_id", "snapshot_id", "epoch",
                 "field", "source", "frame", "timestamp_domain", "history_cutoff_us")


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def _array(items, maximum, minimum=0, *, unique=False):
    result = {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}
    if unique:
        result["uniqueItems"] = True
    return result


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def _ref(name):
    return {"$ref": "#/$defs/" + name}


def _integer():
    return {"type": "integer", "minimum": 0, "maximum": MAX_WIRE_INTEGER}


def _number(*, nonnegative=False):
    # JSON parsing and semantic validation additionally reject every NaN/Inf.
    result = {"type": "number"}
    if nonnegative:
        result["minimum"] = 0
    return result


def _identity():
    return {"type": "string", "minLength": 1, "maxLength": 256, "pattern": r"\S"}


def _text():
    return {"type": "string", "maxLength": 2048}


def _sha256():
    return {"type": "string", "pattern": "^[0-9a-f]{64}$"}


def _prediction_schema():
    payload = PREDICTION_SCHEMA.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PREDICTION_SCHEMA_SHA256:
        raise ValueError("pinned World-State prediction schema content mismatch")
    return json.loads(payload)


def _prediction_reply_schema():
    payload = PREDICTION_REPLY_SCHEMA.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PREDICTION_REPLY_SCHEMA_SHA256:
        raise ValueError("pinned World-State prediction reply schema content mismatch")
    return json.loads(payload)


def _definitions():
    pinned = _prediction_schema()
    definitions = deepcopy(pinned["$defs"])
    embedded = {key: deepcopy(value) for key, value in pinned.items()
                if key not in ("$schema", "$id", "title", "$defs")}
    # Existing wire fields and definitions remain exact; the enclosing service
    # is explicitly a smaller bounded profile of their allowed dimensions.
    embedded["allOf"] = [{"properties": {
        "history": {"maxItems": 32}, "candidates": {"maxItems": 8}}}]
    definitions["prediction_request"] = embedded
    definitions["task_kind"] = {"enum": ["PICKUP", "SCORE", "TRANSIT", "HOLD"]}
    definitions["goal"] = _object({
        "field_pose": _ref("pose"),
        "position_tolerance_m": _number(nonnegative=True),
        "heading_tolerance_rad": _number(nonnegative=True),
        "velocity_tolerance_mps": _number(nonnegative=True),
    })
    definitions["context"] = _object({
        "revision": _integer(), "time_us": _integer(), "source_id": _identity(),
        "possession": {"enum": ["EMPTY", "HELD", "UNKNOWN"]},
        "mechanism_ready": {"type": ["boolean", "null"]},
        "enabled": {"type": "boolean"}, "command_authority": {"type": "boolean"},
        "cancel_requested": {"type": "boolean"},
    })
    definitions["obstacle_circle"] = _object({
        "obstacle_id": _identity(), "center_m": _ref("vec2"),
        "velocity_mps": _ref("vec2"), "radius_m": _number(nonnegative=True),
    })
    definitions["obstacles"] = _object({
        "map_version": _integer(), "time_us": _integer(), "source_id": _identity(),
        "status": {"enum": ["AVAILABLE", "UNAVAILABLE"]},
        "circles": _array(_ref("obstacle_circle"), 128),
    })
    definitions["path_evidence"] = _object({
        "status": {"enum": ["UNASSESSED", "REACHABLE", "BLOCKED"]},
        "distance_m": _nullable(_number(nonnegative=True)),
        "planner_request_id": _nullable(_identity()),
        "valid_until_us": _nullable(_integer()),
    })
    definitions["task"] = _object({
        "task_id": _identity(), "candidate_id": _identity(), "kind": _ref("task_kind"),
        "goal": _nullable(_ref("goal")),
        "target_track_ids": _array(_integer(), 32, unique=True),
        "base_utility": _number(), "path_evidence": _ref("path_evidence"),
    })
    return definitions


def _schema(title, properties, definitions, name):
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://frcworldmodel.local/schemas/" + name,
        "title": title, **_object(properties), "$defs": definitions,
    }


def build_request_schema():
    """Return a fresh request schema; it contains only local fragment refs."""
    return _schema("FRC advisory decision-support request v1", {
        "schema_version": {"const": SCHEMA_VERSION},
        "service_epoch": _identity(), "decision_generation": _integer(),
        "task_revision": _integer(), "policy_id": _identity(),
        "mode": {"enum": ["replay", "shadow"]},
        "action_sequence_hash": _sha256(),
        "prediction_request": _ref("prediction_request"), "context": _ref("context"),
        "current_task_id": _nullable(_identity()), "obstacles": _ref("obstacles"),
        "tasks": _array(_ref("task"), 8, 1),
    }, _definitions(), "decision-request-v1.schema.json")


def build_reply_schema():
    """Return a fresh reply schema with explicit advisory-only authority."""
    definitions = _definitions()
    pinned = _prediction_schema()
    pinned_reply = _prediction_reply_schema()
    if any(definitions.get(key) != value for key, value in pinned_reply["$defs"].items()):
        raise ValueError("pinned predictor request/reply definitions differ")
    definitions["prediction_reply"] = {
        key: deepcopy(value) for key, value in pinned_reply.items()
        if key not in ("$schema", "$id", "title", "$defs")
    }
    identities = {key: deepcopy(pinned["properties"][key]) for key in IDENTITY_KEYS}
    identities.update(obstacle_map_version=_integer(), context_revision=_integer())
    definitions["identity"] = _object(identities)
    sample = _object({
        "offset_us": _integer(), "valid_mask": {"type": "boolean"},
        "x_m": _number(), "y_m": _number(), "vx_mps": _number(), "vy_mps": _number(),
        "physical_covariance": _nullable(_ref("covariance")),
    })
    sample["allOf"] = [{
        "if": {"properties": {"valid_mask": {"const": False}}},
        "then": {"properties": {key: {"const": None if key == "physical_covariance" else 0}
                                 for key in ("x_m", "y_m", "vx_mps", "vy_mps", "physical_covariance")}},
    }]
    definitions["outcome_sample"] = sample
    definitions["outcome"] = _object({
        "samples": _array(_ref("outcome_sample"), 256),
        "remaining_goal_distance_m": _nullable(_number(nonnegative=True)),
        "progress_m": _number(), "route_distance_m": _nullable(_number(nonnegative=True)),
        "travel_time_proxy_s": _nullable(_number(nonnegative=True)),
        "distance_basis": {"enum": ["NONE", "PLANNER_EVIDENCE", "STRAIGHT_LINE_LOWER_BOUND"]},
        "minimum_clearance_m": _nullable(_number()),
        "uncertainty_radius_m": _number(nonnegative=True),
        "task_success_probability": {"type": "null"},
        "task_completion_time_s": {"type": "null"}, "mechanism_outcome": {"type": "null"},
    })
    ranking = _object({
        "task_id": _identity(), "candidate_id": _identity(), "kind": _ref("task_kind"),
        "eligible": {"type": "boolean"}, "reasons": _array(_text(), 32),
        "score": _nullable(_number()), "baseline_score": _nullable(_number()),
        "forecast_kind": {"enum": ["KINEMATIC_ASSUMPTION", "LEARNED_FORECAST"]},
        "fallback_reason": _nullable(_text()), "outcome": _ref("outcome"),
        "goal": _nullable(_ref("goal")), "requires_path_planning": {"type": "boolean"},
    })
    ranking["allOf"] = [{
        "if": {"properties": {"forecast_kind": {"const": "KINEMATIC_ASSUMPTION"}}},
        "then": {"properties": {"outcome": {"properties": {
            "samples": {"items": {"properties": {"physical_covariance": {"type": "null"}}}}
        }}}},
    }]
    definitions["ranking"] = ranking
    schema = _schema("FRC advisory decision-support reply v1", {
        "schema_version": {"const": SCHEMA_VERSION},
        "service_epoch": _identity(), "decision_generation": _integer(),
        "task_revision": _integer(), "policy_id": _identity(),
        "mode": {"enum": ["replay", "shadow"]}, "request_sha256": _sha256(),
        "action_sequence_hash": _sha256(), "identity": _ref("identity"),
        "generated_us": _integer(), "valid_until_us": _integer(),
        "status": {"enum": ["BASELINE", "LEARNED_ADVISORY", "REJECTED", "CANCELLED"]},
        "reason": _nullable(_text()), "selected_task_id": _nullable(_identity()),
        "baseline_selected_task_id": _nullable(_identity()),
        "rankings": _array(_ref("ranking"), 8),
        "measured": {"const": False}, "influences_commands": {"const": False},
        "requires_planner_validation": {"const": True},
        "learned_prediction_reply": _nullable(_ref("prediction_reply")),
        "learned_domain_id": _nullable(_identity()), "learned_status": _identity(),
    }, definitions, "decision-reply-v1.schema.json")
    schema["allOf"] = [{
        "if": {"properties": {"rankings": {"contains": {
            "properties": {"forecast_kind": {"const": "LEARNED_FORECAST"}},
            "required": ["forecast_kind"],
        }}}},
        "then": {"properties": {
            "learned_prediction_reply": _ref("prediction_reply"),
            "learned_domain_id": _identity(),
        }},
    }]
    return schema


def write_schemas(*directories):
    """Write identical public/package schema bytes to explicitly supplied dirs."""
    schemas = {REQUEST_SCHEMA.name: build_request_schema(), REPLY_SCHEMA.name: build_reply_schema()}
    for directory in directories:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        for name, schema in schemas.items():
            (destination / name).write_text(json.dumps(schema, indent=2, allow_nan=False) + "\n")
