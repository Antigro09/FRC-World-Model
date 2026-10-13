"""Versioned advisory decisions around the unchanged World-State wire contract.

These gates run off the periodic robot path. A valid recommendation never
authorizes motion, changes a measured fact, or establishes physical safety.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from .bindings import ContractError, IDENTITIES, WirePolicy, _finite, _pairs, validate_reply, validate_request

CONTRACT = Path(__file__).with_name("_contracts") / "decision-support-v1"
VERSION = "frc-decision-support/1"


def canonical_json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_hash(request):
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def action_sequence_hash(prediction_request):
    return request_hash(prediction_request["candidates"])


@lru_cache(maxsize=2)
def _validator(kind):
    from jsonschema import Draft202012Validator
    schema = json.loads((CONTRACT / f"decision-{kind}-v1.schema.json").read_text())
    return Draft202012Validator(schema)


def load_decision(payload, kind="request", *, max_bytes=1_000_000):
    if kind not in ("request", "reply") or not isinstance(payload, str) or len(payload.encode("utf-8")) > max_bytes:
        raise ContractError("unsupported or oversized decision document")
    try:
        raw = json.loads(payload, object_pairs_hook=_pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(ContractError("nonfinite JSON")))
        _finite(raw)
        _validator(kind).validate(raw)
        return raw
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("decision schema validation: " + type(error).__name__) from error


@dataclass(frozen=True)
class DecisionPolicy:
    """Supplied domain and scoring configuration; no production physical defaults."""
    policy_id: str
    wire_policy: WirePolicy
    footprint_radius_m: float
    clearance_margin_m: float
    uncertainty_sigma: float
    ego_error_growth_mps: float
    obstacle_error_growth_mps: float
    nominal_speed_mps: float
    travel_cost_per_m: float
    time_cost_per_s: float
    risk_cost: float
    uncertainty_cost_per_m: float
    switching_cost: float
    progress_reward_per_m: float
    max_context_age_us: int = 250_000
    max_track_age_us: int = 250_000
    max_request_ttl_us: int = 500_000
    max_candidates: int = 8
    max_history: int = 32
    max_obstacles: int = 128
    max_samples: int = 101
    max_bytes: int = 1_000_000
    inference_timeout_s: float = .02
    computation_budget_s: float = .25
    approved_domain_id: str = "unqualified"

    def __post_init__(self):
        positive = (self.footprint_radius_m, self.uncertainty_sigma, self.nominal_speed_mps)
        nonnegative = (self.clearance_margin_m, self.ego_error_growth_mps, self.obstacle_error_growth_mps,
                       self.travel_cost_per_m, self.time_cost_per_s, self.risk_cost, self.uncertainty_cost_per_m,
                       self.switching_cost, self.progress_reward_per_m)
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in positive):
            raise ContractError("invalid positive decision policy value")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in nonnegative):
            raise ContractError("invalid nonnegative decision policy value")
        caps = ((self.max_context_age_us, 2_000_000), (self.max_track_age_us, 2_000_000),
                (self.max_request_ttl_us, 2_000_000), (self.max_candidates, 8), (self.max_history, 32),
                (self.max_obstacles, 128), (self.max_samples, 101), (self.max_bytes, 1_000_000))
        if any(type(n) is not int or not 1 <= n <= cap for n, cap in caps):
            raise ContractError("decision resource bounds outside hard ceilings")
        if not 0 < self.inference_timeout_s <= .05 or not 0 < self.computation_budget_s <= 1:
            raise ContractError("invalid host computation budget")
        if not self.policy_id or len(self.policy_id) > 256 or not self.approved_domain_id:
            raise ContractError("missing decision policy identity")
        if self.nominal_speed_mps > self.wire_policy.max_speed_mps:
            raise ContractError("nominal speed exceeds configured domain")


def load_policy(path):
    """Read an explicit local policy file; callers supply and review its domain."""
    path = Path(path)
    if path.stat().st_size > 64_000:
        raise ContractError("decision policy file too large")
    raw = json.loads(path.read_text(), object_pairs_hook=_pairs,
                     parse_constant=lambda _: (_ for _ in ()).throw(ContractError("nonfinite policy")))
    _finite(raw)
    wire = raw.pop("wire_policy")
    wire["model_calibrations"] = {k: tuple(v) for k, v in wire["model_calibrations"].items()}
    wire["allowed_fields"] = tuple(tuple(v) for v in wire["allowed_fields"])
    wire["allowed_configuration_revisions"] = tuple(wire["allowed_configuration_revisions"])
    wire["mechanisms"] = {k: tuple(v) for k, v in wire["mechanisms"].items()}
    return DecisionPolicy(wire_policy=WirePolicy(**wire), **raw)


def authority_for(request):
    r = request["prediction_request"]
    return {"field": r["field"], "source": r["source"], "epoch": r["epoch"], "task_revision": request["task_revision"]}


def decision_identity(request):
    r = request["prediction_request"]
    return {**{key: r[key] for key in IDENTITIES},
            "obstacle_map_version": r["history"][-1]["snapshot"]["obstacle_map_version"],
            "context_revision": request["context"]["revision"]}


def validate_decision_request(request, policy, now_us):
    # Roundtrip freezes caller-owned dictionaries and applies strict schema.
    request = load_decision(canonical_json(request), max_bytes=policy.max_bytes)
    r = validate_request(request["prediction_request"], now_us=now_us,
                         max_sync_error_us=policy.wire_policy.max_sync_error_us, policy=policy.wire_policy)
    if request["policy_id"] != policy.policy_id:
        raise ContractError("decision policy mismatch")
    if request["action_sequence_hash"] != action_sequence_hash(r):
        raise ContractError("action sequence hash mismatch")
    if r["valid_until_us"] - r["issued_us"] > policy.max_request_ttl_us:
        raise ContractError("decision expiry exceeds resource bound")
    if len(r["candidates"]) > policy.max_candidates or len(r["history"]) > policy.max_history or r["horizon_us"] // r["step_us"] + 1 > policy.max_samples:
        raise ContractError("decision candidate/history/horizon capacity")
    latest = r["history"][-1]["snapshot"]
    if latest["ego"]["time_us"] != r["history_cutoff_us"]:
        raise ContractError("latest ego is not anchored at history cutoff")
    context, obstacles = request["context"], request["obstacles"]
    if not 0 <= r["history_cutoff_us"] - context["time_us"] <= policy.max_context_age_us or now_us - context["time_us"] > policy.max_context_age_us:
        raise ContractError("robot context stale/future")
    if obstacles["map_version"] != latest["obstacle_map_version"] or not 0 <= r["history_cutoff_us"] - obstacles["time_us"] <= policy.max_context_age_us or now_us - obstacles["time_us"] > policy.max_context_age_us:
        raise ContractError("obstacle envelope map/time mismatch")
    circles = obstacles["circles"]
    if len(circles) > policy.max_obstacles or len({c["obstacle_id"] for c in circles}) != len(circles):
        raise ContractError("obstacle envelope capacity/identity")
    for circle in circles:
        if not policy.wire_policy.inside(circle["center_m"]["x"], circle["center_m"]["y"]) or math.hypot(*circle["velocity_mps"].values()) > policy.wire_policy.max_speed_mps:
            raise ContractError("obstacle envelope outside configured domain")
    tasks = request["tasks"]
    if len(tasks) != len(r["candidates"]) or {t["candidate_id"] for t in tasks} != {c["candidate_id"] for c in r["candidates"]} or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ContractError("task/candidate pairing must be one-to-one")
    known = {track["id"] for track in latest["tracks"]}
    for task in tasks:
        if len(set(task["target_track_ids"])) != len(task["target_track_ids"]) or not set(task["target_track_ids"]).issubset(known):
            raise ContractError("unknown/duplicate task target")
        if task["kind"] == "PICKUP" and not task["target_track_ids"]:
            raise ContractError("pickup requires authoritative target references")
        if (task["kind"] == "HOLD") != (task["goal"] is None):
            raise ContractError("task goal semantics mismatch")
        if task["goal"]:
            p = task["goal"]["field_pose"]["position_m"]
            if not policy.wire_policy.inside(p["x"], p["y"]):
                raise ContractError("task goal outside configured field")
        evidence = task["path_evidence"]
        if evidence["status"] == "REACHABLE":
            if evidence["distance_m"] is None or evidence["planner_request_id"] is None or evidence["valid_until_us"] is None or not now_us < evidence["valid_until_us"] <= r["valid_until_us"]:
                raise ContractError("missing/stale planner evidence")
        elif any(evidence[k] is not None for k in ("distance_m", "planner_request_id", "valid_until_us")):
            raise ContractError("unassessed/blocked path cannot claim reachability")
        candidate = next(c for c in r["candidates"] if c["candidate_id"] == task["candidate_id"])
        if task["kind"] == "HOLD" and any(any(s["body_twist"].values()) or s["mechanisms"] for s in candidate["samples"]):
            raise ContractError("HOLD requires neutral chassis and no mechanism commands")
    return request


def consume_decision_reply(reply, request, *, policy, now_us, current_request):
    request = validate_decision_request(request, policy, now_us)
    current = validate_decision_request(current_request, policy, now_us)
    reply = load_decision(canonical_json(reply), "reply", max_bytes=policy.max_bytes)
    if request_hash(current) != request_hash(request):
        raise ContractError("current decision context changed; recompute")
    for key in ("service_epoch", "decision_generation", "task_revision", "policy_id", "mode", "action_sequence_hash"):
        if reply[key] != request[key]:
            raise ContractError("reply decision identity mismatch")
    if reply["identity"] != decision_identity(request) or reply["request_sha256"] != request_hash(request):
        raise ContractError("reply source/map/model/context/hash mismatch")
    r = request["prediction_request"]
    if not r["issued_us"] <= reply["generated_us"] <= now_us < reply["valid_until_us"] <= r["valid_until_us"]:
        raise ContractError("decision reply stale/future/expired")
    if reply["status"] in ("REJECTED", "CANCELLED"):
        raise ContractError("decision recommendation unavailable")
    expected = {(t["task_id"], t["candidate_id"]) for t in request["tasks"]}
    if {(t["task_id"], t["candidate_id"]) for t in reply["rankings"]} != expected or len(reply["rankings"]) != len(expected):
        raise ContractError("decision ranking dimensions")
    selected = next((t for t in reply["rankings"] if t["task_id"] == reply["selected_task_id"]), None)
    if reply["selected_task_id"] is not None and (selected is None or not selected["eligible"]):
        raise ContractError("selected task is not eligible")
    for ranking in reply["rankings"]:
        task = next(t for t in request["tasks"] if t["task_id"] == ranking["task_id"])
        if ranking["kind"] != task["kind"] or ranking["goal"] != task["goal"]:
            raise ContractError("reply task/goal mutated")
        samples = ranking["outcome"]["samples"]
        if [s["offset_us"] for s in samples] != list(range(0, r["horizon_us"] + 1, r["step_us"])):
            raise ContractError("decision forecast grid mismatch")
        if ranking["forecast_kind"] == "KINEMATIC_ASSUMPTION" and any(s["physical_covariance"] is not None for s in samples):
            raise ContractError("kinematic assumptions cannot claim physical covariance")
    evidence = reply["learned_prediction_reply"]
    if evidence is not None:
        if policy.approved_domain_id == "unqualified" or reply["learned_domain_id"] != policy.approved_domain_id:
            raise ContractError("unqualified learned domain")
        evidence = validate_reply(r, evidence, now_us=now_us, policy=policy.wire_policy)
    elif reply["learned_domain_id"] is not None:
        raise ContractError("missing learned evidence")
    # Reproduce advisory scores from the original immutable candidates. This
    # also rejects forged eligibility, covariance, goals, selection, and samples.
    from .decision_service import rank_decision, _select
    expected_rankings, expected_baseline, learned = rank_decision(request, policy,
        now_us=reply["generated_us"], learned_prediction_reply=evidence, learned_status=reply["learned_status"])
    if canonical_json(reply["rankings"]) != canonical_json(expected_rankings) or reply["selected_task_id"] != _select(expected_rankings) or reply["baseline_selected_task_id"] != expected_baseline:
        raise ContractError("decision ranking/provenance/selection inconsistent with evidence")
    if reply["status"] != ("LEARNED_ADVISORY" if learned else "BASELINE"):
        raise ContractError("decision status does not match evidence")
    # Sensing prerequisites can age while transport runs. Re-rank at consumption
    # time and reject an eligible task that has since become ineligible.
    current_rankings, _, _ = rank_decision(request, policy, now_us=now_us,
        learned_prediction_reply=evidence, learned_status=reply["learned_status"])
    current_by_id = {row["task_id"]: row for row in current_rankings}
    if any(row["eligible"] and not current_by_id[row["task_id"]]["eligible"] for row in reply["rankings"]):
        raise ContractError("decision sensing prerequisites expired in transport")
    return reply
