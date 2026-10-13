"""Bounded, shadow-first adaptive task recommendations; no actuator interface."""
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import copy
import math
import threading
import time
import uuid

from .bindings import ContractError, validate_reply
from .decision_bindings import (VERSION, action_sequence_hash, authority_for, canonical_json,
    consume_decision_reply, decision_identity, load_decision, request_hash, validate_decision_request, _validator)
from .decision_dynamics import kinematic_forecasts, screen_forecast
from .transport import NetworkFailure, roundtrip


@dataclass(frozen=True)
class LearnedEvidence:
    reply: dict
    in_domain: bool
    domain_id: str


def _select(rankings):
    eligible = [r for r in rankings if r["eligible"]]
    return min(eligible, key=lambda r: (-r["score"], r["task_id"]))["task_id"] if eligible else None


def _wire_samples(samples):
    return [{"offset_us": s["offset_us"], "valid_mask": True,
             **{k: s[k] for k in ("x_m", "y_m", "vx_mps", "vy_mps")},
             "physical_covariance": s.get("physical_covariance")} for s in samples]


def _rank(request, forecasts, policy, *, learned_candidates=(), fallback_reason="learned-unavailable", now_us=None, deadline=None):
    r = request["prediction_request"]
    snapshot = r["history"][-1]["snapshot"]
    context = request["context"]
    now_us = r["issued_us"] if now_us is None else now_us
    tracks = {t["id"]: t for t in snapshot["tracks"]}
    obstacles = copy.deepcopy(request["obstacles"]["circles"])
    # Obstacle extrapolation retains its original stamp, never a new observation.
    age_s = (r["history_cutoff_us"] - request["obstacles"]["time_us"]) / 1e6
    for obstacle in obstacles:
        for axis in ("x", "y"):
            obstacle["center_m"][axis] += obstacle["velocity_mps"][axis] * age_s
        obstacle["radius_m"] += policy.obstacle_error_growth_mps * age_s
    rankings = []
    start = snapshot["ego"]["field_pose"]["position_m"]
    for task in request["tasks"]:
        if deadline is not None and time.monotonic() >= deadline:
            raise ContractError("decision candidate computation budget exhausted")
        samples = forecasts[task["candidate_id"]]
        reasons = []
        if context["cancel_requested"]:
            reasons.append("robot-task-cancelled")
        if task["kind"] != "HOLD":
            if not context["enabled"] or not context["command_authority"]:
                reasons.append("robot-execution-authority-unavailable")
            if request["obstacles"]["status"] != "AVAILABLE":
                reasons.append("occupancy-unavailable")
            if task["path_evidence"]["status"] == "BLOCKED":
                reasons.append("planner-blocked")
        if task["kind"] in ("PICKUP", "SCORE"):
            if context["mechanism_ready"] is not True:
                reasons.append("measured-mechanism-readiness-unavailable")
            wanted = "EMPTY" if task["kind"] == "PICKUP" else "HELD"
            if context["possession"] != wanted:
                reasons.append("measured-possession-prerequisite")
        for track_id in task["target_track_ids"]:
            track = tracks[track_id]
            if now_us - track["last_measurement_us"] > policy.max_track_age_us or track["lifecycle"] != "CONFIRMED":
                reasons.append("target-stale-or-unconfirmed")
            cov = track["uncertainty"]
            spread = math.hypot(cov["xx_m2"] - cov["yy_m2"], 2 * cov["xy_m2"])
            if (cov["xx_m2"] + cov["yy_m2"] + spread) / 2 > policy.wire_policy.max_variance_m2:
                reasons.append("target-uncertainty-out-of-domain")
        screen = screen_forecast(samples, r, obstacles,
            footprint_radius_m=policy.footprint_radius_m, clearance_margin_m=policy.clearance_margin_m,
            uncertainty_sigma=policy.uncertainty_sigma, ego_error_growth_mps=policy.ego_error_growth_mps,
            obstacle_error_growth_mps=policy.obstacle_error_growth_mps)
        if task["kind"] != "HOLD" and screen["blocked"]:
            reasons.append("conservative-envelope-intersection")
        field_padding = policy.footprint_radius_m + policy.clearance_margin_m + screen["uncertainty_radius_m"]
        for a, b in zip(samples, samples[1:]):
            dt = (b["offset_us"] - a["offset_us"]) / 1e6
            curve_padding = (math.hypot(a["vx_mps"], a["vy_mps"]) * abs(a["omega_radps"]) * dt * dt / 8
                             if "omega_radps" in a else policy.wire_policy.max_speed_mps * dt / 2)
            field_padding = max(field_padding, policy.footprint_radius_m + policy.clearance_margin_m
                                + screen["uncertainty_radius_m"] + curve_padding)
        bounds = policy.wire_policy
        if task["kind"] != "HOLD" and any(not (bounds.min_x_m + field_padding <= s["x_m"] <= bounds.max_x_m - field_padding
                and bounds.min_y_m + field_padding <= s["y_m"] <= bounds.max_y_m - field_padding) for s in samples):
            reasons.append("forecast-outside-field")
        end = samples[-1]
        remaining = route_distance = travel_time = None
        progress = 0.0
        distance_basis = "NONE"
        if task["goal"]:
            goal = task["goal"]["field_pose"]["position_m"]
            initial_distance = math.hypot(goal["x"] - start["x"], goal["y"] - start["y"])
            remaining = math.hypot(goal["x"] - end["x_m"], goal["y"] - end["y_m"])
            progress = initial_distance - remaining
            evidence = task["path_evidence"]
            route_distance = evidence["distance_m"] if evidence["status"] == "REACHABLE" else initial_distance
            distance_basis = "PLANNER_EVIDENCE" if evidence["status"] == "REACHABLE" else "STRAIGHT_LINE_LOWER_BOUND"
            travel_time = route_distance / policy.nominal_speed_mps
        clearance = screen["minimum_clearance_m"]
        risk_proxy = 0.0 if clearance is None else 1.0 / (1.0 + max(0.0, clearance))
        switching = policy.switching_cost if request["current_task_id"] not in (None, task["task_id"]) else 0.0
        score = (task["base_utility"] - policy.travel_cost_per_m * (route_distance or 0.0)
                 - policy.time_cost_per_s * (travel_time or 0.0) - policy.risk_cost * risk_proxy
                 - policy.uncertainty_cost_per_m * screen["uncertainty_radius_m"] - switching
                 + policy.progress_reward_per_m * progress)
        if not math.isfinite(score):
            raise ContractError("decision score overflow")
        learned = task["candidate_id"] in learned_candidates
        rankings.append({"task_id": task["task_id"], "candidate_id": task["candidate_id"], "kind": task["kind"],
            "eligible": not reasons, "reasons": sorted(set(reasons)), "score": score if not reasons else None,
            "baseline_score": score if not reasons else None,
            "forecast_kind": "LEARNED_FORECAST" if learned else "KINEMATIC_ASSUMPTION",
            "fallback_reason": None if learned else fallback_reason,
            "outcome": {"samples": _wire_samples(samples), "remaining_goal_distance_m": remaining,
                "progress_m": progress, "route_distance_m": route_distance, "travel_time_proxy_s": travel_time,
                "distance_basis": distance_basis, "minimum_clearance_m": clearance,
                "uncertainty_radius_m": screen["uncertainty_radius_m"], "task_success_probability": None,
                "task_completion_time_s": None, "mechanism_outcome": None},
            "goal": task["goal"], "requires_path_planning": task["kind"] != "HOLD"})
    return sorted(rankings, key=lambda row: (not row["eligible"], -(row["score"] or 0), row["task_id"]))


def _reply(request, now_us, *, status, reason=None, rankings=(), baseline_selected=None, learned_status="unavailable",
           learned_prediction_reply=None, learned_domain_id=None):
    r = request["prediction_request"]
    return {"schema_version": VERSION, **{k: request[k] for k in
        ("service_epoch", "decision_generation", "task_revision", "policy_id", "mode", "action_sequence_hash")},
        "request_sha256": request_hash(request), "identity": decision_identity(request), "generated_us": now_us,
        "valid_until_us": r["valid_until_us"], "status": status, "reason": reason,
        "selected_task_id": _select(rankings), "baseline_selected_task_id": baseline_selected,
        "rankings": list(rankings), "measured": False, "influences_commands": False,
        "requires_planner_validation": True, "learned_status": learned_status,
        "learned_prediction_reply": learned_prediction_reply, "learned_domain_id": learned_domain_id}


def rank_decision(request, policy, *, now_us, learned_prediction_reply=None, learned_status="learned-unavailable", deadline=None):
    """Compare optional evidence with the same candidates and baseline constraints."""
    forecasts = kinematic_forecasts(request["prediction_request"])
    baseline = _rank(request, forecasts, policy, now_us=now_us, fallback_reason=learned_status, deadline=deadline)
    if learned_prediction_reply is None:
        return baseline, _select(baseline), set()
    learned_candidates = set()
    if learned_prediction_reply is not None:
        ego_target = next((t["target_id"] for t in request["prediction_request"]["targets"] if t["kind"] == "EGO"), None)
        for forecast in learned_prediction_reply["forecasts"]:
            if forecast["target_id"] != ego_target or not all(s["valid_mask"] for s in forecast["samples"]):
                continue
            forecasts[forecast["candidate_id"]] = [{**s, "physical_covariance": {
                "xx_m2": s["covariance_xx_m2"], "xy_m2": s["covariance_xy_m2"], "yy_m2": s["covariance_yy_m2"]}}
                for s in forecast["samples"]]
            learned_candidates.add(forecast["candidate_id"])
    rankings = _rank(request, forecasts, policy, learned_candidates=learned_candidates,
                     fallback_reason=learned_status, now_us=now_us, deadline=deadline)
    baseline_by_id = {entry["task_id"]: entry for entry in baseline}
    for entry in rankings:
        prior = baseline_by_id[entry["task_id"]]
        entry["baseline_score"] = prior["score"]
        if not prior["eligible"]:
            entry["eligible"], entry["score"] = False, None
            entry["reasons"] = sorted(set(entry["reasons"] + prior["reasons"]))
    rankings.sort(key=lambda row: (not row["eligible"], -(row["score"] or 0), row["task_id"]))
    return rankings, _select(baseline), learned_candidates


class DecisionService:
    """One evaluation and at most one optional inference at a time.

    The mapped robot clock is mandatory. A host clock only bounds computation.
    Uncooperative inference can occupy one worker after timeout; it cannot queue
    more work or publish a late result. Production needs a killable process.
    """
    def __init__(self, policy, *, authority, now_us, predictor=None):
        # Resolve local schemas/dependencies once during startup, outside any
        # incoming request's processing budget. No model or network I/O occurs.
        _validator("request")
        _validator("reply")
        self.policy, self.authority, self.now_us, self.predictor = policy, copy.deepcopy(authority), now_us, predictor
        self.service_epoch = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._work_slot = threading.Lock()
        self._inference_slot = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decision-shadow-inference")
        self._cancel = threading.Event()
        self._version = 0
        self._highwater = None
        self._queued_fence = None
        self._context_fence = None
        self._last_now = None
        self._closed = False

    def cancel(self):
        with self._lock:
            self._cancel.set()
            self._version += 1

    def reset(self, authority):
        with self._lock:
            self.cancel()
            self.authority = copy.deepcopy(authority)
            self.service_epoch = uuid.uuid4().hex
            self._highwater = self._queued_fence = self._last_now = self._context_fence = None
            return self.service_epoch

    def close(self):
        with self._lock:
            self._closed = True
            self.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _admission(self, request, now, queued_highwater=None):
        r = request["prediction_request"]
        high = (request["decision_generation"], r["snapshot_id"], r["history_cutoff_us"],
                request["context"]["revision"], r["history"][-1]["snapshot"]["obstacle_map_version"])
        if self._closed: return "service-stopped", high
        if request["service_epoch"] != self.service_epoch: return "service-epoch-mismatch", high
        if authority_for(request) != self.authority: return "robot-source-field-epoch-task-mismatch", high
        if self._last_now is not None and now < self._last_now: return "robot-clock-rollback", high
        if self._context_fence is not None and request["context"]["revision"] == self._context_fence["revision"]:
            prior = {k: v for k, v in self._context_fence.items() if k != "time_us"}
            current = {k: v for k, v in request["context"].items() if k != "time_us"}
            if prior != current:
                return "context-changed-without-revision", high
        for prior in (self._highwater, queued_highwater):
            if prior is not None and (high[0] <= prior[0] or any(a < b for a, b in zip(high[1:], prior[1:]))):
                return "duplicate-or-reordered-decision", high
        return None, high

    def _begin(self, request, now):
        with self._lock:
            reason, high = self._admission(request, now)
            if not reason and self._queued_fence is not None and high[0] < self._queued_fence[0]:
                reason = "superseded-before-worker-start"
            if reason:
                return reason, None, None
            self._last_now, self._highwater = now, high
            self._context_fence = copy.deepcopy(request["context"])
            self._cancel.set()
            self._cancel = threading.Event()
            self._version += 1
            return None, self._cancel, self._version

    def _optional(self, request, cancellation, deadline):
        if self.predictor is None:
            return None, "learned-unavailable"
        if not self._inference_slot.acquire(blocking=False):
            return None, "learned-overload"
        try:
            future = self._pool.submit(self.predictor, copy.deepcopy(request["prediction_request"]), cancellation)
        except RuntimeError:
            self._inference_slot.release()
            return None, "learned-service-stopped"
        future.add_done_callback(lambda _: self._inference_slot.release())
        try:
            evidence = future.result(timeout=max(0.000001, min(self.policy.inference_timeout_s, deadline - time.monotonic())))
            if not isinstance(evidence, LearnedEvidence) or evidence.in_domain is not True or evidence.domain_id != self.policy.approved_domain_id or self.policy.approved_domain_id == "unqualified":
                return None, "learned-out-of-domain-or-unqualified"
            reply = validate_reply(request["prediction_request"], evidence.reply, now_us=self.now_us(),
                                   policy=self.policy.wire_policy)
            return reply, "learned-qualified-shadow-only"
        except TimeoutError:
            cancellation.set()
            future.cancel()
            return None, "learned-timeout"
        except Exception:
            return None, "learned-rejected-or-unavailable"

    def evaluate(self, payload):
        deadline = time.monotonic() + self.policy.computation_budget_s
        raw = load_decision(payload, max_bytes=self.policy.max_bytes)
        now = self.now_us()
        try:
            request = validate_decision_request(raw, self.policy, now)
        except ContractError as error:
            return _reply(raw, now, status="REJECTED", reason=str(error))
        reason, cancellation, version = self._begin(request, now)
        if reason:
            return _reply(request, now, status="REJECTED", reason=reason)
        if not self._work_slot.acquire(blocking=False):
            return _reply(request, now, status="REJECTED", reason="decision-overload-use-reactive-fallback")
        try:
            learned_reply, learned_status = self._optional(request, cancellation, deadline)
            generated = self.now_us()
            rankings, baseline_selected, learned_candidates = rank_decision(request, self.policy,
                now_us=generated, learned_prediction_reply=learned_reply, learned_status=learned_status, deadline=deadline)
            with self._lock:
                if version != self._version:
                    return _reply(request, generated, status="CANCELLED", reason="superseded-or-cancelled")
            # A timeout sets the inference token but still permits its ready baseline.
            if generated < now or generated >= request["prediction_request"]["valid_until_us"] or time.monotonic() >= deadline:
                return _reply(request, generated, status="REJECTED", reason="decision-expired-or-budget-exhausted")
            try:
                validate_decision_request(request, self.policy, generated)
            except ContractError:
                return _reply(request, generated, status="REJECTED", reason="context-or-path-expired-during-computation")
            result = _reply(request, generated, status="LEARNED_ADVISORY" if learned_candidates else "BASELINE",
                rankings=rankings, baseline_selected=baseline_selected, learned_status=learned_status,
                learned_prediction_reply=learned_reply if learned_candidates else None,
                learned_domain_id=self.policy.approved_domain_id if learned_candidates else None)
            validated_result = load_decision(canonical_json(result), "reply", max_bytes=self.policy.max_bytes)
            published_now = self.now_us()
            if time.monotonic() >= deadline or published_now < generated or published_now >= request["prediction_request"]["valid_until_us"]:
                return _reply(request, published_now, status="REJECTED", reason="decision-expired-or-budget-exhausted")
            with self._lock:
                if version != self._version:
                    return _reply(request, published_now, status="CANCELLED", reason="superseded-or-cancelled")
            return validated_result
        except (ContractError, ArithmeticError):
            return _reply(request, self.now_us(), status="REJECTED", reason="invalid-or-overflowing-decision-computation")
        finally:
            self._work_slot.release()

    def __call__(self, payload):
        return canonical_json(self.evaluate(payload))


class LatestDecisionWorker:
    """One worker and one replaceable pending request, both bounded."""
    def __init__(self, service):
        self.service = service
        self._condition = threading.Condition()
        self._pending = None
        self._queued_highwater = None
        self._queue_epoch = service.service_epoch
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="latest-decision")
        self._thread.start()

    def submit(self, payload):
        # Bad input never cancels healthy work. No file/network I/O in this queue.
        now = self.service.now_us()
        request = validate_decision_request(load_decision(payload, max_bytes=self.service.policy.max_bytes),
                                            self.service.policy, now)
        future = Future()
        with self._condition:
            if self._stopped:
                raise ContractError("decision worker stopped")
            with self.service._lock:
                if self._queue_epoch != self.service.service_epoch:
                    self._queued_highwater = None
                    self._queue_epoch = self.service.service_epoch
                reason, high = self.service._admission(request, now, self._queued_highwater)
                if reason:
                    raise ContractError(reason)
                self._queued_highwater = high
                self.service._queued_fence = high
                self.service._context_fence = copy.deepcopy(request["context"])
                self.service.cancel()
            if self._pending is not None:
                self._pending[1].cancel()
            self._pending = payload, future
            self._condition.notify()
        return future

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopped or self._pending is not None)
                if self._stopped:
                    return
                payload, future = self._pending
                self._pending = None
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(self.service.evaluate(payload))
                except Exception as error:
                    future.set_exception(error)

    def close(self):
        with self._condition:
            self._stopped = True
            self.service.cancel()
            if self._pending is not None:
                self._pending[1].cancel()
                self._pending = None
            self._condition.notify()
        self._thread.join(timeout=1.1)


def safe_decision_roundtrip(address, request, *, policy, now_us, current_request, timeout_s=.05):
    """Offline loopback consumer plus deterministic fallback over identical inputs.

    Whole-robot integration must keep its existing reactive fallback on roboRIO.
    No invalid request can be repaired by a fabricated sensing fact.
    """
    validated = validate_decision_request(request, policy, now_us())
    try:
        reply = load_decision(roundtrip(address, canonical_json(validated), timeout_s=timeout_s,
                                       max_bytes=policy.max_bytes), "reply", max_bytes=policy.max_bytes)
        return consume_decision_reply(reply, validated, policy=policy, now_us=now_us(), current_request=current_request)
    except (NetworkFailure, ContractError):
        now = now_us()
        validated = validate_decision_request(validated, policy, now)
        if request_hash(current_request) != request_hash(validated):
            raise ContractError("request changed during unavailable inference; reactive robot fallback required")
        forecasts = kinematic_forecasts(validated["prediction_request"])
        rankings = _rank(validated, forecasts, policy, fallback_reason="remote-unavailable-timeout-or-rejected", now_us=now)
        result = _reply(validated, now, status="BASELINE", rankings=rankings, baseline_selected=_select(rankings),
                        learned_status="remote-unavailable-timeout-or-rejected")
        return consume_decision_reply(result, validated, policy=policy, now_us=now_us(), current_request=current_request)
