"""Replay evaluation and analytical baselines on identical strict offline windows.

Predictor callbacks receive no future measured states. All scores are empirical
forecast errors against the recorded label provenance, never physical covariance.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .data import FeatureLayout, LogRecord, Window, canonical_hash, wrap_angle


@dataclass(frozen=True)
class PredictorInput:
    cutoff_ns: int
    timestep_ns: int
    history: tuple[tuple[float, ...], ...]
    history_actions: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]
    episode_key: tuple[Any, ...]
    window_id: str

    @classmethod
    def from_window(cls, window: Window) -> "PredictorInput":
        return cls(window.cutoff_ns, window.timestep_ns, tuple(r.values for r in window.history),
                   tuple(r.values for r in window.history_actions), tuple(r.values for r in window.actions),
                   window.episode_key, window.window_id)


@dataclass(frozen=True)
class EgoLayout:
    x: int
    y: int
    heading: int
    vx_robot: int
    vy_robot: int
    omega: int

    def validate(self, layout: FeatureLayout) -> None:
        indices = tuple(asdict(self).values())
        if len(set(indices)) != 6 or any(type(i) is not int or i < 0 or i >= len(layout.names) for i in indices):
            raise ValueError("six distinct ego indices required")
        expected = ("m", "m", "rad", "m/s", "m/s", "rad/s")
        if tuple(layout.units[i] for i in indices) != expected or self.heading not in layout.angle_indices:
            raise ValueError("ego units/angle mismatch")


@dataclass(frozen=True)
class MechanismResponse:
    state_index: int
    action_index: int
    time_constant_s: float
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class AnalyticalConfig:
    ego: EgoLayout
    command_indices: tuple[int, int, int]
    velocity_time_constants_s: tuple[float, float, float]
    command_limits: tuple[float, float, float] | None = None
    mechanisms: tuple[MechanismResponse, ...] = ()
    calibration_source: str = "unprovided; synthetic plumbing only"

    def validate(self, state_layout: FeatureLayout, action_layout: FeatureLayout) -> None:
        self.ego.validate(state_layout)
        if len(self.command_indices) != 3 or len(set(self.command_indices)) != 3 or any(type(i) is not int or i < 0 or i >= len(action_layout.names) for i in self.command_indices):
            raise ValueError("three distinct command indices required")
        if tuple(action_layout.units[i] for i in self.command_indices) != ("m/s", "m/s", "rad/s"):
            raise ValueError("robot-relative command units mismatch")
        if len(self.velocity_time_constants_s) != 3 or any(not math.isfinite(t) or t <= 0 for t in self.velocity_time_constants_s):
            raise ValueError("positive finite response constants required")
        if self.command_limits is not None and (len(self.command_limits) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.command_limits)):
            raise ValueError("invalid command limits")
        seen = set()
        for mechanism in self.mechanisms:
            if not 0 <= mechanism.state_index < len(state_layout.names) or not 0 <= mechanism.action_index < len(action_layout.names):
                raise ValueError("mechanism index")
            if mechanism.state_index in seen or mechanism.state_index in asdict(self.ego).values():
                raise ValueError("mechanism state overlap")
            seen.add(mechanism.state_index)
            if state_layout.units[mechanism.state_index] != action_layout.units[mechanism.action_index]:
                raise ValueError("mechanism unit mismatch")
            if not math.isfinite(mechanism.time_constant_s) or mechanism.time_constant_s <= 0:
                raise ValueError("invalid mechanism response constant")
            if any(x is not None and not math.isfinite(x) for x in (mechanism.minimum, mechanism.maximum)):
                raise ValueError("invalid mechanism limit")
            if mechanism.minimum is not None and mechanism.maximum is not None and mechanism.minimum > mechanism.maximum:
                raise ValueError("mechanism minimum exceeds maximum")


def persistence_forecast(inputs: PredictorInput) -> tuple[tuple[float, ...], ...]:
    return tuple(inputs.history[-1] for _ in inputs.actions)


def _integrate_body_velocity(state: list[float], ego: EgoLayout, vx: float, vy: float, omega: float, dt: float) -> None:
    """Exact constant body-twist integration, with a stable zero-turn branch."""
    theta = state[ego.heading]
    if abs(omega * dt) < 1e-8:
        # Midpoint tends to exact straight motion and avoids division cancellation.
        middle = theta + omega * dt / 2
        dx = (vx * math.cos(middle) - vy * math.sin(middle)) * dt
        dy = (vx * math.sin(middle) + vy * math.cos(middle)) * dt
    else:
        end = theta + omega * dt
        dx = (vx * (math.sin(end) - math.sin(theta)) + vy * (math.cos(end) - math.cos(theta))) / omega
        dy = (vx * (math.cos(theta) - math.cos(end)) + vy * (math.sin(end) - math.sin(theta))) / omega
    state[ego.x] += dx
    state[ego.y] += dy
    state[ego.heading] = wrap_angle(theta + omega * dt)


def constant_velocity_forecast(inputs: PredictorInput, ego: EgoLayout) -> tuple[tuple[float, ...], ...]:
    """Constant accepted state velocity in robot-relative frame; mechanism holds."""
    state = list(inputs.history[-1])
    dt = inputs.timestep_ns / 1e9
    forecast = []
    for _ in inputs.actions:
        _integrate_body_velocity(state, ego, state[ego.vx_robot], state[ego.vy_robot], state[ego.omega], dt)
        forecast.append(tuple(state))
    return tuple(forecast)


def analytical_response_forecast(inputs: PredictorInput, config: AnalyticalConfig) -> tuple[tuple[float, ...], ...]:
    """Advisory first-order command response; constants need real calibration.

    Chassis integration uses the interval-average first-order velocity. Heading
    uses the same average omega; coupled accelerating twist is an approximation.
    No mechanism mode is inferred from an untimed goal or waypoint.
    """
    state = list(inputs.history[-1])
    dt = inputs.timestep_ns / 1e9
    velocities = (config.ego.vx_robot, config.ego.vy_robot, config.ego.omega)
    forecast = []
    for action in inputs.actions:
        average = []
        for i, (state_index, action_index, tau) in enumerate(zip(velocities, config.command_indices, config.velocity_time_constants_s)):
            command = action[action_index]
            if config.command_limits is not None:
                limit = config.command_limits[i]
                command = max(-limit, min(limit, command))
            prior = state[state_index]
            decay = math.exp(-dt / tau)
            average.append(command + (prior - command) * tau * (1 - decay) / dt)
            state[state_index] = command + (prior - command) * decay
        _integrate_body_velocity(state, config.ego, *average, dt)
        for mechanism in config.mechanisms:
            command = action[mechanism.action_index]
            if mechanism.minimum is not None:
                command = max(mechanism.minimum, command)
            if mechanism.maximum is not None:
                command = min(mechanism.maximum, command)
            prior = state[mechanism.state_index]
            state[mechanism.state_index] = command + (prior - command) * math.exp(-dt / mechanism.time_constant_s)
        forecast.append(tuple(state))
    return tuple(forecast)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("nonempty values and valid fraction required")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {"count": len(values), "mae": sum(values) / len(values), "rmse": math.sqrt(sum(x*x for x in values) / len(values)),
            "p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "p99": percentile(values, 0.99), "max": max(values)}


def evaluate_windows(windows: Iterable[Window], predictor: Callable[[PredictorInput], Sequence[Sequence[float]]],
                     state_layout: FeatureLayout, *, ego: EgoLayout | None = None, name: str = "unnamed", mode: str = "open_loop") -> dict[str, Any]:
    """One-step/open-loop/refreshed reports stay distinct; no teacher forcing.

    Refreshed mode means independent logged windows are scored from each window's
    measured cutoff. This evaluator never refreshes within a multi-step rollout.
    """
    if mode not in ("one_step", "open_loop", "refreshed"):
        raise ValueError("invalid evaluation mode")
    if ego is not None:
        ego.validate(state_layout)
    metrics: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sessions: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    failures: dict[str, int] = defaultdict(int)
    provenance: dict[str, int] = defaultdict(int)
    evaluated_ids = []
    for window in windows:
        inputs = PredictorInput.from_window(window)
        try:
            forecast = tuple(tuple(float(x) for x in row) for row in predictor(inputs))
            if len(forecast) != len(window.targets) or any(len(row) != len(state_layout.names) for row in forecast):
                raise ValueError("forecast_shape")
            if any(not math.isfinite(x) for row in forecast for x in row):
                raise ValueError("forecast_nonfinite")
        except Exception as exc:
            # Failures count as missing predictions. Production fallback is owned
            # by the service; no fabricated success/error values enter metrics.
            failures[type(exc).__name__ + ":" + str(exc)] += 1
            continue
        evaluated_ids.append(window.window_id)
        for step, (prediction, target) in enumerate(zip(forecast, window.targets), 1):
            if mode == "one_step" and step > 1:
                break
            error = [wrap_angle(p - t) if i in state_layout.angle_indices else p - t for i, (p, t) in enumerate(zip(prediction, target.values))]
            values = {feature: abs(error[i]) for i, feature in enumerate(state_layout.names)}
            if ego is not None:
                values["position_error_m"] = math.hypot(error[ego.x], error[ego.y])
                values["heading_error_rad"] = abs(error[ego.heading])
                values["velocity_error_mps"] = math.hypot(error[ego.vx_robot], error[ego.vy_robot])
                values["omega_error_radps"] = abs(error[ego.omega])
            horizon_ns = step * window.timestep_ns
            for feature, value in values.items():
                metrics[horizon_ns][feature].append(value)
                sessions[str(window.episode_key[0])][horizon_ns][feature].append(value)
            provenance[target.label_provenance] += 1
    result = {"name": name, "mode": mode, "evaluated_window_ids":evaluated_ids,
            "successful_windows": len(evaluated_ids), "missing_predictions": dict(failures), "label_provenance": dict(provenance),
            "per_horizon_ns": {str(h): {feature: _summary(v) for feature, v in sorted(features.items())} for h, features in sorted(metrics.items())},
            "per_session": {session: {str(h): {feature: _summary(v) for feature, v in sorted(features.items())} for h, features in sorted(horizons.items())} for session, horizons in sorted(sessions.items())},
            "uncertainty_semantics": "empirical recorded-label error only; no sensor covariance or calibrated physical covariance", "ranking": "advisory; executed logs do not establish counterfactual ranking"}
    # Bind actual results, failures, layout and selected windows, not IDs alone.
    result["state_layout"] = asdict(state_layout)
    result["evaluation_hash"] = canonical_hash(result)
    return result


def ablate_inputs(predictor: Callable[[PredictorInput], Sequence[Sequence[float]]], *, history: bool = False, actions: bool = False,
                  neutral_action: Sequence[float] | None = None) -> Callable[[PredictorInput], Sequence[Sequence[float]]]:
    """Explicit named ablation, preserving horizon; zero is never assumed neutral."""
    if actions and neutral_action is None:
        raise ValueError("action ablation requires explicitly documented neutral action")
    def ablated(inputs: PredictorInput) -> Sequence[Sequence[float]]:
        future = inputs.actions
        past_actions = inputs.history_actions
        if actions:
            neutral = tuple(float(x) for x in neutral_action)
            if any(len(a) != len(neutral) for a in future) or any(not math.isfinite(x) for x in neutral):
                raise ValueError("neutral action shape/finite mismatch")
            future = tuple(neutral for _ in future)
            past_actions = tuple(neutral for _ in past_actions)
        return predictor(PredictorInput(inputs.cutoff_ns, inputs.timestep_ns,
                                        inputs.history[-1:] if history else inputs.history,
                                        () if history else past_actions, future, inputs.episode_key, inputs.window_id))
    return ablated


def calibrate_session_bounds(calibration_report: Mapping[str, Any], *, split: str, quantile: float = 0.95,
                             physical_tolerances: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Held-out session-max error bounds, recorded BEFORE inspecting final test.

    This finite-sample descriptive quantile is not a guaranteed coverage claim.
    Missing real tolerances keeps the result advisory.
    """
    if split != "calibration" or not 0 < quantile <= 1:
        raise ValueError("calibration-only report and quantile required")
    if physical_tolerances is not None and any(not math.isfinite(v) or v <= 0 for v in physical_tolerances.values()):
        raise ValueError("physical tolerances must be finite and positive")
    maxima: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for horizons in calibration_report["per_session"].values():
        for horizon, features in horizons.items():
            for feature, summary in features.items():
                maxima[horizon][feature].append(summary["max"])
    bounds = {h: {f: percentile(values, quantile) for f, values in sorted(features.items())} for h, features in sorted(maxima.items())}
    result = {"fit_split": "calibration", "quantile": quantile, "session_count": len(calibration_report["per_session"]),
              "bounds": bounds, "physical_tolerances": dict(physical_tolerances or {}),
              "advisory_only": True, "source_evaluation_hash": calibration_report["evaluation_hash"],
              "coverage_claim": "none; descriptive session-max quantile, requires held-out coverage verification"}
    result["calibration_hash"] = canonical_hash(result)
    return result
