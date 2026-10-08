"""Provisional internal ego/mechanism baseline; never measured world state.

This layout is NOT an exported World-State binding. Its dimensions are explicit
until the upstream versioned schema and golden vectors are matched. It uses
robot-relative accepted/expected commands, metres, radians, and seconds.
"""
from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Optional, Tuple

STATE_LAYOUT = ("x_m", "y_m", "heading_rad", "vx_robot_mps", "vy_robot_mps",
                "omega_radps", "mechanism_position", "mechanism_velocity")
ACTION_LAYOUT = ("vx_robot_mps", "vy_robot_mps", "omega_radps", "mechanism_setpoint")
LAYOUT_ID = "provisional-internal-ego-mechanism-v1"
LAYOUT_STATUS = "pending-world-state-contract-match"


def wrap_angle(angle: float) -> float:
    """Canonical heading in [-pi, pi), preserving periodic semantics."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _finite(values: Iterable[float]) -> None:
    if any(not math.isfinite(v) for v in values):
        raise ValueError("non-finite physical value")


@dataclass(frozen=True)
class EgoMechanismState:
    x_m: float
    y_m: float
    heading_rad: float
    vx_robot_mps: float
    vy_robot_mps: float
    omega_radps: float
    mechanism_position: float
    mechanism_velocity: float

    def __post_init__(self) -> None:
        _finite(self.vector())

    def vector(self) -> Tuple[float, ...]:
        return tuple(getattr(self, key) for key in STATE_LAYOUT)

    @classmethod
    def from_vector(cls, value: Iterable[float]) -> "EgoMechanismState":
        values = tuple(value)
        if len(values) != len(STATE_LAYOUT):
            raise ValueError("state requires exactly eight provisional features")
        return cls(*values)


@dataclass(frozen=True)
class PhysicalCommand:
    """Command active over [timestamp_ns, timestamp_ns + dt_ns).

    expected: planner/controller adapter's expected *accepted* command sequence;
    applied: timestamped accepted command recorded by the robot. Neither a goal
    label nor an untimed waypoint qualifies. Mechanism units must be documented
    for each dataset/configuration; there is no default hardware unit.
    """
    timestamp_ns: int
    vx_robot_mps: float
    vy_robot_mps: float
    omega_radps: float
    mechanism_setpoint: float
    mechanism_mode: str = "position"
    provenance: str = "expected"
    accepted: bool = True
    execution_deviation: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int):
            raise ValueError("command timestamp must be integer nanoseconds")
        if self.timestamp_ns < 0:
            raise ValueError("negative command timestamp")
        _finite(self.vector())
        if self.mechanism_mode not in ("position", "hold"):
            raise ValueError("unsupported mechanism mode")
        if self.provenance not in ("applied", "expected") or not self.accepted:
            raise ValueError("only accepted applied/expected physical commands qualify")
        if self.execution_deviation is not None:
            if len(self.execution_deviation) != len(ACTION_LAYOUT):
                raise ValueError("execution deviation requires four physical features")
            _finite(self.execution_deviation)

    def vector(self) -> Tuple[float, ...]:
        return tuple(getattr(self, key) for key in ACTION_LAYOUT)


@dataclass(frozen=True)
class DynamicsConfig:
    dt_s: float = 0.02
    chassis_tau_s: float = 0.10
    heading_tau_s: float = 0.10
    mechanism_tau_s: float = 0.20
    max_vx_mps: float = 4.0
    max_vy_mps: float = 4.0
    max_omega_radps: float = 8.0
    mechanism_min: float = -1.0
    mechanism_max: float = 1.0
    mechanism_units: str = "provisional-normalized-unit"
    parameter_provenance: str = "synthetic-unfitted-fixture"

    def __post_init__(self) -> None:
        _finite(v for v in asdict(self).values() if isinstance(v, (int, float)))
        if min(self.dt_s, self.chassis_tau_s, self.heading_tau_s,
               self.mechanism_tau_s, self.max_vx_mps, self.max_vy_mps,
               self.max_omega_radps) <= 0:
            raise ValueError("timestep, time constants and command limits must be positive")
        if self.mechanism_min >= self.mechanism_max or not self.mechanism_units:
            raise ValueError("mechanism limits/units are required")
        if abs(self.dt_s * 1e9 - round(self.dt_s * 1e9)) > 1e-6:
            raise ValueError("dt must be exactly representable in integer nanoseconds")

    @property
    def dt_ns(self) -> int:
        return round(self.dt_s * 1e9)


@dataclass(frozen=True)
class Rollout:
    states: Tuple[EgoMechanismState, ...]
    times_ns: Tuple[int, ...]
    saturated: Tuple[Tuple[bool, ...], ...]
    kind: str = "analytical-forecast"
    measured: bool = False


def _approach(value: float, target: float, tau: float, dt: float) -> Tuple[float, float]:
    decay = math.exp(-dt / tau)
    return target + (value - target) * decay, target * dt + (value - target) * tau * (1.0 - decay)


class AnalyticalDynamics:
    """Configurable closed-loop response baseline, not calibrated by default.

    Velocity response uses exact first-order integrals. Body displacement is
    rotated at interval midpoint heading, so combined rotation/translation is
    a numerical approximation. Parameter fitting must use training sessions
    only and record provenance; these defaults establish plumbing only.
    """
    def __init__(self, config: Optional[DynamicsConfig] = None):
        self.config = config or DynamicsConfig()

    def clipped(self, command: PhysicalCommand) -> Tuple[Tuple[float, ...], Tuple[bool, ...]]:
        cfg = self.config
        bounds = ((-cfg.max_vx_mps, cfg.max_vx_mps), (-cfg.max_vy_mps, cfg.max_vy_mps),
                  (-cfg.max_omega_radps, cfg.max_omega_radps),
                  (cfg.mechanism_min, cfg.mechanism_max))
        raw = command.vector()
        clipped = tuple(max(low, min(high, x)) for x, (low, high) in zip(raw, bounds))
        return clipped, tuple(a != b for a, b in zip(raw, clipped))

    def step(self, state: EgoMechanismState, command: PhysicalCommand) -> Tuple[EgoMechanismState, Tuple[bool, ...]]:
        cfg = self.config
        target, saturated = self.clipped(command)
        vx, dx_body = _approach(state.vx_robot_mps, target[0], cfg.chassis_tau_s, cfg.dt_s)
        vy, dy_body = _approach(state.vy_robot_mps, target[1], cfg.chassis_tau_s, cfg.dt_s)
        omega, dtheta = _approach(state.omega_radps, target[2], cfg.heading_tau_s, cfg.dt_s)
        midpoint = state.heading_rad + dtheta / 2.0
        dx = math.cos(midpoint) * dx_body - math.sin(midpoint) * dy_body
        dy = math.sin(midpoint) * dx_body + math.cos(midpoint) * dy_body
        mechanism_target = state.mechanism_position if command.mechanism_mode == "hold" else target[3]
        mechanism, _ = _approach(state.mechanism_position, mechanism_target,
                                 cfg.mechanism_tau_s, cfg.dt_s)
        next_state = EgoMechanismState(state.x_m + dx, state.y_m + dy,
            wrap_angle(state.heading_rad + dtheta), vx, vy, omega, mechanism,
            (mechanism - state.mechanism_position) / cfg.dt_s)
        return next_state, saturated

    def rollout(self, initial: EgoMechanismState, commands: Iterable[PhysicalCommand],
                start_ns: Optional[int] = None) -> Rollout:
        commands = tuple(commands)
        if not commands:
            raise ValueError("rollout requires at least one command")
        start = commands[0].timestamp_ns if start_ns is None else start_ns
        states: List[EgoMechanismState] = []
        saturation: List[Tuple[bool, ...]] = []
        state = initial
        for index, command in enumerate(commands):
            if command.timestamp_ns != start + index * self.config.dt_ns:
                raise ValueError("command gaps, duplicates and out-of-order intervals are not allowed")
            state, clipped = self.step(state, command)
            states.append(state)
            saturation.append(clipped)
        return Rollout(tuple(states), tuple(start + (i + 1) * self.config.dt_ns
                                            for i in range(len(states))), tuple(saturation))


def persistence(initial: EgoMechanismState, horizon: int) -> Tuple[EgoMechanismState, ...]:
    if horizon <= 0:
        raise ValueError("positive horizon required")
    return (initial,) * horizon


def constant_velocity(initial: EgoMechanismState, horizon: int, dt_s: float) -> Tuple[EgoMechanismState, ...]:
    """Open-loop kinematic baseline with constant body velocity and turn rate."""
    if horizon <= 0 or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("positive finite timestep/horizon required")
    result = []
    state = initial
    for _ in range(horizon):
        theta = state.heading_rad
        omega = state.omega_radps
        dtheta = omega * dt_s
        if abs(omega) < 1e-12:
            dx_body, dy_body = state.vx_robot_mps * dt_s, state.vy_robot_mps * dt_s
        else:
            sin, cos = math.sin(dtheta), math.cos(dtheta)
            dx_body = (state.vx_robot_mps * sin + state.vy_robot_mps * (cos - 1.0)) / omega
            dy_body = (state.vx_robot_mps * (1.0 - cos) + state.vy_robot_mps * sin) / omega
        state = EgoMechanismState(state.x_m + math.cos(theta) * dx_body - math.sin(theta) * dy_body,
            state.y_m + math.sin(theta) * dx_body + math.cos(theta) * dy_body,
            wrap_angle(theta + dtheta), state.vx_robot_mps, state.vy_robot_mps, omega,
            state.mechanism_position + state.mechanism_velocity * dt_s, state.mechanism_velocity)
        result.append(state)
    return tuple(result)
