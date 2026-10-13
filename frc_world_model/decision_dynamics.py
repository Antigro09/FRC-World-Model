"""Bounded advisory kinematics and geometric screening, using only stdlib.

Inputs are already validated ``frc-prediction/1`` requests. The caller bounds
candidate counts, time grids, field domain, and obstacle counts. These helpers
do not estimate robot pose, change tracks, infer possession, or issue commands.
Their open-loop motion and error-growth envelopes are explicit assumptions;
they are neither a physical dynamics model nor a collision-safety certificate.
"""

import math

from .bindings import ContractError


KINEMATIC_ASSUMPTION = "KINEMATIC_ASSUMPTION"


def _number(value, label, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be finite numeric")
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ContractError(f"{label} must be finite and nonnegative" if nonnegative else f"{label} must be finite")
    return value


def _latest_ego(request):
    ego = request["history"][-1]["snapshot"]["ego"]
    if ego["time_us"] != request["history_cutoff_us"]:
        raise ContractError("kinematic anchor must equal history cutoff")
    return ego


def _field_velocity(heading, command):
    cosine, sine = math.cos(heading), math.sin(heading)
    vx, vy = command["vx_mps"], command["vy_mps"]
    return cosine * vx - sine * vy, sine * vx + cosine * vy


def _advance(x, y, heading, command, dt):
    """Integrate one constant body twist exactly, including lateral motion."""
    angle = command["omega_radps"] * dt
    if abs(angle) < 1e-4:
        angle2 = angle * angle
        sinc = 1 - angle2 / 6 + angle2 * angle2 / 120
        cosc = angle * (0.5 - angle2 / 24 + angle2 * angle2 / 720)
    else:
        sinc = math.sin(angle) / angle
        cosc = 2 * math.sin(angle / 2) ** 2 / angle
    dx_body = dt * (command["vx_mps"] * sinc - command["vy_mps"] * cosc)
    dy_body = dt * (command["vx_mps"] * cosc + command["vy_mps"] * sinc)
    cosine, sine = math.cos(heading), math.sin(heading)
    x += cosine * dx_body - sine * dy_body
    y += sine * dx_body + cosine * dy_body
    heading = math.atan2(math.sin(heading + angle), math.cos(heading + angle))
    return x, y, heading


def _sample(offset_us, x, y, heading, command):
    vx, vy = _field_velocity(heading, command)
    return {"offset_us": offset_us, "x_m": x, "y_m": y,
            "heading_rad": heading, "vx_mps": vx, "vy_mps": vy,
            "omega_radps": command["omega_radps"],
            "kind": KINEMATIC_ASSUMPTION, "physical_covariance": None}


def kinematic_forecasts(prediction_request):
    """Return independent ego forecasts under instantaneous command tracking.

    Positions at offset zero are the observed cutoff pose. Velocity there is
    the *counterfactual* derivative of the first candidate command, not a new
    measured velocity. Each command applies over the interval beginning at its
    sample offset. Only candidate ``samples[:-1]`` are integrated: the final
    endpoint command has no interval and is not used. Derivatives are outgoing
    at each interior sample; the final sample repeats the last applied command.
    All times remain anchored to ``history_cutoff_us`` without arrival rebasing.
    Mechanism commands cannot be modeled by this kinematic helper.
    """
    ego = _latest_ego(prediction_request)
    pose = ego["field_pose"]
    result = {}
    for candidate in prediction_request["candidates"]:
        commands = candidate["samples"]
        x, y = pose["position_m"]["x"], pose["position_m"]["y"]
        heading = pose["heading_rad"]
        forecast = [_sample(commands[0]["offset_us"], x, y, heading, commands[0]["body_twist"])]
        for index, command_sample in enumerate(commands[:-1]):
            end_offset = commands[index + 1]["offset_us"]
            dt = (end_offset - command_sample["offset_us"]) / 1_000_000
            x, y, heading = _advance(x, y, heading, command_sample["body_twist"], dt)
            derivative = (commands[index + 1]["body_twist"] if index + 1 < len(commands) - 1
                          else command_sample["body_twist"])
            forecast.append(_sample(end_offset, x, y, heading, derivative))
        result[candidate["candidate_id"]] = forecast
    return result


def extrapolated_track_position(track, absolute_us):
    """Read-only constant-velocity assumption from a track's estimate time.

    The caller qualifies track age, lifecycle, confidence, and uncertainty. This
    helper does not upgrade an extrapolation to an observed object position.
    """
    if type(absolute_us) is not int or absolute_us < track["estimate_us"]:
        raise ContractError("track extrapolation requires time at or after estimate")
    dt = (absolute_us - track["estimate_us"]) / 1_000_000
    return (track["position_m"]["x"] + track["velocity_mps"]["x"] * dt,
            track["position_m"]["y"] + track["velocity_mps"]["y"] * dt)


def _covariance_radius(covariance, sigma):
    xx = _number(covariance["xx_m2"], "covariance xx", nonnegative=True)
    xy = _number(covariance["xy_m2"], "covariance xy")
    yy = _number(covariance["yy_m2"], "covariance yy", nonnegative=True)
    if xy * xy > xx * yy + 1e-12:
        raise ContractError("screening covariance must be PSD")
    eigenvalue = (xx + yy + math.hypot(xx - yy, 2 * xy)) / 2
    radius = sigma * math.sqrt(max(0, eigenvalue))
    if not math.isfinite(radius):
        raise ContractError("screening covariance radius overflow")
    return radius


def _distance_to_segment_origin(start_x, start_y, end_x, end_y):
    if not all(math.isfinite(value) for value in (start_x, start_y, end_x, end_y)):
        raise ContractError("relative screening position overflow")
    dx, dy = end_x - start_x, end_y - start_y
    length = math.hypot(dx, dy)
    if not math.isfinite(length):
        raise ContractError("relative screening segment overflow")
    if length == 0:
        return math.hypot(start_x, start_y)
    ux, uy = dx / length, dy / length
    along = min(length, max(0.0, -(start_x * ux + start_y * uy)))
    return math.hypot(start_x + along * ux, start_y + along * uy)


def _curvature_padding(start, end, dt):
    if start.get("kind") == KINEMATIC_ASSUMPTION and "omega_radps" in start:
        # ||r''|| <= speed*|omega|, so deviation from a chord is <= a*dt²/8.
        # The bound remains conservative for arcs longer than a full rotation.
        return math.hypot(start["vx_mps"], start["vy_mps"]) * abs(start["omega_radps"]) * dt * dt / 8
    # Qualified learned point uncertainty alone cannot establish continuous
    # motion between samples. This explicit half-interval sampled-speed padding
    # is an advisory screening assumption, not a certified physical speed bound.
    speed = max(math.hypot(start["vx_mps"], start["vy_mps"]),
                math.hypot(end["vx_mps"], end["vy_mps"]))
    return speed * dt / 2


def screen_forecast(samples, prediction_request, obstacles, *, footprint_radius_m,
                    clearance_margin_m, uncertainty_sigma, ego_error_growth_mps,
                    obstacle_error_growth_mps):
    """Conservatively screen swept ego circles against moving obstacle circles.

    Obstacle centers are aligned by the caller to ``history_cutoff_us`` and move
    at their supplied field velocity. Only the explicit obstacle list is used;
    observed tracks are never implicitly treated as obstacles or modified.
    Every physical envelope is mandatory configuration, with no defaults.

    Observed ego covariance contributes sigma*sqrt(largest eigenvalue). A sample
    with ``physical_covariance`` uses the larger observed/forecast radius; an
    unqualified sample adds configured ego error growth to the observed radius.
    Obstacle error growth is additional unqualified padding. Each swept segment
    uses the maximum endpoint envelope and a curvature/interpolation bound.

    ``minimum_clearance_m`` is a conservative signed lower bound after all these
    paddings, or None when there are no explicit obstacles. Touching is blocked.
    ``uncertainty_radius_m`` reports the maximum ego-plus-obstacle envelope used
    (excluding footprint, clearance margin, and curvature padding). No field
    containment is checked here; the caller checks its pinned field domain.
    """
    footprint = _number(footprint_radius_m, "footprint radius", nonnegative=True)
    margin = _number(clearance_margin_m, "clearance margin", nonnegative=True)
    sigma = _number(uncertainty_sigma, "uncertainty sigma", nonnegative=True)
    ego_growth = _number(ego_error_growth_mps, "ego error growth", nonnegative=True)
    obstacle_growth = _number(obstacle_error_growth_mps, "obstacle error growth", nonnegative=True)
    observed_radius = _covariance_radius(_latest_ego(prediction_request)["uncertainty"], sigma)
    if not samples:
        raise ContractError("screening requires forecast samples")
    radii = []
    previous = -1
    for sample in samples:
        offset = sample["offset_us"]
        if type(offset) is not int or not previous < offset <= prediction_request["horizon_us"]:
            raise ContractError("screening sample time order or horizon")
        previous = offset
        for key in ("x_m", "y_m", "vx_mps", "vy_mps"):
            _number(sample[key], f"sample {key}")
        if "omega_radps" in sample:
            _number(sample["omega_radps"], "sample omega")
        physical = sample.get("physical_covariance")
        radius = (max(observed_radius, _covariance_radius(physical, sigma)) if physical is not None
                  else observed_radius + ego_growth * offset / 1_000_000)
        if obstacles:
            radius += obstacle_growth * offset / 1_000_000
        if not math.isfinite(radius):
            raise ContractError("screening error envelope overflow")
        radii.append(radius)
    prepared_obstacles = []
    for obstacle in obstacles:
        center, velocity = obstacle["center_m"], obstacle["velocity_mps"]
        prepared_obstacles.append((_number(center["x"], "obstacle x"), _number(center["y"], "obstacle y"),
                                   _number(velocity["x"], "obstacle vx"), _number(velocity["y"], "obstacle vy"),
                                   _number(obstacle["radius_m"], "obstacle radius", nonnegative=True)))
    minimum = None
    for cx, cy, ovx, ovy, obstacle_radius in prepared_obstacles:
        # Also covers a single supplied point, or the anchor before any motion.
        for sample, radius in zip(samples, radii):
            t = sample["offset_us"] / 1_000_000
            distance = math.hypot(sample["x_m"] - cx - ovx * t, sample["y_m"] - cy - ovy * t)
            clearance = distance - footprint - margin - obstacle_radius - radius
            if not math.isfinite(clearance):
                raise ContractError("screening clearance overflow")
            minimum = clearance if minimum is None else min(minimum, clearance)
        for index, (start, end) in enumerate(zip(samples, samples[1:])):
            t0, t1 = start["offset_us"] / 1_000_000, end["offset_us"] / 1_000_000
            distance = _distance_to_segment_origin(start["x_m"] - cx - ovx * t0,
                                                   start["y_m"] - cy - ovy * t0,
                                                   end["x_m"] - cx - ovx * t1,
                                                   end["y_m"] - cy - ovy * t1)
            padding = max(radii[index], radii[index + 1]) + _curvature_padding(start, end, t1 - t0)
            clearance = distance - footprint - margin - obstacle_radius - padding
            if not math.isfinite(clearance):
                raise ContractError("screening swept clearance overflow")
            minimum = clearance if minimum is None else min(minimum, clearance)
    return {"minimum_clearance_m": minimum, "blocked": minimum is not None and minimum <= 0,
            "uncertainty_radius_m": max(radii)}
