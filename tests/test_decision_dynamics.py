import copy
import json
import math

import pytest

from frc_world_model.bindings import CONTRACT, ContractError, validate_request
from frc_world_model.decision_dynamics import (
    extrapolated_track_position, kinematic_forecasts, screen_forecast,
)


def request(commands, *, step_us=1_000_000, heading=0.0, covariance=None):
    raw = json.loads((CONTRACT / "request-v1.json").read_text())
    raw["step_us"] = step_us
    raw["horizon_us"] = step_us * len(commands)
    ego = raw["history"][-1]["snapshot"]["ego"]
    ego["field_pose"] = {"position_m": {"x": 0.0, "y": 0.0}, "heading_rad": heading}
    ego["uncertainty"] = covariance or {"xx_m2": 0.0, "xy_m2": 0.0, "yy_m2": 0.0}
    samples = [{"offset_us": index * step_us,
                "body_twist": {"vx_mps": vx, "vy_mps": vy, "omega_radps": omega},
                "mechanisms": []}
               for index, (vx, vy, omega) in enumerate(commands)]
    # Endpoint commands have no duration and must not change the forecast.
    samples.append({"offset_us": raw["horizon_us"],
                    "body_twist": {"vx_mps": 999.0, "vy_mps": 999.0, "omega_radps": 999.0},
                    "mechanisms": []})
    raw["candidates"] = [{"candidate_id": "test", "authority": "CANDIDATE",
                          "semantics": "BODY_TWIST_OPEN_LOOP_V1", "frame": "ROBOT_RELATIVE",
                          "samples": samples}]
    return validate_request(raw, now_us=1_030_000, max_sync_error_us=1000)


def obstacle(x, y, *, vx=0.0, vy=0.0, radius=0.1):
    return {"obstacle_id": "explicit-obstacle", "center_m": {"x": x, "y": y},
            "velocity_mps": {"x": vx, "y": vy}, "radius_m": radius}


def screen(samples, raw, obstacles, **overrides):
    parameters = {"footprint_radius_m": 0.1, "clearance_margin_m": 0.0,
                  "uncertainty_sigma": 1.0, "ego_error_growth_mps": 0.0,
                  "obstacle_error_growth_mps": 0.0}
    parameters.update(overrides)
    return screen_forecast(samples, raw, obstacles, **parameters)


def test_heading_rotates_robot_relative_translation_to_field_frame():
    raw = request([(1.0, 0.0, 0.0)], heading=math.pi / 2)
    forecast = kinematic_forecasts(raw)["test"]
    assert forecast[-1]["x_m"] == pytest.approx(0.0, abs=1e-14)
    assert forecast[-1]["y_m"] == pytest.approx(1.0)
    assert forecast[0]["vx_mps"] == pytest.approx(0.0, abs=1e-14)
    assert forecast[0]["vy_mps"] == pytest.approx(1.0)
    assert forecast[-1]["omega_radps"] == 0.0
    assert all(s["kind"] == "KINEMATIC_ASSUMPTION" and s["physical_covariance"] is None
               and "confidence" not in s for s in forecast)


def test_turning_twist_integrates_exact_circular_arc_and_lateral_component():
    raw = request([(1.0, 0.5, math.pi / 2)])
    end = kinematic_forecasts(raw)["test"][-1]
    assert end["x_m"] == pytest.approx(1.0 / math.pi)
    assert end["y_m"] == pytest.approx(3.0 / math.pi)
    assert end["heading_rad"] == pytest.approx(math.pi / 2)
    assert (end["vx_mps"], end["vy_mps"]) == pytest.approx((-0.5, 1.0))


def test_small_angular_rate_is_continuous_and_heading_is_wrapped():
    raw = request([(1.0, 0.0, 1e-12)])
    end = kinematic_forecasts(raw)["test"][-1]
    assert (end["x_m"], end["y_m"]) == pytest.approx((1.0, 5e-13), abs=1e-15)
    raw = request([(0.0, 0.0, 1.0)], heading=3.0)
    end = kinematic_forecasts(raw)["test"][-1]
    assert end["heading_rad"] == pytest.approx(4.0 - 2 * math.pi)


def test_commands_are_interval_start_actions_and_endpoint_command_is_not_used():
    raw = request([(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)])
    forecast = kinematic_forecasts(raw)["test"]
    assert [(s["offset_us"], s["x_m"], s["y_m"]) for s in forecast] == [
        (0, 0.0, 0.0), (1_000_000, 1.0, 0.0), (2_000_000, 1.0, 2.0)]
    assert (forecast[1]["vx_mps"], forecast[1]["vy_mps"]) == (0.0, 2.0)
    assert (forecast[-1]["vx_mps"], forecast[-1]["vy_mps"]) == (0.0, 2.0)


def test_candidate_forecasts_are_isolated_and_observed_inputs_are_immutable():
    raw = request([(1.0, 0.0, 0.0)])
    hold = copy.deepcopy(raw["candidates"][0])
    hold["candidate_id"] = "hold"
    hold["samples"][0]["body_twist"]["vx_mps"] = 0.0
    raw["candidates"].append(hold)
    original = copy.deepcopy(raw)
    forecasts = kinematic_forecasts(raw)
    assert forecasts["hold"][-1]["x_m"] == 0.0
    assert forecasts["test"][-1]["x_m"] == 1.0
    assert forecasts["test"][0] is not forecasts["hold"][0]
    forecasts["test"][0]["x_m"] = -99.0
    assert forecasts["hold"][0]["x_m"] == 0.0
    assert raw == original


def test_does_not_rebase_an_older_observation_to_cutoff():
    raw = request([(0.0, 0.0, 0.0)])
    raw["history_cutoff_us"] += 1
    with pytest.raises(ContractError, match="anchor"):
        kinematic_forecasts(raw)


def test_swept_static_collision_detects_tunneling_between_safe_endpoints():
    raw = request([(2.0, 0.0, 0.0)])
    forecast = kinematic_forecasts(raw)["test"]
    result = screen(forecast, raw, [obstacle(1.0, 0.0)])
    assert result["blocked"] is True
    assert result["minimum_clearance_m"] == pytest.approx(-0.2)
    assert all(math.hypot(s["x_m"] - 1.0, s["y_m"]) > 0.2 for s in forecast)


def test_relative_sweep_detects_moving_obstacle_crossing_stationary_ego():
    raw = request([(0.0, 0.0, 0.0)])
    forecast = kinematic_forecasts(raw)["test"]
    moving = obstacle(0.0, -1.0, vy=2.0)
    before = copy.deepcopy(moving)
    result = screen(forecast, raw, [moving])
    assert result["blocked"] is True
    assert result["minimum_clearance_m"] == pytest.approx(-0.2)
    assert moving == before
    assert screen(forecast, raw, [obstacle(0.0, -1.0)])["blocked"] is False


def test_curvature_padding_catches_collision_on_arc_away_from_chord():
    raw = request([(math.pi, 0.0, math.pi)])
    forecast = kinematic_forecasts(raw)["test"]
    # Arc is a unit semicircle through (1, 1); endpoint chord lies at x=0.
    result = screen(forecast, raw, [obstacle(1.0, 1.0, radius=0.01)], footprint_radius_m=0.01)
    assert forecast[-1]["x_m"] == pytest.approx(0.0, abs=1e-14)
    assert forecast[-1]["y_m"] == pytest.approx(2.0)
    assert result["blocked"] is True
    assert result["minimum_clearance_m"] < 0.0


def test_observed_covariance_uses_largest_eigenvalue_and_explicit_growth():
    raw = request([(0.0, 0.0, 0.0)], covariance={"xx_m2": 0.04, "xy_m2": 0.03, "yy_m2": 0.04})
    forecast = kinematic_forecasts(raw)["test"]
    result = screen(forecast, raw, [obstacle(2.0, 0.0)], uncertainty_sigma=2.0,
                    ego_error_growth_mps=0.1, obstacle_error_growth_mps=0.2)
    radius = 2.0 * math.sqrt(0.07) + 0.3
    assert result["uncertainty_radius_m"] == pytest.approx(radius)
    assert result["minimum_clearance_m"] == pytest.approx(2.0 - 0.2 - radius)


def test_qualified_sample_covariance_has_no_invented_ego_growth_or_confidence():
    raw = request([(0.0, 0.0, 0.0)], covariance={"xx_m2": 0.04, "xy_m2": 0.0, "yy_m2": 0.04})
    forecast = kinematic_forecasts(raw)["test"]
    for sample in forecast:
        sample.pop("heading_rad")
        sample.pop("omega_radps")
        sample["kind"] = "LEARNED_FORECAST"
        sample["physical_covariance"] = {"xx_m2": 0.01, "xy_m2": 0.0, "yy_m2": 0.01}
    forecast[-1]["physical_covariance"]["xx_m2"] = 0.25
    result = screen(forecast, raw, [obstacle(2.0, 0.0)], ego_error_growth_mps=10.0)
    assert result["uncertainty_radius_m"] == pytest.approx(0.5)
    assert result["minimum_clearance_m"] == pytest.approx(1.3)


def test_learned_points_get_explicit_intersample_padding_without_omega():
    raw = request([(1.0, 0.0, 0.0)])
    forecast = kinematic_forecasts(raw)["test"]
    for sample in forecast:
        sample.pop("omega_radps")
        sample.pop("heading_rad")
        sample["kind"] = "LEARNED_FORECAST"
        sample["physical_covariance"] = {"xx_m2": 0.0, "xy_m2": 0.0, "yy_m2": 0.0}
    result = screen(forecast, raw, [obstacle(0.5, 0.6)])
    assert result["blocked"] is True
    assert result["minimum_clearance_m"] == pytest.approx(-0.1)


def test_no_explicit_obstacles_does_not_turn_tracks_into_obstacles():
    raw = request([(0.0, 0.0, 0.0)])
    raw["history"][-1]["snapshot"]["tracks"][0]["position_m"] = {"x": 0.0, "y": 0.0}
    original = copy.deepcopy(raw)
    result = screen(kinematic_forecasts(raw)["test"], raw, [], ego_error_growth_mps=0.1,
                    obstacle_error_growth_mps=99.0)
    assert result == {"minimum_clearance_m": None, "blocked": False, "uncertainty_radius_m": 0.1}
    assert raw == original


@pytest.mark.parametrize("parameter,value", [
    ("footprint_radius_m", -0.1), ("clearance_margin_m", float("nan")),
    ("uncertainty_sigma", float("inf")), ("ego_error_growth_mps", -0.1),
    ("obstacle_error_growth_mps", True),
])
def test_invalid_physical_envelope_rejected(parameter, value):
    raw = request([(0.0, 0.0, 0.0)])
    with pytest.raises(ContractError):
        screen(kinematic_forecasts(raw)["test"], raw, [], **{parameter: value})


def test_empty_reordered_nonfinite_and_invalid_covariance_forecasts_rejected():
    raw = request([(0.0, 0.0, 0.0)])
    forecasts = kinematic_forecasts(raw)["test"]
    bad_nan = copy.deepcopy(forecasts)
    bad_nan[-1]["x_m"] = float("nan")
    bad_covariance = copy.deepcopy(forecasts)
    bad_covariance[-1]["physical_covariance"] = {"xx_m2": 0.01, "xy_m2": 1.0, "yy_m2": 0.01}
    for samples in ([], list(reversed(forecasts)), bad_nan, bad_covariance):
        with pytest.raises(ContractError):
            screen(samples, raw, [])


def test_track_position_is_anchored_to_estimate_time_and_never_mutates_track():
    raw = request([(0.0, 0.0, 0.0)])
    track = raw["history"][-1]["snapshot"]["tracks"][0]
    track["velocity_mps"] = {"x": 2.0, "y": -1.0}
    before = copy.deepcopy(track)
    assert extrapolated_track_position(track, 1_500_000) == (4.0, 1.5)
    assert track == before
    with pytest.raises(ContractError):
        extrapolated_track_position(track, 999_999)
