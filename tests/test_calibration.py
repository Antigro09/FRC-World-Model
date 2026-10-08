import copy
from types import SimpleNamespace

import pytest
from frc_world_model.calibration import freeze_thresholds,validate_thresholds,assess_thresholds
from frc_world_model.data import canonical_hash


def package():
    summary={"count":1,"mae":.1,"rmse":.1,"p50":.1,"p95":.1,"p99":.1,"max":.1}
    features={name:summary for name in ("position_error_m","heading_error_rad","velocity_error_mps","omega_error_radps")}
    report={"successful_windows":1,"per_session":{"calibration-session":{"20000000":features}},
            "per_horizon_ns":{"20000000":features},"missing_predictions":{},"name":"rwm","mode":"open_loop"}
    report["evaluation_hash"]=canonical_hash(report)
    return SimpleNamespace(manifest={"model_id":"synthetic-only"},metadata={
        "training":{"evaluation":{"calibration":{"rwm":report}}},"dataset":{"ingestion":{"dataset_hash":"data"},"split":{"calibration":["session"]},"configuration":{"synthetic_fixture":True}},
        "normalization":{"state":{"layout":{"names":["s"]}},"action":{"layout":{"names":["a"]}}}})


def tolerances():
    return {n:.2 for n in ("position_error_m","heading_error_rad","velocity_error_mps","omega_error_radps")}


def test_frozen_thresholds_bind_results_model_dataset_layout_horizon_and_are_used():
    p=package(); policy=freeze_thresholds(p,tolerances())
    assert validate_thresholds(policy,p)==policy
    report=p.metadata["training"]["evaluation"]["calibration"]["rwm"]
    assessed=assess_thresholds(report,policy)
    assert assessed["per_horizon_ns"]["20000000"]["position_error_m"]["all_observed_within_tolerance"]
    bad=copy.deepcopy(p);bad.manifest["model_id"]="different"
    with pytest.raises(ValueError):validate_thresholds(policy,bad)
    bad=copy.deepcopy(p);bad.metadata["dataset"]["ingestion"]["dataset_hash"]="changed"
    with pytest.raises(ValueError):validate_thresholds(policy,bad)
    changed=copy.deepcopy(policy);changed["bounds"]["20000000"]["position_error_m"]=100.
    changed["calibration_hash"]=canonical_hash({k:v for k,v in changed.items() if k!="calibration_hash"})
    with pytest.raises(ValueError):validate_thresholds(changed,p)
    report["per_horizon_ns"]["20000000"]["position_error_m"]={**report["per_horizon_ns"]["20000000"]["position_error_m"],"max":100.}
    with pytest.raises(ValueError,match="checksum"):freeze_thresholds(p,tolerances())


@pytest.mark.parametrize("value",[-1.,0.,float("nan"),float("inf"),True])
def test_arbitrary_checksums_negative_unknown_or_missing_tolerances_rejected(value):
    p=package();bad=tolerances();bad["position_error_m"]=value
    with pytest.raises(ValueError):freeze_thresholds(p,bad)
    with pytest.raises(ValueError):validate_thresholds({"fit_split":"calibration","calibration_hash":"arbitrary","physical_tolerances":bad},p)
    with pytest.raises(ValueError):freeze_thresholds(p,{"position_error_m":.1})
