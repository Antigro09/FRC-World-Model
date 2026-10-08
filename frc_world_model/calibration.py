"""Checksummed pre-test thresholds, bound to frozen model and held-out report."""
import math

from .data import canonical_hash
from .evaluation import calibrate_session_bounds

REQUIRED_TOLERANCES=("position_error_m","heading_error_rad","velocity_error_mps","omega_error_radps")


def _reference(package):
    report=package.metadata["training"].get("evaluation",{}).get("calibration",{}).get("rwm")
    if not report or not report.get("per_session") or not report.get("per_horizon_ns") or report.get("successful_windows",0)<=0:
        raise ValueError("nonempty frozen held-out calibration report required")
    if report["evaluation_hash"]!=canonical_hash({k:v for k,v in report.items() if k!="evaluation_hash"}):
        raise ValueError("calibration evaluation checksum mismatch")
    dataset=package.metadata["dataset"]
    provenance={"model_id":package.manifest["model_id"],"dataset_hash":dataset["ingestion"]["dataset_hash"],
        "configuration_hash":canonical_hash(dataset["configuration"]),"split_hash":canonical_hash(dataset["split"]),
        "state_layout_hash":canonical_hash(package.metadata["normalization"]["state"]["layout"]),
        "action_layout_hash":canonical_hash(package.metadata["normalization"]["action"]["layout"]),
        "calibration_evaluation_hash":report["evaluation_hash"],"calibration_sessions":sorted(report["per_session"]),
        "horizons_ns":sorted(report["per_horizon_ns"],key=int)}
    return report,provenance


def freeze_thresholds(package,physical_tolerances,*,quantile=.95):
    report,provenance=_reference(package)
    if any(key not in physical_tolerances for key in REQUIRED_TOLERANCES):
        raise ValueError("footprint/stopping/delay/heading/velocity tolerances required")
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in physical_tolerances.values()):
        raise ValueError("physical tolerances must be finite and positive")
    metrics=set.intersection(*(set(features) for features in report["per_horizon_ns"].values()))
    if not set(physical_tolerances).issubset(metrics): raise ValueError("tolerance names not in calibrated metrics")
    result=calibrate_session_bounds(report,split="calibration",quantile=quantile,physical_tolerances=physical_tolerances)
    result.pop("calibration_hash")
    result["provenance"]=provenance
    result["calibration_hash"]=canonical_hash(result)
    return result


def validate_thresholds(thresholds,package):
    if thresholds.get("calibration_hash")!=canonical_hash({k:v for k,v in thresholds.items() if k!="calibration_hash"}):
        raise ValueError("threshold artifact checksum mismatch")
    expected=freeze_thresholds(package,thresholds.get("physical_tolerances",{}),quantile=thresholds.get("quantile",0))
    if thresholds!=expected:
        raise ValueError("threshold model/dataset/layout/split/session/horizon or bound mismatch")
    return thresholds


def assess_thresholds(report,thresholds):
    """Describe observed session maxima; no statistical safety/coverage claim."""
    result={}
    for horizon,features in report["per_horizon_ns"].items():
        result[horizon]={}
        for name,tolerance in thresholds["physical_tolerances"].items():
            if name not in features or horizon not in thresholds["bounds"]:
                raise ValueError("final test metric/horizon lacks frozen threshold")
            maxima=[v[horizon][name]["max"] for v in report["per_session"].values() if horizon in v]
            bound=thresholds["bounds"][horizon][name]
            result[horizon][name]={"physical_tolerance":tolerance,"observed_max":features[name]["max"],
                "all_observed_within_tolerance":features[name]["max"]<=tolerance,
                "frozen_session_max_bound":bound,"session_count":len(maxima),
                "sessions_within_frozen_bound":sum(v<=bound for v in maxima)}
    return {"advisory_only":True,"coverage_claim":"none; descriptive held-out session maxima",
            "threshold_artifact_sha256":thresholds["calibration_hash"],"per_horizon_ns":result,
            "missing_predictions":report["missing_predictions"]}
