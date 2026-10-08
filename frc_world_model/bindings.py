"""Exact World-State frc-prediction/1 JSON binding, off the periodic path."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

CONTRACT = Path(__file__).with_name("_contracts") / "world-state-v1"
IDENTITIES = ("schema_version","model_id","request_id","snapshot_id","epoch","field","source","frame","timestamp_domain","history_cutoff_us")


class ContractError(ValueError):
    pass


def _finite(value):
    if isinstance(value,float) and not math.isfinite(value):
        raise ContractError("nonfinite wire value")
    if isinstance(value,dict):
        for child in value.values(): _finite(child)
    if isinstance(value,list):
        for child in value: _finite(child)


def _pairs(pairs):
    result = {}
    for key,value in pairs:
        if key in result: raise ContractError("duplicate JSON key")
        result[key] = value
    return result


def load_document(payload, kind, *, max_bytes=1_000_000):
    if kind not in ("request","reply") or len(payload.encode()) > max_bytes:
        raise ContractError("unsupported or oversized document")
    try:
        raw = json.loads(payload,object_pairs_hook=_pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(ContractError("nonfinite JSON constant")))
        _finite(raw)
        from jsonschema import Draft202012Validator
        schema = json.loads((CONTRACT / f"prediction-{kind}-v1.schema.json").read_text())
        Draft202012Validator(schema).validate(raw)
    except (ValueError,TypeError,KeyError) as error:
        raise ContractError(str(error)) from error
    except Exception as error:
        # Validation errors never expose a traceback or become a forecast.
        raise ContractError("schema validation: " + type(error).__name__) from error
    return raw


@dataclass(frozen=True)
class WirePolicy:
    """All physical/domain values are supplied; there is no production default."""
    model_calibrations: Mapping[str,tuple[str,...]]
    allowed_fields: tuple[tuple[str,str,str],...]
    allowed_configuration_revisions: tuple[str,...]
    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float
    max_speed_mps: float
    max_omega_radps: float
    min_variance_m2: float
    max_variance_m2: float
    max_initial_offset_m: float
    max_horizon_us: int
    max_history_age_us: int
    max_sync_error_us: int
    mechanisms: Mapping[str,tuple[str,float,float]]

    def __post_init__(self):
        numbers = (self.min_x_m,self.min_y_m,self.max_x_m,self.max_y_m,self.max_speed_mps,self.max_omega_radps,
                   self.min_variance_m2,self.max_variance_m2,self.max_initial_offset_m)
        if not all(math.isfinite(x) for x in numbers) or self.min_x_m>=self.max_x_m or self.min_y_m>=self.max_y_m:
            raise ContractError("invalid physical policy")
        if min(self.max_speed_mps,self.max_omega_radps,self.min_variance_m2,self.max_variance_m2)<=0 or self.min_variance_m2>self.max_variance_m2 or self.max_initial_offset_m<0:
            raise ContractError("invalid positive physical bounds")
        if not self.model_calibrations or not self.allowed_fields or not self.allowed_configuration_revisions:
            raise ContractError("missing approved model/field/configuration domain")
        if any(type(t) is not int or t<0 for t in (self.max_horizon_us,self.max_history_age_us,self.max_sync_error_us)) or min(self.max_horizon_us,self.max_history_age_us)==0:
            raise ContractError("invalid microsecond policy bounds")

    def inside(self,x,y):
        return self.min_x_m<=x<=self.max_x_m and self.min_y_m<=y<=self.max_y_m


def _covariance(xx,xy,yy,policy=None):
    if xx<0 or yy<0 or xy*xy>xx*yy+1e-12:
        raise ContractError("covariance is not PSD")
    if policy:
        spread = math.hypot(xx-yy,2*xy)
        if (xx+yy-spread)/2<policy.min_variance_m2 or (xx+yy+spread)/2>policy.max_variance_m2:
            raise ContractError("covariance outside calibrated bounds")


def _action(twist, mechanisms, policy):
    if policy and (math.hypot(twist["vx_mps"],twist["vy_mps"])>policy.max_speed_mps or abs(twist["omega_radps"])>policy.max_omega_radps):
        raise ContractError("command outside physical domain")
    seen = set()
    for item in mechanisms:
        if item["channel"] in seen or (item["unit"]=="bool" and item["value"] not in (0,1)):
            raise ContractError("invalid mechanism channel/bool value")
        seen.add(item["channel"])
        if policy:
            bounds = policy.mechanisms.get(item["channel"])
            if not bounds or bounds[0]!=item["unit"] or not bounds[1]<=item["value"]<=bounds[2]:
                raise ContractError("mechanism outside configured domain")


def validate_request(request, *, now_us, max_sync_error_us, policy=None):
    try:
        r = load_document(json.dumps(request,allow_nan=False),"request")
    except (TypeError,ValueError) as error:
        raise ContractError("invalid request document") from error
    if type(now_us) is not int or now_us<0 or type(max_sync_error_us) is not int or max_sync_error_us<0:
        raise ContractError("microsecond clock/sync prerequisite")
    if not r["history_cutoff_us"]<=r["issued_us"]<=now_us<r["valid_until_us"]:
        raise ContractError("request causal time or expiry")
    effective_sync_limit = min(max_sync_error_us,policy.max_sync_error_us) if policy else max_sync_error_us
    if not r["sync"]["valid"] or r["sync"]["maximum_error_us"]>effective_sync_limit:
        raise ContractError("unsynchronized request")
    horizon,step = r["horizon_us"],r["step_us"]
    if horizon%step or horizon//step>=256:
        raise ContractError("invalid fixed time grid")
    if policy:
        field = tuple(r["field"][k] for k in ("season","map_id","geometry_revision"))
        if r["model_id"] not in policy.model_calibrations or field not in policy.allowed_fields or r["source"]["configuration_revision"] not in policy.allowed_configuration_revisions:
            raise ContractError("unapproved model/field/configuration")
        if horizon>policy.max_horizon_us or now_us-r["history_cutoff_us"]>min(policy.max_history_age_us,horizon):
            raise ContractError("request horizon or history too old")
    last_t,last_id = -1,-1
    for h in r["history"]:
        snap,ego = h["snapshot"],h["snapshot"]["ego"]
        t = ego["time_us"]
        if snap["epoch"]!=r["epoch"] or snap["field"]!=r["field"] or not last_t<t<=r["history_cutoff_us"] or snap["id"]<=last_id:
            raise ContractError("history identity/order/cutoff")
        last_t,last_id = t,snap["id"]
        cov = ego["uncertainty"]
        _covariance(cov["xx_m2"],cov["xy_m2"],cov["yy_m2"])
        if policy and r["history_cutoff_us"]-t>policy.max_history_age_us:
            raise ContractError("history sample too old")
        if h["valid_mask"] and policy:
            p,v = ego["field_pose"]["position_m"],ego["field_velocity"]
            if not ego["localization_valid"] or not policy.inside(p["x"],p["y"]) or math.hypot(v["linear_mps"]["x"],v["linear_mps"]["y"])>policy.max_speed_mps or abs(v["angular_radps"])>policy.max_omega_radps:
                raise ContractError("ego outside physical domain")
        track_ids = set()
        for track in snap["tracks"]:
            if track["id"] in track_ids or track["epoch"]!=r["epoch"] or not track["last_measurement_us"]<=track["estimate_us"]<=t:
                raise ContractError("track identity/time")
            track_ids.add(track["id"])
            cov = track["uncertainty"]
            _covariance(cov["xx_m2"],cov["xy_m2"],cov["yy_m2"])
            if policy and (not policy.inside(track["position_m"]["x"],track["position_m"]["y"]) or math.hypot(track["velocity_mps"]["x"],track["velocity_mps"]["y"])>policy.max_speed_mps):
                raise ContractError("track outside physical domain")
            for stamp in track["provenance"]:
                if not stamp["capture_us"]<=track["last_measurement_us"] or not stamp["capture_us"]<=stamp["publication_us"]<=r["issued_us"]:
                    raise ContractError("capture/publication provenance time")
        accepted = h["accepted_command"]
        if accepted:
            if not accepted["issued_us"]<=accepted["accepted_us"]<=t:
                raise ContractError("accepted command time")
            _action(accepted["body_twist"],accepted["mechanisms"],policy)
    latest = r["history"][-1]
    if last_id!=r["snapshot_id"] or not latest["valid_mask"] or not latest["snapshot"]["ego"]["localization_valid"]:
        raise ContractError("latest valid snapshot prerequisite")
    targets,tracks,ego_count = set(),set(),0
    known_tracks = {t["id"] for t in latest["snapshot"]["tracks"]}
    for target in r["targets"]:
        if target["target_id"] in targets: raise ContractError("duplicate target")
        targets.add(target["target_id"])
        if target["kind"]=="EGO": ego_count+=1
        else:
            if target["track_id"] in tracks or target["track_id"] not in known_tracks: raise ContractError("missing/duplicate track target")
            tracks.add(target["track_id"])
    if ego_count>1: raise ContractError("duplicate ego target")
    candidates = set()
    grid = list(range(0,horizon+1,step))
    for candidate in r["candidates"]:
        if candidate["candidate_id"] in candidates or [s["offset_us"] for s in candidate["samples"]]!=grid:
            raise ContractError("candidate identity/time grid")
        candidates.add(candidate["candidate_id"])
        for sample in candidate["samples"]: _action(sample["body_twist"],sample["mechanisms"],policy)
    return r


def parse_request(payload, *, now_us, max_sync_error_us, policy=None):
    return validate_request(load_document(payload,"request"),now_us=now_us,max_sync_error_us=max_sync_error_us,policy=policy)


def decode_ego_history6(request):
    values = []
    for h in request["history"]:
        ego = h["snapshot"]["ego"]
        if not h["valid_mask"] or not ego["localization_valid"]:
            raise ContractError("unknown ego history cannot be zero-filled")
        theta = ego["field_pose"]["heading_rad"]
        p,v = ego["field_pose"]["position_m"],ego["field_velocity"]["linear_mps"]
        c,s = math.cos(theta),math.sin(theta)
        values.append((p["x"],p["y"],theta,c*v["x"]+s*v["y"],-s*v["x"]+c*v["y"],ego["field_velocity"]["angular_radps"]))
    return tuple(values)


def candidate_body_actions3(request,candidate_id,*,state_dim=6,action_dim=3):
    if (state_dim,action_dim)!=(6,3):
        raise ContractError("v1 has no mechanism state; missing learned prerequisite")
    candidate = next((c for c in request["candidates"] if c["candidate_id"]==candidate_id),None)
    if candidate is None: raise ContractError("missing candidate")
    if any(s["mechanisms"] for s in candidate["samples"]):
        raise ContractError("ego-only model cannot discard configured mechanism inputs")
    return tuple(tuple(s["body_twist"][k] for k in ("vx_mps","vy_mps","omega_radps")) for s in candidate["samples"][:-1])


def make_masked_reply(request,generated_us):
    if not request["issued_us"]<=generated_us<request["valid_until_us"]:
        raise ContractError("abstention reply time")
    r = {k:request[k] for k in IDENTITIES}
    r.update(valid_until_us=request["valid_until_us"],horizon_us=request["horizon_us"],step_us=request["step_us"],
             generated_us=generated_us,kind="LEARNED_FORECAST",uncertainty_calibration_id="uncalibrated-abstention-v1")
    r["forecasts"] = [{"candidate_id":c["candidate_id"],"target_id":t["target_id"],
        "samples":[{"offset_us":offset,"valid_mask":False,"x_m":0,"y_m":0,"vx_mps":0,"vy_mps":0,
                    "covariance_xx_m2":0,"covariance_xy_m2":0,"covariance_yy_m2":0,"confidence":None}
                   for offset in range(0,r["horizon_us"]+1,r["step_us"])]}
        for c in request["candidates"] for t in request["targets"]]
    return load_document(json.dumps(r,allow_nan=False),"reply")


def validate_reply(request,reply,*,now_us,policy,current_snapshot=None):
    try:
        reply = load_document(json.dumps(reply,allow_nan=False),"reply")
    except (TypeError,ValueError) as error:
        raise ContractError("invalid reply document") from error
    validate_request(request,now_us=now_us,max_sync_error_us=policy.max_sync_error_us,policy=policy)
    if any(reply[k]!=request[k] for k in IDENTITIES): raise ContractError("reply identity/epoch/frame/cutoff mismatch")
    if not request["issued_us"]<=reply["generated_us"]<=now_us<reply["valid_until_us"]<=request["valid_until_us"]:
        raise ContractError("reply stale/future/expiry")
    if reply["horizon_us"]!=request["horizon_us"] or reply["step_us"]!=request["step_us"]:
        raise ContractError("reply time grid mismatch")
    if reply["uncertainty_calibration_id"] not in policy.model_calibrations.get(reply["model_id"],()):
        raise ContractError("uncalibrated reply; baseline required")
    latest = request["history"][-1]["snapshot"]
    if current_snapshot is not None:
        # Reuse the exported snapshot schema rather than accepting a weaker
        # ad-hoc current-world layout. Current snapshots are never rebased.
        try:
            from jsonschema import Draft202012Validator
            schema = json.loads((CONTRACT / "prediction-request-v1.schema.json").read_text())
            _finite(current_snapshot)
            Draft202012Validator({"$ref":"#/$defs/snapshot","$defs":schema["$defs"]}).validate(current_snapshot)
        except Exception as error:
            raise ContractError("invalid current snapshot") from error
        if current_snapshot["id"]<request["snapshot_id"] or current_snapshot["epoch"]!=request["epoch"] or current_snapshot["field"]!=request["field"] or current_snapshot["obstacle_map_version"]!=latest["obstacle_map_version"]:
            raise ContractError("current world identity/epoch/map changed")
        ego = current_snapshot["ego"]
        if not ego["localization_valid"] or not latest["ego"]["time_us"]<=ego["time_us"]<=now_us or now_us-ego["time_us"]>policy.max_history_age_us:
            raise ContractError("current localization stale/future/invalid")
        p,v = ego["field_pose"]["position_m"],ego["field_velocity"]
        if not policy.inside(p["x"],p["y"]) or math.hypot(v["linear_mps"]["x"],v["linear_mps"]["y"])>policy.max_speed_mps or abs(v["angular_radps"])>policy.max_omega_radps:
            raise ContractError("current ego outside physical domain")
        cov = ego["uncertainty"]
        _covariance(cov["xx_m2"],cov["xy_m2"],cov["yy_m2"])
        required_tracks = {t["track_id"] for t in request["targets"] if t["kind"]=="OBJECT_TRACK"}
        if not required_tracks.issubset({t["id"] for t in current_snapshot["tracks"]}): raise ContractError("target track no longer current")
    wanted = {(c["candidate_id"],t["target_id"]) for c in request["candidates"] for t in request["targets"]}
    seen,usable = set(),False
    grid = list(range(0,request["horizon_us"]+1,request["step_us"]))
    for forecast in reply["forecasts"]:
        pair = (forecast["candidate_id"],forecast["target_id"])
        if pair in seen or pair not in wanted or [s["offset_us"] for s in forecast["samples"]]!=grid:
            raise ContractError("forecast pair/dimensions/time grid")
        seen.add(pair)
        target = next(t for t in request["targets"] if t["target_id"]==pair[1])
        anchor = latest["ego"]["field_pose"]["position_m"] if target["kind"]=="EGO" else next(t["position_m"] for t in latest["tracks"] if t["id"]==target["track_id"])
        previous = None
        for sample in forecast["samples"]:
            if not sample["valid_mask"]: continue  # schema already requires canonical zeros
            x,y = sample["x_m"],sample["y_m"]
            if not policy.inside(x,y) or math.hypot(sample["vx_mps"],sample["vy_mps"])>policy.max_speed_mps:
                raise ContractError("forecast outside physical domain")
            allowance = policy.max_initial_offset_m+policy.max_speed_mps*sample["offset_us"]/1e6
            if math.hypot(x-anchor["x"],y-anchor["y"])>allowance:
                raise ContractError("forecast displacement from anchor")
            if previous and math.hypot(x-previous["x_m"],y-previous["y_m"])>policy.max_speed_mps*(sample["offset_us"]-previous["offset_us"])/1e6+1e-9:
                raise ContractError("forecast exceeds displacement bound")
            previous = sample
            _covariance(sample["covariance_xx_m2"],sample["covariance_xy_m2"],sample["covariance_yy_m2"],policy)
            usable |= sample["offset_us"]>=now_us-request["history_cutoff_us"]
    if seen!=wanted or not usable: raise ContractError("missing pairs or no remaining valid forecasts")
    return reply


def parse_reply(payload,request,*,now_us,policy,current_snapshot=None):
    return validate_reply(request,load_document(payload,"reply"),now_us=now_us,policy=policy,current_snapshot=current_snapshot)
