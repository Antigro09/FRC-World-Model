"""Exact-wire frozen CPU shadow adapter. Its replies always abstain in v1.

Raw model outputs may be retained as shadow diagnostics; they cannot supply a
calibrated covariance or authorize advisory promotion. No robot endpoint exists.
"""
from dataclasses import dataclass
import json
import threading

from .bindings import ContractError, candidate_body_actions3, decode_ego_history6, make_masked_reply, parse_request
from .converter import EGO_LAYOUT, COMMAND_LAYOUT
from .data import FeatureLayout, ReversibleNormalizer


@dataclass(frozen=True)
class HistoryPrerequisites:
    configuration_revision: str
    complete_accepted_command_history: bool


def prepare_wire_inputs(request, config, prerequisites):
    if not prerequisites.complete_accepted_command_history or prerequisites.configuration_revision != request["source"]["configuration_revision"]:
        raise ContractError("complete accepted-command history/configuration prerequisite missing")
    if (config.state_dim,config.action_dim)!=(6,3):
        raise ContractError("wire v1 has no mechanism-state inputs for this model layout")
    if request["step_us"]!=config.timestep_us or request["horizon_us"]//request["step_us"]>config.forecast_horizon:
        raise ContractError("wire/model fixed timestep or horizon mismatch")
    if len(request["history"])<config.history_horizon:
        raise ContractError("insufficient fixed learned history")
    history=request["history"][-config.history_horizon:]
    times=[h["snapshot"]["ego"]["time_us"] for h in history]
    if times[-1]!=request["history_cutoff_us"] or any(b-a!=config.timestep_us for a,b in zip(times,times[1:])):
        raise ContractError("history gaps or cutoff not on learned time grid")
    states=decode_ego_history6({**request,"history":history})
    commands={}
    for h in request["history"]:
        a=h["accepted_command"]
        if a:
            if a["mechanisms"]: raise ContractError("configured mechanism inputs cannot be discarded")
            key=a["accepted_us"]
            value=tuple(a["body_twist"][k] for k in ("vx_mps","vy_mps","omega_radps"))
            if key in commands and commands[key]!=value: raise ContractError("conflicting accepted commands")
            commands[key]=value
    past=[]
    for start,end in zip(times,times[1:]):
        available=[t for t in commands if t<=start]
        if not available or any(start<t<end for t in commands):
            raise ContractError("missing or changing command within fixed history interval")
        past.append(commands[max(available)])
    future={c["candidate_id"]:candidate_body_actions3(request,c["candidate_id"],state_dim=config.state_dim,action_dim=config.action_dim)
            for c in request["candidates"]}
    return states,tuple(past),future


def _normalizer(raw):
    if raw["fit_split"]!="train": raise ContractError("normalization not fitted on train")
    return ReversibleNormalizer(FeatureLayout.from_dict(raw["layout"]),tuple(raw["centers"]),tuple(raw["scales"]),raw["fit_count"],raw["training_values_hash"],raw["angle_method"])


class FrozenWireShadow:
    """Serialized handler for the supplied localhost transport or offline replay.

    now_us must be the explicitly mapped robot clock; host arrival time is never
    substituted. Unavailable inputs/inference keep canonical masked replies.
    """
    def __init__(self, package, *, prerequisites, now_us, max_sync_error_us, max_candidates=8):
        from .frozen import load_frozen_model
        self.package,self.frozen=load_frozen_model(package)
        raw=self.package.metadata["normalization"]
        self.state_norm,self.action_norm=_normalizer(raw["state"]),_normalizer(raw["action"])
        if self.state_norm.layout!=EGO_LAYOUT or self.action_norm.layout!=COMMAND_LAYOUT:
            raise ContractError("frozen package layout is not exact wire ego6/body3")
        if not 1<=max_candidates<=8: raise ContractError("unsupported candidate capacity")
        self.prerequisites,self.now_us,self.max_sync_error_us,self.max_candidates=prerequisites,now_us,max_sync_error_us,max_candidates
        self._lock=threading.Lock()
        self.last_diagnostic=None

    def __call__(self,payload):
        import torch
        request=parse_request(payload,now_us=self.now_us(),max_sync_error_us=self.max_sync_error_us)
        if request["model_id"]!=self.package.manifest["model_id"]:
            raise ContractError("request model ID does not match frozen package")
        # Shed optional prediction instead of waiting behind another rollout.
        if not self._lock.acquire(blocking=False):
            return json.dumps(make_masked_reply(request,self.now_us()),separators=(",",":"))
        diagnostic={"request_id":request["request_id"],"snapshot_id":request["snapshot_id"],"epoch":request["epoch"],
                    "model_id":request["model_id"],"source":request["source"],"history_cutoff_us":request["history_cutoff_us"],
                    "mode":"shadow","influences_commands":False,"kind":"RAW_UNCALIBRATED_FORECAST",
                    "physical_covariance":None,"sensor_error":None,"empirical_error":None}
        try:
            if len(request["candidates"])>self.max_candidates: raise ContractError("candidate overload; optional prediction shed")
            states,past,future=prepare_wire_inputs(request,self.frozen.config,self.prerequisites)
            normalized_states=torch.tensor([[self.state_norm.transform(s) for s in states]],dtype=torch.float32)
            outputs={}
            for candidate,actions in future.items():
                if self.now_us()>=request["valid_until_us"]: raise ContractError("prediction expired during rollout")
                h_actions=past+actions[:1]
                forecast=self.frozen.rollout(normalized_states,
                    torch.tensor([[self.action_norm.transform(a) for a in h_actions]],dtype=torch.float32),
                    torch.tensor([[self.action_norm.transform(a) for a in actions]],dtype=torch.float32))
                outputs[candidate]={"ego_mean":tuple(self.state_norm.inverse(s) for s in forecast.member_states.mean(0)[0].tolist()),
                    "normalized_head_std":forecast.member_stds.mean(0)[0].tolist(),
                    "normalized_ensemble_disagreement":forecast.member_states.std(0,correction=0)[0].tolist()}
            diagnostic.update(candidates=outputs,reason="raw native inference complete; no approved physical covariance")
        except Exception as error:
            diagnostic.update(reason=type(error).__name__+": "+str(error),candidates={})
        finally:
            self.last_diagnostic=diagnostic
            self._lock.release()
        return json.dumps(make_masked_reply(request,self.now_us()),allow_nan=False,separators=(",",":"))
