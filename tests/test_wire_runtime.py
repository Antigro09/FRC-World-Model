import copy
import json

import pytest
import torch

from frc_world_model.bindings import CONTRACT,ContractError,load_document
from frc_world_model.converter import EGO_LAYOUT,COMMAND_LAYOUT
from frc_world_model.data import ReversibleNormalizer
from frc_world_model.rwm import ModelConfig,make_model
from frc_world_model.training import write_package
from frc_world_model.wire_runtime import FrozenWireShadow,HistoryPrerequisites,prepare_wire_inputs


def request():
    r=json.loads((CONTRACT/"request-v1.json").read_text())
    for h in r["history"]:
        h["accepted_command"]["accepted_us"]=h["snapshot"]["ego"]["time_us"]
        h["accepted_command"]["mechanisms"]=[]
    r["targets"]=r["targets"][:1]
    for c in r["candidates"]:
        for s in c["samples"]: s["mechanisms"]=[]
    return r


def test_timed_command_alignment_and_missing_grid_prerequisites():
    r=request(); config=ModelConfig(state_dim=6,action_dim=3,history_horizon=2,forecast_horizon=2,timestep_us=100000)
    p=HistoryPrerequisites("robot-config-v1",True)
    states,past,future=prepare_wire_inputs(r,config,p)
    assert past==((.2,0.,0.),) and future["hold"]==((0.,0.,0.),(0.,0.,0.))
    assert future["transit"]==((.2,0.,0.),(.2,0.,0.))
    assert len(states)==2
    for bad in (HistoryPrerequisites("changed",True),HistoryPrerequisites("robot-config-v1",False)):
        with pytest.raises(ContractError): prepare_wire_inputs(r,config,bad)
    with pytest.raises(ContractError): prepare_wire_inputs(r,ModelConfig(),p)
    bad=copy.deepcopy(r); bad["history"][0]["snapshot"]["ego"]["time_us"]-=1
    with pytest.raises(ContractError,match="gaps"): prepare_wire_inputs(bad,config,p)
    bad=copy.deepcopy(r); bad["history"][1]["accepted_command"]["accepted_us"]-=1
    with pytest.raises(ContractError,match="changing"): prepare_wire_inputs(bad,config,p)


def test_frozen_native_shadow_is_exact_wire_abstention_and_candidate_isolation(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(0)
    r=request();c=ModelConfig(state_dim=6,action_dim=3,history_horizon=2,forecast_horizon=2,timestep_us=100000)
    sn=ReversibleNormalizer.fit([(0.,0.,0.,0.,0.,0.),(1.,2.,.1,.2,.1,.2)],EGO_LAYOUT,split="train")
    an=ReversibleNormalizer.fit([(0.,0.,0.),(.2,.1,.2)],COMMAND_LAYOUT,split="train")
    manifest=write_package(tmp_path/"package",make_model(c),c,sn,an,{"ingestion":{"schema_version":"synthetic-wire-test"}},
                           {"synthetic_fixture":True,"updates":0})
    r["model_id"]=manifest["model_id"]
    shadow=FrozenWireShadow(tmp_path/"package",prerequisites=HistoryPrerequisites("robot-config-v1",True),
                            now_us=lambda:1030000,max_sync_error_us=1000)
    reply=load_document(shadow(json.dumps(r)),"reply")
    assert all(not s["valid_mask"] for f in reply["forecasts"] for s in f["samples"])
    assert shadow.last_diagnostic["influences_commands"] is False
    expected=copy.deepcopy(shadow.last_diagnostic["candidates"])
    assert set(expected)=={"hold","transit"}
    shadow(json.dumps(r))
    assert expected==shadow.last_diagnostic["candidates"]
    shadow._lock.acquire()
    try:
        overloaded=load_document(shadow(json.dumps(r)),"reply")
        assert all(not s["valid_mask"] for f in overloaded["forecasts"] for s in f["samples"])
    finally: shadow._lock.release()
    r["model_id"]="wrong-weights"
    with pytest.raises(ContractError,match="model ID"): shadow(json.dumps(r))
