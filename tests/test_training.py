from dataclasses import asdict
import json

import pytest
import torch

from frc_world_model.data import FeatureLayout, SplitConfig, WindowConfig, load_jsonl, prepare_offline
from frc_world_model.rwm import ModelConfig
from frc_world_model.training import train_bounded, training_tensors, write_package, load_frozen_model, predictor_callback
from frc_world_model.evaluation import PredictorInput
from tools.synthetic_smoke import generate


def test_tiny_training_package_serialization_and_tensor_parity(tmp_path):
    log, config_path = generate(tmp_path/"fixture")
    cfg = json.loads(config_path.read_text())
    state, action = FeatureLayout.from_dict(cfg["state_layout"]), FeatureLayout.from_dict(cfg["action_layout"])
    report = load_jsonl(log, state, action)
    split, windows, sn, an = prepare_offline(report, state, action, WindowConfig(**cfg["window"]),
                                            SplitConfig((.25,.25,.25,.25),"synthetic-v1","day"))
    assert set(split.records) == {"train","validation","calibration","test"}
    assert all(w.windows for w in windows.values())
    tensor_s, tensor_a = training_tensors(windows["train"].windows, sn, an)
    assert tensor_s.shape[1:] == (40,8) and tensor_a.shape[1:] == (40,4)
    assert torch.equal(tensor_a[:,1:32], torch.tensor([[an.transform(r.values) for r in w.history_actions] for w in windows["train"].windows]))
    c = ModelConfig()
    model, trained = train_bounded(windows["train"].windows,sn,an,c,max_updates=1,batch_size=2)
    assert len(trained["losses"]) == 1 and trained["device"] == "cpu"
    manifest = {"ingestion":report.manifest(),"split":split.manifest()}
    write_package(tmp_path/"package",model,c,sn,an,manifest,trained)
    package, frozen = load_frozen_model(tmp_path/"package")
    assert package.metadata["calibration"]["status"] == "uncalibrated"
    inp = PredictorInput.from_window(windows["validation"].windows[0])
    expected = predictor_callback(model,c,sn,an)(inp)
    actual = predictor_callback(frozen.model,c,sn,an)(inp)
    assert expected == actual
    with pytest.raises(ValueError):
        train_bounded(windows["train"].windows,sn,an,c,max_updates=101)
    with pytest.raises(FileExistsError):
        write_package(tmp_path/"package",model,c,sn,an,manifest,trained)
