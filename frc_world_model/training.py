"""Model-only CPU orchestration; no policy training or external logging."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .data import ReversibleNormalizer, Window, canonical_hash
from .rwm import FrozenRWM, ModelConfig, indexed_training_actions, make_model


def training_tensors(windows, state_norm, action_norm):
    if not windows:
        raise ValueError("no complete authorized training windows")
    states, actions = [], []
    for w in windows:
        states.append([state_norm.transform(r.values) for r in w.history + w.targets])
        actions.append([action_norm.transform(r.values) for r in w.history_actions + w.actions])
    state = torch.tensor(states, dtype=torch.float32)
    action = indexed_training_actions(torch.tensor(actions, dtype=torch.float32))
    if not torch.isfinite(state).all() or not torch.isfinite(action).all():
        raise ValueError("nonfinite complete window")
    return state, action


def train_bounded(windows, state_norm, action_norm, config: ModelConfig, *,
                  max_updates=2, batch_size=4, seed=0):
    """Hard bounded smoke/reference path. Larger experiments require new scope.

    FRC auxiliary dimensions remain zero until an explicit labelled target
    adapter exists. No loss masks or synthetic possession targets are invented.
    """
    if not 1 <= max_updates <= 100 or not 1 <= batch_size <= 32:
        raise ValueError("bounded offline path requires <=100 updates and batch<=32")
    if config.extension_dim or config.contact_dim or config.termination_dim:
        raise ValueError("auxiliary training requires a separately approved label adapter")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    state, action = training_tensors(windows, state_norm, action_norm)
    if state.shape[1:] != (config.history_horizon + config.forecast_horizon, config.state_dim):
        raise ValueError("state dimensions do not match resolved model config")
    if action.shape[-1] != config.action_dim:
        raise ValueError("action dimension mismatch")
    model = make_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    started = time.perf_counter()
    for _ in range(max_updates):
        ids = torch.randperm(state.shape[0], generator=generator)[:batch_size]
        model.reset()
        optimizer.zero_grad(set_to_none=True)
        components = model.compute_loss(state[ids], action[ids], None, None, None,
                                        bootstrap=config.bootstrap)
        # Same reference coefficients for active state + bound terms; others zero.
        coefficients = (1.,1.,1.,.1,1.,1.,1.)
        loss = sum(weight*component for weight,component in zip(coefficients,components))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.reset()
    return model, {"updates": max_updates, "batch_size": batch_size, "seed": seed,
                   "losses": losses, "duration_s": time.perf_counter() - started,
                   "optimizer": {"type": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-5},
                   "device": "cpu", "external_logging": False,
                   "selected_loss": "upstream sampled MSE + bound regularization",
                   "bootstrap": config.bootstrap, "method": config.method}


def predictor_callback(model, config, state_norm, action_norm):
    frozen = FrozenRWM(model, config)
    def predict(inputs):
        if len(inputs.history) != config.history_horizon:
            raise ValueError("fixed learned history prerequisite missing")
        # The history interval actions are H-1; the command at cutoff belongs to
        # the future candidate. This is the same action indexing used in training.
        history_actions = inputs.history_actions + inputs.actions[:1]
        if len(history_actions) != config.history_horizon:
            raise ValueError("accepted history commands missing")
        forecast = frozen.rollout(
            torch.tensor([[state_norm.transform(s) for s in inputs.history]], dtype=torch.float32),
            torch.tensor([[action_norm.transform(a) for a in history_actions]], dtype=torch.float32),
            torch.tensor([[action_norm.transform(a) for a in inputs.actions]], dtype=torch.float32))
        mean = forecast.member_states.mean(0)[0].tolist()
        return tuple(state_norm.inverse(s) for s in mean)
    return predict


def write_package(output: Path, model, config, state_norm, action_norm,
                  dataset_manifest, training_report, calibration=None):
    """Exclusive package creation; CPU tensor state only, no ANYmal weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset_hash = canonical_hash(dataset_manifest)
    layout_id = canonical_hash(asdict(state_norm.layout))
    schema_version = dataset_manifest["ingestion"]["schema_version"]
    metadata = {
        "config": config.resolved(),
        "schema": {"version": schema_version, "layout_id": layout_id,
                   "layout": asdict(state_norm.layout), "binding": "offline local records"},
        "normalization": {"fit_split": "train", "dataset_sha256": dataset_hash,
                          "mean": state_norm.centers, "scale": state_norm.scales,
                          "layout": asdict(state_norm.layout), "state": state_norm.metadata(),
                          "action": action_norm.metadata()},
        "calibration": calibration or {"status": "uncalibrated", "advisory_only": True,
                                        "reason": "no approved real held-out calibration sessions/tolerances"},
        "dataset": dataset_manifest,
        "training": training_report,
    }
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / "weights.pt")
    artifacts = {"weights": {"path": "weights.pt"}}
    for name, value in metadata.items():
        path = output / (name + ".json")
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        artifacts[name] = {"path": path.name}
    for record in artifacts.values():
        record["sha256"] = hashlib.sha256((output / record["path"]).read_bytes()).hexdigest()
    model_id = "frc-rwm-" + artifacts["weights"]["sha256"][:16]
    manifest = {"format": "frc-world-model-frozen-v1", "model_id": model_id,
                "schema_version": schema_version, "layout_id": layout_id,
                "model_kind": "rwm-u", "artifacts": artifacts,
                "mode": "replay/shadow", "influences_commands": False,
                "config_sha256": canonical_hash(config.resolved()),
                "claims": "tiny training proves plumbing only; uncalibrated advisory predictor"}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest



# Backward-compatible offline import; runtime imports .frozen directly.
from .frozen import load_frozen_model
