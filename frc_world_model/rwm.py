"""Pinned RWM source adapter, with explicit carry-isolation correctness change.

No simulator, actor, optimizer, external logger, weight download, or device
selection occurs on import. Dimensions are task dimensions, not ANYmal weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import threading

import torch

from .upstream.system_dynamics import SystemDynamicsEnsemble

CORE_COMMIT = "18eebcdd7145284c8d5eed5d8ed1a4b96c649693"
LITE_COMMIT = "13a798e9d35dabf12c0e6e02977b25ec64dfb2bd"
ARCHITECTURE = {
    "type": "rnn", "rnn_type": "gru", "rnn_num_layers": 2,
    "rnn_hidden_size": 256, "state_mean_shape": [128],
    "state_logstd_shape": [128], "extension_shape": [128],
    "contact_shape": [128], "termination_shape": [128],
}


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = 8
    action_dim: int = 4
    extension_dim: int = 0
    contact_dim: int = 0
    termination_dim: int = 0
    ensemble_size: int = 5
    history_horizon: int = 32
    forecast_horizon: int = 8
    timestep_us: int = 20_000
    bootstrap: bool = False
    loss: str = "sampled_mse"
    method: str = "RWM-U-source/carry-isolated-v1"

    def __post_init__(self):
        dimensions = (self.state_dim,self.action_dim,self.extension_dim,self.contact_dim,self.termination_dim,
                      self.ensemble_size,self.history_horizon,self.forecast_horizon,self.timestep_us)
        if any(type(v) is not int for v in dimensions) or type(self.bootstrap) is not bool:
            raise ValueError("integer model dimensions/timing and boolean bootstrap required")
        if min(self.state_dim, self.action_dim, self.ensemble_size,
               self.history_horizon, self.forecast_horizon, self.timestep_us) <= 0:
            raise ValueError("positive model dimensions and timing required")
        if min(self.extension_dim, self.contact_dim, self.termination_dim) < 0:
            raise ValueError("negative auxiliary dimension")
        if self.loss != "sampled_mse":
            raise ValueError("only explicitly selected upstream sampled MSE is enabled")
        if max(self.state_dim,self.action_dim,self.extension_dim,self.contact_dim,self.termination_dim)>256 or self.ensemble_size>8 or self.history_horizon>128 or self.forecast_horizon>64 or self.timestep_us>1_000_000:
            raise ValueError("model capacity exceeds bounded offline/reference path")
        if self.method!="RWM-U-source/carry-isolated-v1":
            raise ValueError("method change requires a separate adapter implementation")

    def resolved(self) -> dict[str, Any]:
        return {**asdict(self), "device": "cpu", "architecture": ARCHITECTURE,
                "core_commit": CORE_COMMIT, "lite_commit": LITE_COMMIT,
                "auxiliary_label_policy": "disabled-unless-real-labels-exist"}


class CarryIsolatedEnsemble(SystemDynamicsEnsemble):
    """Reset shared GRU before each member's independent training rollout.

    Upstream compute_loss loops heads without resetting either shared base.
    This changes that observed behavior; it does not change GRU equations,
    heads, sampling, residuals, loss or the shared-base ensemble structure.
    """

    def compute_state_loss(self, head, state_batch, action_batch):
        self.state_base.reset()
        return super().compute_state_loss(head, state_batch, action_batch)

    def compute_auxiliary_loss(self, head, *args):
        self.auxiliary_base.reset()
        return super().compute_auxiliary_loss(head, *args)


def make_model(config: ModelConfig, *, reference: bool = False):
    cls = SystemDynamicsEnsemble if reference else CarryIsolatedEnsemble
    return cls(state_dim=config.state_dim, action_dim=config.action_dim,
               extension_dim=config.extension_dim, contact_dim=config.contact_dim,
               termination_dim=config.termination_dim, device="cpu",
               ensemble_size=config.ensemble_size, history_horizon=config.history_horizon,
               architecture_config={k: list(v) if isinstance(v, list) else v
                                    for k, v in ARCHITECTURE.items()})


def indexed_training_actions(applied_intervals: torch.Tensor) -> torch.Tensor:
    """Map actions for s[i] -> s[i+1] to upstream index i+1.

    The upstream trainer slices action_batch[:, 1:H+1] for states[:, :H].
    Its index zero is unused. Never shift targets to hide this convention.
    """
    if applied_intervals.ndim != 3 or applied_intervals.shape[1] < 1:
        raise ValueError("expected [batch, intervals, action] tensor")
    return torch.cat((torch.zeros_like(applied_intervals[:, :1]), applied_intervals), dim=1)


@dataclass(frozen=True)
class TensorForecast:
    member_states: torch.Tensor  # [members,batch,horizon,state], physical calibration separate
    member_stds: torch.Tensor
    extensions: torch.Tensor | None
    contact_logits: torch.Tensor | None
    termination_logits: torch.Tensor | None
    uncertainty_semantics: str = "normalized spread; not calibrated physical covariance"

    def diagnostics(self):
        return {"normalized_head_std":self.member_stds.mean(0),
                "normalized_ensemble_disagreement":self.member_states.std(0,correction=0),
                "sensor_error":None,"empirical_forecast_error":None,
                "covariance_calibration":"unavailable; separate held-out sessions required",
                "sampling":"mean rollout unless sample=True; a head std is not multi-step error covariance"}


class FrozenRWM:
    """Independent candidate rollouts; recurrent carry never crosses a call.

    Historical states are pre-transition values with corresponding interval
    actions. Last history action is replaced by the first future command so
    the first forecast is conditioned on the command actually being ranked.
    Future states are always this model's outputs, never measurements.
    """

    def __init__(self, model: SystemDynamicsEnsemble, config: ModelConfig):
        self.model, self.config = model.eval(), config
        self.model.requires_grad_(False)
        self._carry_lock = threading.RLock()

    def rollout(self, history_states: torch.Tensor, history_actions: torch.Tensor,
                future_actions: torch.Tensor, *, sample: bool = False,
                seed: int = 0) -> TensorForecast:
        with self._carry_lock:
            return self._rollout(history_states, history_actions, future_actions, sample=sample, seed=seed)

    def _rollout(self, history_states: torch.Tensor, history_actions: torch.Tensor,
                 future_actions: torch.Tensor, *, sample: bool = False,
                 seed: int = 0) -> TensorForecast:
        c = self.config
        if history_states.ndim != 3 or history_actions.ndim != 3 or future_actions.ndim != 3:
            raise ValueError("expected batch/time/features")
        batch, history, state_dim = history_states.shape
        horizon = future_actions.shape[1]
        if (state_dim != c.state_dim or history != c.history_horizon
                or history_actions.shape != (batch, history, c.action_dim)
                or future_actions.shape != (batch, horizon, c.action_dim)
                or not 0 < horizon <= c.forecast_horizon):
            raise ValueError("unsupported shape or horizon")
        for tensor in (history_states, history_actions, future_actions):
            if tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
                raise ValueError("finite CPU tensors required")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        all_states, all_stds, all_aux = [], [], [[], [], []]
        try:
            with torch.inference_mode():
                for member in range(c.ensemble_size):
                    self.model.reset()
                    state_input = history_states.clone()
                    action_input = history_actions.clone()
                    action_input[:, -1] = future_actions[:, 0]
                    member_states, member_stds, auxiliary = [], [], [[], [], []]
                    for step in range(horizon):
                        if step:
                            action_input = future_actions[:, step:step+1]
                        latent = self.model.state_base(state_input, action_input)
                        mean, std = self.model.state_heads[member](latent, state_input)
                        aux_latent = self.model.auxiliary_base(state_input, action_input)
                        outputs = self.model.auxiliary_heads[member](aux_latent, state_input)
                        value = mean + std * torch.randn(mean.shape, generator=generator) if sample else mean
                        member_states.append(value.clone())
                        member_stds.append(std.clone())
                        for target, output in zip(auxiliary, outputs):
                            if output is not None:
                                target.append(output.clone())
                        state_input = value.unsqueeze(1)
                    all_states.append(torch.stack(member_states, dim=1))
                    all_stds.append(torch.stack(member_stds, dim=1))
                    for target, output in zip(all_aux, auxiliary):
                        if output:
                            target.append(torch.stack(output, dim=1))
        finally:
            self.model.reset()
        return TensorForecast(torch.stack(all_states), torch.stack(all_stds),
                              *(torch.stack(v) if v else None for v in all_aux))
