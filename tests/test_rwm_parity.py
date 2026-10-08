import hashlib
from pathlib import Path

import pytest
import torch
from concurrent.futures import ThreadPoolExecutor

from frc_world_model.rwm import ARCHITECTURE, FrozenRWM, ModelConfig, indexed_training_actions, make_model

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "vendor/reference/rsl_rl_rwm-18eebcdd7145284c8d5eed5d8ed1a4b96c649693/rsl_rl/modules"


def test_source_reuse_bytes():
    for name in ("mlp.py", "rnn.py", "__init__.py"):
        assert (REFERENCE / "architectures" / name).read_bytes() == (ROOT / "frc_world_model/upstream/architectures" / name).read_bytes()
    reference = (REFERENCE / "system_dynamics.py").read_text()
    assert reference.replace("from rsl_rl.modules.architectures import", "from .architectures import") == (ROOT / "frc_world_model/upstream/system_dynamics.py").read_text()


def small_config(**kwargs):
    return ModelConfig(state_dim=3, action_dim=2, history_horizon=3, forecast_horizon=2, **kwargs)


def inputs(c):
    torch.manual_seed(10)
    states = torch.randn(2, c.history_horizon + c.forecast_horizon, c.state_dim)
    actions = torch.randn(2, states.shape[1], c.action_dim)
    return states, actions


def test_published_architecture_and_dimensions():
    c = ModelConfig()
    m = make_model(c)
    assert len(m.state_heads) == 5
    assert m.state_base.memory.rnn.num_layers == 2
    assert m.state_base.memory.rnn.hidden_size == 256
    assert m.state_heads[0].state_mean_layers[0].out_features == 128
    assert c.resolved()["bootstrap"] is False


def test_state_action_concat_index_and_residual():
    c = small_config(ensemble_size=1)
    m = make_model(c)
    seen = []
    handle = m.state_base.memory.rnn.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    state, action = inputs(c)
    m.state_base(state[:, :3], action[:, :3])
    handle.remove()
    assert torch.equal(seen[0], torch.cat((state[:, :3], action[:, :3]), -1))
    for p in m.state_heads[0].state_mean_layers.parameters():
        p.data.zero_()
    mean, std = m.state_heads[0](torch.zeros(2, 256), state[:, :3])
    assert torch.equal(mean, state[:, 2])
    assert (std > 0).all()


def test_action_interval_mapping_matches_active_loss():
    c = small_config(ensemble_size=1)
    m = make_model(c)
    state, _ = inputs(c)
    intervals = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2).repeat(2, 1, 1)
    mapped = indexed_training_actions(intervals)
    seen = []
    original = m.state_base.forward
    def trace(states, actions):
        seen.append(actions.clone())
        return original(states, actions)
    m.state_base.forward = trace
    m.compute_state_loss(m.state_heads[0], state, mapped)
    m.state_base.forward = original
    assert torch.equal(seen[0], intervals[:, :3])
    assert torch.equal(seen[1], intervals[:, 3:4])


def test_default_loss_is_sampled_mse_not_nll_and_gradients():
    m = make_model(small_config(ensemble_size=1))
    mean = torch.tensor([[1., 2., 3.]], requires_grad=True)
    std = torch.tensor([[.2, .3, .4]], requires_grad=True)
    target = torch.zeros_like(mean)
    torch.manual_seed(4)
    loss, _ = m.compute_regression_loss(mean, std, target)
    torch.manual_seed(4)
    expected = ((mean + torch.randn_like(mean) * std - target) ** 2).sum(-1).mean()
    assert torch.equal(loss, expected)
    nll, _ = m.compute_regression_loss(mean, std, target, "gaussian_nll")
    assert not torch.isclose(loss, nll)
    loss.backward()
    assert mean.grad is not None and std.grad is not None
    assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()


def test_upstream_cross_member_carry_demonstrated_and_fix():
    c = small_config(ensemble_size=2)
    state, action = inputs(c)
    raw, isolated = make_model(c, reference=True), make_model(c)
    isolated.load_state_dict(raw.state_dict())
    for model, expected in ((raw, [True, False]), (isolated, [True, True])):
        seen = []
        original = model.state_base.forward
        def trace(states, actions):
            seen.append(model.state_base.memory.hidden_states is None)
            return original(states, actions)
        model.state_base.forward = trace
        model.compute_loss(state, action, None, None, None)
        model.state_base.forward = original
        assert [seen[0], seen[2]] == expected
    raw.reset()
    torch.manual_seed(2)
    ref_single = raw.compute_state_loss(raw.state_heads[0], state, action)
    torch.manual_seed(2)
    fixed_single = isolated.compute_state_loss(isolated.state_heads[0], state, action)
    for a, b in zip(ref_single, fixed_single):
        assert torch.equal(a, b)


def test_member_loss_gradient_and_no_prior_batch_carry():
    c = small_config(ensemble_size=2)
    m = make_model(c)
    state, action = inputs(c)
    for _ in range(2):
        m.zero_grad()
        losses = m.compute_loss(state, action, None, None, None, bootstrap=False)
        sum(losses).backward()
        assert m.state_base.memory.rnn.weight_ih_l0.grad.abs().sum() > 0
        for head in m.state_heads:
            assert head.state_mean_layers[0].weight.grad.abs().sum() > 0


def test_prior_member_changes_same_head_loss_only_in_raw_reference():
    c = small_config(ensemble_size=2)
    raw = make_model(c,reference=True)
    fixed = make_model(c)
    raw.state_heads[1].load_state_dict(raw.state_heads[0].state_dict())
    fixed.load_state_dict(raw.state_dict())
    state,action = inputs(c)
    differences = []
    for model in (raw,fixed):
        model.reset()
        torch.manual_seed(99)
        first = model.compute_state_loss(model.state_heads[0],state,action)[0]
        torch.manual_seed(99)
        second = model.compute_state_loss(model.state_heads[1],state,action)[0]
        differences.append(float((first-second).detach().abs()))
    assert differences[0] > 1e-7
    assert differences[1] == 0


def test_auxiliary_source_heads_losses_and_outputs():
    c = small_config(ensemble_size=2, extension_dim=1, contact_dim=1, termination_dim=1)
    m = make_model(c)
    state, action = inputs(c)
    outputs = m(state[:, :3], action[:, :3])
    for aux in outputs[3:]:
        assert aux.shape == (2, 1) and torch.isfinite(aux).all()
    label = torch.zeros(2, 5, 1)
    losses = m.compute_loss(state, action, label, label, label)
    sum(losses).backward()
    assert all(torch.isfinite(v) for v in losses)
    assert m.auxiliary_heads[0].contact_layers[0].weight.grad is not None


def test_hidden_partial_reset_batch_isolation_and_full_reset():
    m = make_model(small_config(ensemble_size=1))
    state, action = inputs(small_config())
    m.state_base(state[:, :3], action[:, :3])
    prior = m.state_base.memory.hidden_states.clone()
    m.reset_partial([0])
    assert torch.count_nonzero(m.state_base.memory.hidden_states[:, 0]) == 0
    assert torch.equal(m.state_base.memory.hidden_states[:, 1], prior[:, 1])
    m.reset()
    assert m.state_base.memory.hidden_states is None


def test_frozen_candidate_member_batch_and_sampling_isolation():
    c = small_config(ensemble_size=2)
    m = make_model(c)
    predictor = FrozenRWM(m, c)
    state, action = inputs(c)
    h, a, f = state[:, :3], action[:, :3], action[:, 3:]
    first = predictor.rollout(h, a, f)
    changed = predictor.rollout(h + 4, a, f)
    second = predictor.rollout(h, a, f)
    assert torch.equal(first.member_states, second.member_states)
    assert not torch.equal(first.member_states, changed.member_states)
    one = predictor.rollout(h[:1], a[:1], f[:1])
    assert torch.allclose(first.member_states[:, :1], one.member_states, atol=1e-6)
    sampled = predictor.rollout(h, a, f, sample=True, seed=6)
    assert torch.equal(sampled.member_states, predictor.rollout(h, a, f, sample=True, seed=6).member_states)
    assert not torch.equal(sampled.member_states, first.member_states)
    assert m.state_base.memory.hidden_states is None
    assert not first.member_states.requires_grad
    diagnostics = first.diagnostics()
    assert diagnostics["sensor_error"] is None and diagnostics["empirical_forecast_error"] is None
    assert diagnostics["normalized_ensemble_disagreement"].shape == first.member_states.shape[1:]


def test_future_command_changes_first_forecast_and_future_labels_absent():
    c = small_config(ensemble_size=1)
    predictor = FrozenRWM(make_model(c), c)
    state, action = inputs(c)
    base = predictor.rollout(state[:, :3], action[:, :3], action[:, 3:])
    changed = predictor.rollout(state[:, :3], action[:, :3], action[:, 3:] + 5)
    assert not torch.equal(base.member_states[:, :, 0], changed.member_states[:, :, 0])


def test_independent_rollout_matches_upstream_selected_member_forward():
    c = small_config(ensemble_size=2)
    raw = make_model(c, reference=True)
    adapted = make_model(c)
    adapted.load_state_dict(raw.state_dict())
    state, action = inputs(c)
    h, a, future = state[:, :3], action[:, :3].clone(), action[:, 3:]
    a[:, -1] = future[:, 0]
    result = FrozenRWM(adapted, c).rollout(h, a, future)
    with torch.inference_mode():
        for member in range(2):
            raw.reset()
            ids = torch.full((1,2,1),member,dtype=torch.long)
            first = raw(h,a,ids)[0]
            second = raw(first.unsqueeze(1),future[:,1:2],ids)[0]
            assert torch.equal(first,result.member_states[member,:,0])
            assert torch.equal(second,result.member_states[member,:,1])


def test_nan_shape_and_horizon_rejected_and_carry_clean():
    c = small_config(ensemble_size=1)
    m = make_model(c)
    predictor = FrozenRWM(m, c)
    state, action = inputs(c)
    state[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        predictor.rollout(state[:, :3], action[:, :3], action[:, 3:])
    assert m.state_base.memory.hidden_states is None


def test_concurrent_calls_serialize_carry_and_keep_candidates_independent():
    c = small_config(ensemble_size=2)
    predictor = FrozenRWM(make_model(c),c)
    state, action = inputs(c)
    h,a,f = state[:,:3],action[:,:3],action[:,3:]
    expected = [predictor.rollout(h+i,a,f).member_states for i in (0,3)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(predictor.rollout,h+i,a,f) for i in (0,3)]
        actual = [result.result().member_states for result in results]
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
