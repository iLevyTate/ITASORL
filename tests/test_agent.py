"""Regression tests for the recurrent world model's open-loop rollout.

These lock in the rollout API used by run_expB_gap.py / run_expB_kstep.py: the
teacher-forced prefix must match the plain forward pass, and a full-length
context must be identical to it (no open-loop steps), for both output
conventions (absolute and delta).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _model_and_data(seed=0, batch=3, steps=7, obs_dim=5, act_dim=2, hidden=16):
    from agent import RecurrentWorldModel

    torch.manual_seed(seed)
    model = RecurrentWorldModel(obs_dim, act_dim, embed=8, hidden=hidden)
    rng = np.random.default_rng(seed)
    obs = torch.tensor(rng.standard_normal((batch, steps, obs_dim)).astype("float32"))
    act = torch.tensor(rng.standard_normal((batch, steps, act_dim)).astype("float32"))
    return model, obs, act


def test_rollout_teacher_forced_prefix_matches_forward():
    model, obs, act = _model_and_data()
    context = 4
    preds_tf, _ = model(obs, act)
    preds_ro = model.forward_rollout(obs, act, context)
    # for t < context both feed real observations -> identical predictions
    assert torch.allclose(preds_ro[:, :context], preds_tf[:, :context], atol=1e-6)


def test_rollout_full_context_equals_forward():
    model, obs, act = _model_and_data()
    steps = obs.shape[1]
    preds_tf, _ = model(obs, act)
    preds_ro = model.forward_rollout(obs, act, steps)  # no open-loop steps
    assert torch.allclose(preds_ro, preds_tf, atol=1e-6)


def test_rollout_loss_is_finite_in_both_modes():
    model, obs, act = _model_and_data()
    model.delta = False
    assert torch.isfinite(model.rollout_loss(obs, act, 4))
    model.delta = True
    assert torch.isfinite(model.rollout_loss(obs, act, 4))


def test_open_loop_ignores_future_observations():
    """Past the context window the rollout must not peek at real observations:
    perturbing obs after `context` leaves the open-loop predictions unchanged."""
    model, obs, act = _model_and_data()
    context = 4
    base = model.forward_rollout(obs, act, context)
    perturbed = obs.clone()
    perturbed[:, context:] += 5.0  # corrupt the "future" observations
    after = model.forward_rollout(perturbed, act, context)
    assert torch.allclose(base[:, context:], after[:, context:], atol=1e-6)
