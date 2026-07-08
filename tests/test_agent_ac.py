"""Regression tests for the Experiment B-v2 recurrent actor-critic (agent_ac.py).

These lock the agent CONTRACT - action ranges, log-prob/value finiteness, the
shape of the probed recurrent state, and that both the RL and the world-model
objectives produce finite, differentiable losses. They run on CPU in well under a
second; they do NOT test the scientific claim (that needs the full run)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402

OBS, ACT, B, T = 12, 5, 4, 7


def _agent(world_model=True):
    torch.manual_seed(0)
    return RecurrentActorCritic(OBS, ACT, embed=16, hidden=8, world_model=world_model)


def test_act_returns_valid_env_action_and_finite_logp():
    a = _agent()
    obs = torch.randn(B, OBS)
    prev = torch.zeros(B, ACT)
    h = a.initial_state(B, "cpu")
    raw, env, logp, value, h2 = a.act(obs, prev, h)
    assert env.shape == (B, ACT) and raw.shape == (B, ACT)
    # env action ranges: thrust in [0,1], turn in [-1,1], eat/drink/emit in {0,1}
    assert (env[:, 0] >= 0).all() and (env[:, 0] <= 1).all()
    assert (env[:, 1] >= -1).all() and (env[:, 1] <= 1).all()
    binaries = env[:, 2:]
    assert torch.isin(binaries, torch.tensor([0.0, 1.0])).all()
    assert torch.isfinite(logp).all() and logp.shape == (B,)
    assert torch.isfinite(value).all() and value.shape == (B,)
    assert h2.shape == (B, a.hidden)


def test_env_action_bounded_for_extreme_latents():
    """The sigmoid/tanh squash must keep the env action inside the world's actuator
    ranges (thrust [0,1], turn [-1,1]) even for pathological raw latents far outside the
    typical sample range, and stay finite - no NaN/inf from saturation. If an extreme
    actor output escaped these bounds the physics integrator could silently NaN mid-run.
    Binaries pass through unchanged, preserving the caller's {0,1} guarantee."""
    n_bin = ACT - 2
    for mag in (50.0, 1e3, 1e4, 1e8):
        raw_c = torch.tensor([[mag, -mag], [-mag, mag]])          # both signs on both dims
        b = torch.tensor([[1.0] * n_bin, [0.0] * n_bin])
        env = RecurrentActorCritic.to_env_action(raw_c, b)
        assert torch.isfinite(env).all(), f"non-finite env action at latent magnitude {mag}"
        assert (env[:, 0] >= 0).all() and (env[:, 0] <= 1).all(), f"thrust out of [0,1] at {mag}"
        assert (env[:, 1] >= -1).all() and (env[:, 1] <= 1).all(), f"turn out of [-1,1] at {mag}"
        assert torch.isin(env[:, 2:], torch.tensor([0.0, 1.0])).all(), f"binaries not in {{0,1}} at {mag}"


def test_deterministic_act_is_repeatable():
    a = _agent()
    obs, prev, h = torch.randn(B, OBS), torch.zeros(B, ACT), a.initial_state(B, "cpu")
    _, e1, _, _, _ = a.act(obs, prev, h, deterministic=True)
    _, e2, _, _, _ = a.act(obs, prev, h, deterministic=True)
    assert torch.allclose(e1, e2)


def test_states_for_probe_shape_and_no_grad():
    a = _agent()
    obs = torch.randn(B, T, OBS)
    act_in = torch.zeros(B, T, ACT)
    H = a.states_for_probe(obs, act_in)
    assert H.shape == (B, T, a.hidden)
    assert not H.requires_grad


def test_score_actions_differentiable():
    a = _agent()
    obs = torch.randn(B, T, OBS)
    act_in = torch.zeros(B, T, ACT)
    raw = torch.randn(B, T, ACT)
    raw[..., 2:] = (raw[..., 2:] > 0).float()  # binaries must be 0/1 for Bernoulli log-prob
    logp, value, ent, states = a.score_actions(obs, act_in, raw, a.initial_state(B, "cpu"))
    assert logp.shape == (B, T) and value.shape == (B, T) and states.shape == (B, T, a.hidden)
    loss = logp.mean() + value.mean() + ent.mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())


def test_world_model_loss_is_finite_and_differentiable():
    a = _agent(world_model=True)
    obs = torch.randn(B, T, OBS)
    act_in = torch.zeros(B, T, ACT)
    env_act = torch.zeros(B, T, ACT)
    mask = torch.ones(B, T)
    loss, states = a.world_model_loss(obs, act_in, env_act, mask, a.initial_state(B, "cpu"))
    assert torch.isfinite(loss) and states.shape == (B, T, a.hidden)
    loss.backward()


def test_world_model_can_be_disabled():
    a = _agent(world_model=False)
    assert not hasattr(a, "decoder")
