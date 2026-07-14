"""save/load round-trip for trained-agent bundles (spec 2026-07-14: agents were
never persisted, which forced a full retrain when new eval channels appeared)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.experiment_b2 import (  # noqa: E402
    load_agent_bundle, save_agent_bundle, untrained_agent,
)
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def test_bundle_round_trip(tmp_path):
    agent, norm = untrained_agent(P, 0.45, 4, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    path = tmp_path / "agent.pt"
    save_agent_bundle(str(path), agent, norm)
    agent2, norm2 = load_agent_bundle(str(path), device="cpu")
    obs = np.linspace(-1, 1, agent.obs_dim)[None]        # (1, obs_dim)
    x = torch.as_tensor(norm(obs), dtype=torch.float32)
    x2 = torch.as_tensor(norm2(obs), dtype=torch.float32)
    assert torch.allclose(x, x2)
    h = agent.initial_state(1, "cpu")
    prev = torch.zeros(1, agent.act_dim)
    _, a1, _, _, _ = agent.act(x, prev, h, deterministic=True)
    _, a2, _, _, _ = agent2.act(x2, prev, agent2.initial_state(1, "cpu"), deterministic=True)
    assert torch.allclose(a1, a2), "reloaded agent must act identically"
    assert agent2.world_model is True and agent2.sysid_aux is False
