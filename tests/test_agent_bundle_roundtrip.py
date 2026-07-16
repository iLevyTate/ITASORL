"""save_agent_bundle -> load_agent_bundle must be a bit-exact roundtrip."""

import numpy as np
import torch

from itasorl.agent_ac import RecurrentActorCritic
from itasorl.experiment_b2 import RunningNorm, load_agent_bundle, save_agent_bundle


def test_roundtrip_bit_exact(tmp_path):
    torch.manual_seed(0)
    agent = RecurrentActorCritic(20, 4, embed=8, hidden=8, world_model=True)
    norm = RunningNorm(20)
    norm.mean = np.arange(20, dtype=np.float64)
    norm.var = np.full(20, 2.0)
    norm.count = 123.0
    path = str(tmp_path / "agent.pt")
    save_agent_bundle(path, agent, norm)
    agent2, norm2 = load_agent_bundle(path, device="cpu")

    sd1, sd2 = agent.state_dict(), agent2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k
    assert np.array_equal(norm.mean, norm2.mean)
    assert np.array_equal(norm.var, norm2.var)
    assert norm.count == norm2.count
    assert agent2.training is False  # frozen for deterministic readout
    assert norm2.frozen is True  # reloaded norm must never mutate

    obs = torch.zeros(1, 20)
    prev = torch.zeros(1, 4)
    h1 = agent.initial_state(1, "cpu")
    h2 = agent2.initial_state(1, "cpu")
    agent.train(False)
    with torch.no_grad():
        _, act1, _, _, _ = agent.act(obs, prev, h1, deterministic=True)
        _, act2, _, _, _ = agent2.act(obs, prev, h2, deterministic=True)
    assert torch.equal(act1, act2)
