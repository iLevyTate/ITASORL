"""Regression tests for the Experiment B-v2 readout (experiment_b2.py) + stats.py.

The load-bearing guarantees here are the CONFOUND CONTROLS, not the science:
  - the matched-pair readout is BIT-IDENTICAL at L0 (drift off) - the keystone
    control that proves the readout manufactures no signal;
  - it DIVERGES once drift is on;
  - the leakage audit actually catches a reward confound;
  - the TOST equivalence test concludes equivalence only when it should.
All run on CPU with an untrained agent in well under a couple of seconds."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agent_ac import RecurrentActorCritic  # noqa: E402
from experiment_b2 import (  # noqa: E402
    RunningNorm,
    collect_pool,
    leakage_audit_b2,
    matched_pair_recurrent_rollout,
)
from stats import equivalence_test  # noqa: E402
from world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _agent_norm():
    from patch_of_earth import PatchOfEarthV0
    w = PatchOfEarthV0(P)
    od, ad = w.obs_spec.size, w.action_spec.size
    torch.manual_seed(0)
    agent = RecurrentActorCritic(od, ad, embed=16, hidden=8).train(False)
    return agent, RunningNorm(od).freeze()


def test_matched_pair_L0_is_bit_identical():
    """drift=0: authentic and surrogate branches must be identical to the bit -
    the readout adds no signal of its own (mirrors test_world's L0 control)."""
    agent, norm = _agent_norm()
    auth, surr = matched_pair_recurrent_rollout(agent, norm, P, 0.0, n_pairs=3, prefix_steps=4,
                                                branch_steps=6, ray_steps=RS, device="cpu")
    assert len(auth) >= 1
    for a, s in zip(auth, surr):
        assert np.array_equal(a["H"], s["H"]), "L0 branches diverged - readout is not confound-free"


def test_matched_pair_L2_diverges():
    """drift>0: the branches must differ (the artifact is the ONLY difference)."""
    agent, norm = _agent_norm()
    auth, surr = matched_pair_recurrent_rollout(agent, norm, P, 0.5, n_pairs=3, prefix_steps=4,
                                                branch_steps=6, ray_steps=RS, device="cpu")
    assert len(auth) >= 1
    assert any(not np.allclose(a["H"], s["H"]) for a, s in zip(auth, surr))


def test_collect_pool_returns_fixed_length():
    agent, norm = _agent_norm()
    H, spd = collect_pool(agent, norm, P, 0.0, 5, 6, "cpu", 12345, RS)
    assert H.ndim == 3 and H.shape[1] == 6 and H.shape[0] == len(spd)


def _ep(label, rsum):
    return {"H": np.zeros((3, 8), np.float32), "label": label, "speed": 0.0,
            "reward_sum": rsum, "length": 3, "lifetime": 1}


def test_leakage_audit_catches_reward_confound():
    auth = [_ep(0, 0.0) for _ in range(12)]
    surr = [_ep(1, 1.0) for _ in range(12)]   # reward perfectly predicts the label
    assert leakage_audit_b2(auth, surr)["clean"] is False


def test_leakage_audit_passes_when_balanced():
    rng = np.random.default_rng(0)
    auth = [_ep(0, float(rng.random())) for _ in range(12)]
    surr = [_ep(1, float(rng.random())) for _ in range(12)]  # same reward dist for both
    assert leakage_audit_b2(auth, surr)["clean"] is True


def test_tost_equivalence_has_teeth():
    near = equivalence_test([0.50, 0.49, 0.51, 0.50, 0.52])
    high = equivalence_test([0.80, 0.82, 0.79, 0.81, 0.78])
    assert near.equivalent is True
    assert high.equivalent is False
