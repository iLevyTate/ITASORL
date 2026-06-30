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

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import (  # noqa: E402
    RunningNorm,
    collect_pool,
    compute_gae,
    leakage_audit_b2,
    matched_pair_recurrent_rollout,
)
from itasorl.stats import equivalence_test  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _agent_norm():
    from itasorl.patch_of_earth import PatchOfEarthV0
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


def test_collect_pool_anchors_aligned():
    """return_anchors yields per-episode energy/food/drag arrays aligned with H - the
    ceiling controls the readout depends on."""
    agent, norm = _agent_norm()
    out = collect_pool(agent, norm, P, 0.45, 5, 6, "cpu", 222, RS, return_anchors=True)
    assert len(out) == 5
    H, spd, energy, food, drag = out
    k = H.shape[0]
    assert spd.shape == (k,) and energy.shape == (k,) and food.shape == (k,) and drag.shape == (k,)


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


def test_running_norm_matches_numpy_batch_stats():
    """Streamed in random-sized batches, the Welford running mean/var must match
    numpy's batch statistics over the whole stream. This is the contract the probe
    relies on: the frozen normalizer must reflect the true training-data moments.
    Tolerances are well above the observed float64 roundoff (~2e-7 mean, ~1e-6 var)
    but far below the O(var) error any wrong merge formula would produce."""
    rng = np.random.default_rng(0)
    dim = 4
    data = rng.normal(loc=[1.0, -3.0, 10.0, 0.0], scale=[0.5, 2.0, 5.0, 1.0],
                      size=(5000, dim))
    n = RunningNorm(dim)
    i = 0
    while i < len(data):                       # random batch sizes 1..16
        b = int(rng.integers(1, 17))
        n.update(data[i:i + b]); i += b
    assert np.allclose(n.mean, data.mean(0), atol=1e-5)
    assert np.allclose(n.var, data.var(0), rtol=1e-3, atol=1e-4)


def test_running_norm_batch_size_independent():
    """The parallel-variance merge is associative: feeding samples one-by-one must
    give the same mean/var as feeding them as a single batch (to float64 roundoff)."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(200, 4))
    one_by_one = RunningNorm(4)
    for row in data:
        one_by_one.update(row)
    single_batch = RunningNorm(4)
    single_batch.update(data)
    assert np.allclose(one_by_one.mean, single_batch.mean, atol=1e-10)
    assert np.allclose(one_by_one.var, single_batch.var, atol=1e-10)


def test_running_norm_freeze_halts_updates():
    """freeze() must stop all updates so the probe sees the SAME normalization the
    agent trained with - a later (e.g. surrogate-branch) update must be a no-op."""
    rng = np.random.default_rng(2)
    n = RunningNorm(4)
    n.update(rng.normal(size=(500, 4)))
    frozen = n.freeze()
    assert frozen is n                          # returns self for chaining
    snapshot = (n.mean.copy(), n.var.copy(), n.count)
    n.update(rng.normal(loc=100.0, size=(50, 4)))   # would shift moments if applied
    assert np.array_equal(n.mean, snapshot[0])
    assert np.array_equal(n.var, snapshot[1])
    assert n.count == snapshot[2]


def test_running_norm_normalizes_to_zero_mean_unit_std():
    """Applying the frozen normalizer to its own training data yields ~zero-mean,
    ~unit-std features - the standardization the downstream readout expects."""
    rng = np.random.default_rng(3)
    data = rng.normal(loc=[5.0, -2.0], scale=[3.0, 0.5], size=(4000, 2))
    n = RunningNorm(2)
    n.update(data)
    z = n.freeze()(data)
    assert np.abs(z.mean(0)).max() < 1e-3
    assert np.abs(z.std(0) - 1.0).max() < 1e-2


def _ref_gae(rewards, values, gamma, lam, bootstrap):
    """Independent textbook GAE for ONE unpadded episode (the reference oracle)."""
    adv = np.zeros(len(rewards), np.float64)
    gae, next_v = 0.0, bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_v = values[t]
    return adv


def test_compute_gae_no_padding_leakage():
    """A terminated episode shorter than Tmax must get the SAME advantages as the
    textbook per-episode GAE: the value at the padded slot must NOT leak into the
    last valid step via the GAE accumulator (the mask-leakage guard)."""
    gamma, lam = 0.99, 0.95
    # ep0: length 2, terminated (bootstrap 0); ep1: length 3, terminated. Tmax=3.
    reward = torch.tensor([[1.0, 2.0, 0.0], [1.0, 1.0, 1.0]])
    value = torch.tensor([[0.5, 0.3, 0.7], [0.2, 0.2, 0.2]])  # value[0,2]=0.7 is the padded slot
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    terminated = torch.tensor([1.0, 1.0])
    adv, ret = compute_gae(reward, value, mask, terminated, gamma, lam)
    ref0 = _ref_gae([1.0, 2.0], [0.5, 0.3], gamma, lam, bootstrap=0.0)
    ref1 = _ref_gae([1.0, 1.0, 1.0], [0.2, 0.2, 0.2], gamma, lam, bootstrap=0.0)
    assert np.allclose(adv[0, :2].numpy(), ref0, atol=1e-6)
    assert np.allclose(adv[1, :3].numpy(), ref1, atol=1e-6)
    assert adv[0, 2].item() == 0.0                                # padded step zeroed
    assert np.allclose(ret.numpy(), (adv + value).numpy())        # ret = adv + value


def test_compute_gae_truncation_bootstraps_last_value():
    """A truncated (not terminated) short episode bootstraps from its last in-episode
    value - not from the padded slot - when forming the final advantage."""
    gamma, lam = 0.99, 0.95
    reward = torch.tensor([[1.0, 2.0, 0.0], [1.0, 1.0, 1.0]])
    value = torch.tensor([[0.5, 0.3, 0.7], [0.2, 0.2, 0.2]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    terminated = torch.tensor([0.0, 1.0])                         # ep0 TRUNCATED
    adv, _ = compute_gae(reward, value, mask, terminated, gamma, lam)
    ref0 = _ref_gae([1.0, 2.0], [0.5, 0.3], gamma, lam, bootstrap=0.3)  # last in-ep value
    assert np.allclose(adv[0, :2].numpy(), ref0, atol=1e-6)


def test_tost_equivalence_has_teeth():
    near = equivalence_test([0.50, 0.49, 0.51, 0.50, 0.52])
    high = equivalence_test([0.80, 0.82, 0.79, 0.81, 0.78])
    assert near.equivalent is True
    assert high.equivalent is False
