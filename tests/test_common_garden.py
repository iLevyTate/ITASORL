"""Common-garden channel tests (spec 2026-07-14): differing prefix world,
identical authentic tail; the probe reads tail-only states.

Load-bearing guarantees mirror test_experiment_b2.py:
  - L0 (drift off): auth- and surr-prefix tails are BIT-IDENTICAL, so the
    channel manufactures no signal;
  - drift on: tails diverge (prefix history is the only difference);
  - on synthetic tails, a PERSISTENT group signal scores high on both windows
    while a REACTIVE (early-only) signal collapses to chance on the late window."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.agent_ac import RecurrentActorCritic  # noqa: E402
from itasorl.experiment_b2 import RunningNorm, cg_probe, common_garden_rollout  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4


def _agent_norm():
    from itasorl.patch_of_earth import PatchOfEarthV0
    w = PatchOfEarthV0(P)
    torch.manual_seed(0)
    agent = RecurrentActorCritic(w.obs_spec.size, w.action_spec.size, embed=16, hidden=8).train(False)
    return agent, RunningNorm(w.obs_spec.size).freeze()


def test_cg_L0_tails_bit_identical():
    agent, norm = _agent_norm()
    auth, surr = common_garden_rollout(agent, norm, P, 0.0, n_pairs=3, prefix_steps=4,
                                       tail_steps=6, ray_steps=RS, device="cpu")
    assert len(auth) >= 1
    for a, s in zip(auth, surr):
        assert np.array_equal(a, s), "L0 common-garden tails diverged - channel is not confound-free"


def test_cg_drift_tails_diverge():
    agent, norm = _agent_norm()
    # this call is also the guard on the surrogate-RNG key filter inside
    # common_garden_rollout: without it, set_state KeyErrors here at drift>0
    auth, surr = common_garden_rollout(agent, norm, P, 0.5, n_pairs=3, prefix_steps=4,
                                       tail_steps=6, ray_steps=RS, device="cpu")
    assert any(not np.array_equal(a, s) for a, s in zip(auth, surr)), \
        "drift-on prefixes left identical tails - snapshot carry is broken"


def _synthetic_tails(rng, n, T, hid, offset_fn):
    """Tails where group 1 carries offset_fn(t) added to one hidden unit."""
    auth = [rng.normal(0, 1, (T, hid)).astype(np.float32) for _ in range(n)]
    surr = []
    for _ in range(n):
        H = rng.normal(0, 1, (T, hid)).astype(np.float32)
        H[:, 0] += np.array([offset_fn(t) for t in range(T)], np.float32)
        surr.append(H)
    return auth, surr


def test_cg_probe_persistent_vs_reactive():
    rng = np.random.default_rng(0)
    T = 24
    pers_a, pers_s = _synthetic_tails(rng, 60, T, 8, lambda t: 3.0)          # persistent
    reac_a, reac_s = _synthetic_tails(rng, 60, T, 8, lambda t: 3.0 if t < 4 else 0.0)  # reactive
    pers = cg_probe(pers_a, pers_s, late_k=8, seed=0)
    reac = cg_probe(reac_a, reac_s, late_k=8, seed=0)
    assert pers["cg_tail_target"] > 0.9 and pers["cg_latetail_target"] > 0.9
    assert abs(reac["cg_latetail_target"] - 0.5) < 0.20, "late window must not see an early-only signal"
