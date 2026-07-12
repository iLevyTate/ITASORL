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
    pooled_readout,
)
from itasorl.stats import equivalence_test  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4
# Every device the box exposes: CPU always, CUDA only when present (skipped in CI).
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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
    """return_anchors yields per-episode energy/food/drag/reward arrays aligned with H - the
    ceiling controls plus the reward channel the pooled leakage audit depends on."""
    agent, norm = _agent_norm()
    out = collect_pool(agent, norm, P, 0.45, 5, 6, "cpu", 222, RS, return_anchors=True)
    assert len(out) == 7
    H, spd, energy, food, drag, reward, traces = out
    k = H.shape[0]
    assert spd.shape == (k,) and energy.shape == (k,) and food.shape == (k,) and drag.shape == (k,)
    assert reward.shape == (k,) and np.isfinite(reward).all()


def test_collect_pool_traces_match_anchor_means():
    """The per-timestep behavior traces (speed/energy/food/drag) must be the very
    accumulators whose means are the anchors - the behavior-mediation audit's
    per-timestep control depends on this alignment with H."""
    agent, norm = _agent_norm()
    out = collect_pool(agent, norm, P, 0.45, 5, 6, "cpu", 222, RS, return_anchors=True)
    H, spd, energy, food, drag, reward, traces = out
    k, steps = H.shape[0], H.shape[1]
    assert traces.shape == (k, steps, 4)
    np.testing.assert_allclose(traces[:, :, 0].mean(1), spd, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(traces[:, :, 1].mean(1), energy, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(traces[:, :, 2].mean(1), food, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(traces[:, :, 3].mean(1), drag, rtol=1e-5, atol=1e-6)


def test_pooled_readout_dump_contains_traces(tmp_path):
    """--dump-states must persist bta/bts (k, steps, 4) so the per-timestep
    behavior control can run offline on new dumps."""
    agent, norm = _agent_norm()
    p = tmp_path / "dump.npz"
    pooled_readout(agent, norm, P, 0.45, n_eps=6, steps=5, ray_steps=RS,
                   device="cpu", dump_path=str(p))
    with np.load(p) as z:
        assert z["bta"].shape[1:] == (5, 4)
        assert z["bts"].shape[1:] == (5, 4)
        assert z["bta"].shape[0] == z["Ha"].shape[0]
        assert z["bts"].shape[0] == z["Hs"].shape[0]


def test_collect_pool_excludes_early_deaths(monkeypatch):
    """Under harsh metabolism some episodes die before `steps`; those must be DROPPED,
    never truncated or padded, so the pool holds only full-length survivors. A shorter
    dead episode leaking in would silently bias the B-v2 readout. Here a lowered starting
    energy forces a MIX of deaths and survivors."""
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "SURVIVAL_METAB",
                        {"E0": 0.2, "basal_E": 0.4, "Hyd0": 8.0, "basal_Hyd": 0.005})
    agent, norm = _agent_norm()
    n_eps, steps = 10, 12
    H, spd = collect_pool(agent, norm, P, 0.45, n_eps, steps, "cpu", 999, RS)
    assert 0 < H.shape[0] < n_eps, "expected a MIX of early deaths and full-length survivors"
    assert H.shape[1] == steps and all(row.shape[0] == steps for row in H), \
        "a non-full-length (dead/truncated) episode leaked into the pool"
    assert H.shape[0] == spd.shape[0], "H and speeds misaligned after dropping early deaths"


def test_pooled_readout_too_few_survivors_guard(monkeypatch):
    """When fewer than 5 episodes survive in EITHER pool, the readout must return a
    well-formed all-NaN result flagged `too_few_survivors` - never crash, never emit a
    spurious AUROC from a handful of episodes. Guards the harsh-metabolism edge where the
    pool nearly empties. A `collect_pool` stub returns 3 (<5) survivors deterministically."""
    import itasorl.experiment_b2 as b2

    def _tiny_pool(*args, return_anchors=False, **kw):
        steps, hid, k = args[5], args[0].hidden, 3           # 3 < 5 -> guard must fire
        H, s = np.zeros((k, steps, hid), np.float32), np.zeros(k)
        z = np.zeros(k)
        bt = np.zeros((k, steps, 4), np.float32)
        return (H, s, z, z, z, z, bt) if return_anchors else (H, s)

    monkeypatch.setattr(b2, "collect_pool", _tiny_pool)
    agent, norm = _agent_norm()
    out = pooled_readout(agent, norm, P, 0.45, n_eps=10, steps=6, ray_steps=RS, device="cpu")
    assert out["too_few_survivors"] is True
    for key in ("target", "target_lo", "target_hi", "target_var", "target_full",
                "selectivity", "speed", "anchor_energy", "ceiling_drag", "pool_reward_leak"):
        assert np.isnan(out[key]), f"{key} must be NaN when the pool is too small"
    assert out["pool_leak_clean"] is False   # cannot certify clean with too few survivors


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


def _stub_pool(reward_auth, reward_surr, seed=0):
    """Build a collect_pool stub whose two pools (drift 0.0 vs >0) carry prescribed
    per-episode reward, so we can drive pooled_readout's leakage audit deterministically.
    H is random noise (target ~ chance); only the reward channel is controlled."""
    rng = np.random.default_rng(seed)

    def _pool(*args, return_anchors=False, **kw):
        drift, steps, hid = args[3], args[5], args[0].hidden
        reward = np.asarray(reward_auth if drift == 0.0 else reward_surr, float)
        k = len(reward)
        H = rng.standard_normal((k, steps, hid)).astype(np.float32)
        z = np.zeros(k)
        bt = np.zeros((k, steps, 4), np.float32)
        return (H, z, z, z, z, reward, bt) if return_anchors else (H, z)
    return _pool


def test_pooled_readout_flags_reward_confound(monkeypatch):
    """The headline pooled endpoint must now catch a reward confound: if summed reward
    perfectly separates the worlds, pool_leak_clean is False and pool_reward_leak is high.
    This is the control that was missing on the pooled path (only matched-pair had it)."""
    import itasorl.experiment_b2 as b2
    k = 20
    monkeypatch.setattr(b2, "collect_pool",
                        _stub_pool(np.zeros(k), np.ones(k)))   # reward = world label
    agent, norm = _agent_norm()
    out = pooled_readout(agent, norm, P, 0.45, n_eps=k, steps=6, ray_steps=RS, device="cpu")
    assert out["too_few_survivors"] is False
    assert out["pool_leak_clean"] is False
    assert out["pool_reward_leak"] > 0.9
    assert out["deaths_auth"] == 0 and out["deaths_surr"] == 0   # stub returns all survivors
    assert out["n_auth"] == k and out["n_surr"] == k


def test_pooled_readout_reward_audit_clean_when_balanced(monkeypatch):
    """When reward carries no world signal (identical values in both pools), the pooled
    audit certifies clean and reward_leak sits at chance - so a real clean result means the
    pooled target reads the dynamics artifact, not a reward confound."""
    import itasorl.experiment_b2 as b2
    k = 40
    shared = np.linspace(-1.0, 1.0, k)                          # same reward in both worlds
    monkeypatch.setattr(b2, "collect_pool", _stub_pool(shared, shared.copy()))
    agent, norm = _agent_norm()
    out = pooled_readout(agent, norm, P, 0.45, n_eps=k, steps=6, ray_steps=RS, device="cpu")
    assert out["pool_leak_clean"] is True
    assert abs(out["pool_reward_leak"] - 0.5) < 0.1


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


# ---- variance-signature probe features (PR1) ----
from itasorl.experiment_b import (  # noqa: E402
    episode_features,
    episode_features_full,
    episode_features_var,
    probe_auroc,
)


def test_episode_feature_builder_shapes():
    """var doubles hidden (std ++ jerk); full is level ++ var (4x hidden)."""
    H = np.random.default_rng(0).normal(size=(7, 5, 6)).astype(np.float32)
    assert episode_features(H).shape == (7, 12)
    assert episode_features_var(H).shape == (7, 12)
    assert episode_features_full(H).shape == (7, 24)


def test_episode_features_var_single_step_is_finite():
    """steps==1 has no step-to-step delta; the jerk block must be zeros, not a crash."""
    H = np.random.default_rng(1).normal(size=(4, 1, 3)).astype(np.float32)
    Xv = episode_features_var(H)
    assert Xv.shape == (4, 6)
    assert np.isfinite(Xv).all()
    assert np.allclose(Xv[:, 3:], 0.0)                 # the mean|delta| half is zero


def test_volatility_signature_needs_dispersion_features():
    """The load-bearing PR1 claim: when two classes share the same per-episode LEVEL
    but differ only in within-episode VOLATILITY, the level probe [mean h, final h]
    is near chance while the dispersion probe separates them almost perfectly. This is
    exactly the authentic (constant drag) vs surrogate (drifting drag) contrast, and it
    proves the new features measure what we say they do."""
    rng = np.random.default_rng(0)
    n, steps, hid = 60, 24, 4
    lvl0 = rng.normal(size=(n, 1, hid))
    lvl1 = rng.normal(size=(n, 1, hid))                # SAME level distribution as class 0
    H0 = np.repeat(lvl0, steps, axis=1) + rng.normal(scale=1e-3, size=(n, steps, hid))  # ~constant
    H1 = np.repeat(lvl1, steps, axis=1) + rng.normal(scale=1.0, size=(n, steps, hid))   # volatile
    H = np.concatenate([H0, H1]).astype(np.float32)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    level_auc = probe_auroc(episode_features(H), y)
    var_auc = probe_auroc(episode_features_var(H), y)
    full_auc = probe_auroc(episode_features_full(H), y)
    assert level_auc < 0.70, f"level probe should be near chance, got {level_auc:.3f}"
    assert var_auc > 0.85, f"dispersion probe should separate volatility, got {var_auc:.3f}"
    assert var_auc - level_auc > 0.20                  # the readout gap is the finding
    assert full_auc > 0.85                             # level ++ var retains the signal


# ---- sysid-aux positive-control head (PR2) ----
def test_sysid_head_absent_by_default():
    """The ceiling control must be opt-in: a default agent has no sysid head, so the
    headline readout-not-reward runs are never accidentally supervised on drag."""
    od, ad = 10, 5
    torch.manual_seed(0)
    plain = RecurrentActorCritic(od, ad, embed=16, hidden=8)
    assert not plain.sysid_aux and not hasattr(plain, "sysid_head")
    ceil = RecurrentActorCritic(od, ad, embed=16, hidden=8, sysid_aux=True)
    assert ceil.sysid_aux and hasattr(ceil, "sysid_head")


def test_predict_sysid_shape():
    """predict_sysid maps recurrent states (B,T,H) -> a per-step scalar (B,T)."""
    od, ad = 10, 5
    torch.manual_seed(0)
    agent = RecurrentActorCritic(od, ad, embed=16, hidden=8, sysid_aux=True)
    states = torch.zeros(3, 6, 8)
    out = agent.predict_sysid(states)
    assert out.shape == (3, 6)


def test_collect_pool_batch_exposes_drift_target():
    """collect_episodes_ac must expose a per-step drift_w target aligned with the mask,
    so the sysid-aux loss regresses h_t onto the drag the agent actually experienced."""
    from itasorl.experiment_b2 import collect_episodes_ac
    agent, norm = _agent_norm()
    b = collect_episodes_ac(agent, norm, P, 0.45, 4, 6, "cpu", 4321, RS)
    assert "drift_w" in b
    assert b["drift_w"].shape == b["mask"].shape
    assert torch.isfinite(b["drift_w"]).all()


# ---- B-v3 per-episode drag-regime coupling (PR3) ----
def _drift_trace(mode, drift_sigma, steps=30, seed=7):
    """Run one episode with a fixed action and return the per-step _drift_w trace."""
    from itasorl.patch_of_earth import PatchOfEarthV0
    from itasorl.world import SeedBundle
    w = PatchOfEarthV0(P, drift_sigma=drift_sigma, drift_mode=mode)
    w.ray_steps = RS
    w.reset(SeedBundle(world=seed, weather=seed + 1, ecology=seed + 2))
    trace = []
    for _ in range(steps):
        w.step(np.array([0.8, 0.1, 0.0, 0.0, 0.0], np.float32))
        trace.append(float(w._drift_w))
    return np.array(trace)


def test_regime_mode_is_constant_within_episode():
    """regime surrogate holds a SINGLE per-episode drag offset (identifiable), unlike the
    ar1 surrogate which wanders step-to-step."""
    regime = _drift_trace("regime", 0.45)
    ar1 = _drift_trace("ar1", 0.45)
    assert np.ptp(regime) < 1e-9 and regime[0] != 0.0          # constant, non-zero regime
    assert np.ptp(ar1) > 1e-6                                   # ar1 genuinely varies


def test_regime_offset_in_expected_band():
    """The per-episode offset is drift_sigma * U(0.5, 1.5): centered on drift_sigma and
    bounded away from 0, a clear persistent regime shift."""
    vals = [_drift_trace("regime", 0.45, steps=1, seed=s)[0] for s in range(40)]
    vals = np.array(vals)
    assert (vals >= 0.45 * 0.5 - 1e-6).all() and (vals <= 0.45 * 1.5 + 1e-6).all()
    assert vals.std() > 0.0                                     # varies across episodes


def test_regime_mode_authentic_is_unperturbed():
    """drift_sigma=0 must reproduce the exact authentic world in regime mode too (L0)."""
    assert np.allclose(_drift_trace("regime", 0.0), 0.0)


def test_matched_pair_L2_diverges_regime_mode(monkeypatch):
    """Regime mode: the surrogate branch must carry its per-episode drag offset past the
    snapshot restore, so matched-pair branches genuinely diverge. Regression for the
    B-v3 degeneracy where set_state clobbered the reset-drawn offset with the prefix's
    0.0 and mp_target collapsed to exactly 0.5 in every cell (fullruns/07022026)."""
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "DRIFT_MODE", "regime")
    agent, norm = _agent_norm()
    auth, surr = matched_pair_recurrent_rollout(agent, norm, P, 0.5, n_pairs=3, prefix_steps=4,
                                                branch_steps=6, ray_steps=RS, device="cpu")
    assert len(auth) >= 1
    assert any(not np.allclose(a["H"], s["H"]) for a, s in zip(auth, surr)), \
        "regime-mode matched-pair branches are identical - the drag offset was lost"


def test_matched_pair_L0_bit_identical_regime_mode(monkeypatch):
    """Regime mode keystone control: with drift OFF the branches must remain identical
    to the bit, exactly as in ar1 mode - the fix must not manufacture signal at L0."""
    import itasorl.experiment_b2 as b2
    monkeypatch.setattr(b2, "DRIFT_MODE", "regime")
    agent, norm = _agent_norm()
    auth, surr = matched_pair_recurrent_rollout(agent, norm, P, 0.0, n_pairs=3, prefix_steps=4,
                                                branch_steps=6, ray_steps=RS, device="cpu")
    assert len(auth) >= 1
    for a, s in zip(auth, surr):
        assert np.array_equal(a["H"], s["H"]), "L0 branches diverged under regime mode"


@pytest.mark.parametrize("device", DEVICES)
def test_matched_pair_L0_bit_identical_on_device(device):
    """The keystone L0 confound control must hold on EVERY device, not just CPU.
    A GPU kernel that broke branch bit-identity would silently manufacture a
    world-identity signal on GPU runs only - a confound that could masquerade as
    a positive result. Guards the CUDA path used by every `fullruns/` GPU run."""
    agent, norm = _agent_norm()
    agent = agent.to(device)
    auth, surr = matched_pair_recurrent_rollout(agent, norm, P, 0.0, n_pairs=3, prefix_steps=4,
                                                branch_steps=6, ray_steps=RS, device=device)
    assert len(auth) >= 1
    for a, s in zip(auth, surr):
        assert np.array_equal(a["H"], s["H"]), f"L0 branches diverged on {device}"


@pytest.mark.parametrize("device", DEVICES)
def test_readout_states_deterministic_across_runs(device):
    """Same device + same seeds => bit-identical recurrent states across a repeat
    run, so every downstream probe AUROC is identical by construction. This is the
    invariant the replication gap (lab 0.595 vs Colab 0.523) is measured against:
    within a device the pipeline is deterministic, so any surviving cross-run drift
    is attributable to device / seed / code differences, not our readout. Cross-DEVICE
    equality is deliberately NOT asserted (CPU vs CUDA float reductions differ)."""
    agent, norm = _agent_norm()
    agent = agent.to(device)
    H1, s1 = collect_pool(agent, norm, P, 0.45, 6, 6, device, 4242, RS)
    H2, s2 = collect_pool(agent, norm, P, 0.45, 6, 6, device, 4242, RS)
    assert H1.shape[0] >= 1, "no survivors - determinism check is vacuous"
    assert np.array_equal(H1, H2), f"recurrent states not reproducible on {device}"
    assert np.array_equal(s1, s2), f"speeds not reproducible on {device}"


def test_summary_parses_strongest_drift_cell():
    """The run summary must report the strongest (test) drift cell, not the drift-0
    control. Regression for the 07022026 headline that said 'at chance' (0.52, the
    L0 control) for a run whose test-drift survival target was 0.633."""
    from itasorl.results_io import parse_step_metrics
    log = (
        "drift=0.00 block\n"
        "survival   PRIMARY pool target = 0.520+/-0.040   speed(+ctrl) = 0.961\n"
        "predictor  PRIMARY pool target = 0.537+/-0.070   speed(+ctrl) = 0.903\n"
        "drift=0.45 block\n"
        "survival   PRIMARY pool target = 0.633+/-0.057   speed(+ctrl) = 0.959\n"
        "predictor  PRIMARY pool target = 0.536+/-0.059   speed(+ctrl) = 0.911\n"
        "At strongest drift=0.45: survival pooled target |dev|=0.133\n"
    )
    m = parse_step_metrics("expB2", log)
    assert m["survival_pool_target_mean"] == 0.633
    assert m["survival_pool_target_std"] == 0.057
    assert m["predictor_pool_target_mean"] == 0.536
    assert m["organism_encodes_world"] == "weak"
    assert m["survival_deviation_from_chance"] == 0.133
