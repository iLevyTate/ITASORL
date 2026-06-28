"""Regression tests for the ITASORL world contract and surrogate ladder.

The keystone guarantees (spec sec. 9/11/12): a byte-identical obs format between
authentic and surrogate, exact snapshot/restore, an L0 branch that is identical
to the authentic branch (the chance control), and an L1 branch that differs from
authentic *only* by the documented observation quantization.
"""

import numpy as np

from patch_of_earth import PatchOfEarthV0, first_config_obs_spec
from world import (
    DEFAULT_OBS_SPEC,
    L0Identity,
    L1Discretize,
    Level,
    SeedBundle,
    matched_pair_rollout,
)

CONST_ACTION = np.array([0.5, 0.1, 1.0, 0.0, 0.0], dtype=np.float32)


def _const_policy(obs):
    # Obs-independent on purpose: L1 then cannot cause behavioral divergence, so
    # the only authentic/surrogate difference is the observation quantization.
    return CONST_ACTION


def _make_world():
    w = PatchOfEarthV0()
    w.ray_steps = 4  # light raymarch keeps the tests fast
    return w


def _seeds():
    return SeedBundle(world=7, weather=8, ecology=9)


# --- obs format contract (spec sec. 9) --------------------------------------

def test_obs_spec_size():
    assert DEFAULT_OBS_SPEC.size == 24 * 5 + 4 * 3 + 14


def test_mask_preserves_length_and_format_hash():
    masked = first_config_obs_spec()  # smell channel disabled
    assert masked.size == DEFAULT_OBS_SPEC.size  # length preserved
    # masking changes the mask, not the *format* identity (names/sizes/order/dtype)
    assert masked.identity_hash() == DEFAULT_OBS_SPEC.identity_hash()


def test_masked_channel_is_zero_filled_not_removed():
    masked = first_config_obs_spec()
    smell = masked.slices()["smell"]
    v = masked.assemble({"smell": np.ones(4 * 3, dtype=np.float32)})
    assert np.all(v[smell] == 0.0)


# --- snapshot / restore determinism (spec sec. 11/12) -----------------------

def test_get_set_state_roundtrip_is_exact():
    live = _make_world()
    live.reset(_seeds())
    for _ in range(5):
        live.step(CONST_ACTION)
    snap = live.get_state()

    restored = _make_world()
    restored.reset(_seeds())
    restored.set_state(snap)

    a = live.step(CONST_ACTION)
    b = restored.step(CONST_ACTION)
    np.testing.assert_array_equal(a.obs, b.obs)
    assert a.reward == b.reward
    assert a.terminated == b.terminated


# --- matched-pair confound control (spec sec. 11) ---------------------------

def test_matched_pair_L0_is_bit_identical():
    pe = matched_pair_rollout(
        make_world=_make_world,
        make_surrogate=L0Identity,
        seeds=_seeds(),
        policy=_const_policy,
        prefix_steps=5,
        branch_steps=8,
        pair_id=0,
        rng=np.random.default_rng(0),
    )
    assert pe.level == Level.L0
    assert len(pe.authentic) == len(pe.surrogate) == 8
    for a, s in zip(pe.authentic, pe.surrogate):
        np.testing.assert_array_equal(a.obs, s.obs)
        assert a.reward == s.reward


def test_matched_pair_L1_only_quantizes_observation():
    delta = 1.0 / 32
    pe = matched_pair_rollout(
        make_world=_make_world,
        make_surrogate=lambda base: L1Discretize(base, delta=delta),
        seeds=_seeds(),
        policy=_const_policy,
        prefix_steps=5,
        branch_steps=6,
        pair_id=1,
        rng=np.random.default_rng(1),
    )
    assert pe.level == Level.L1
    for a, s in zip(pe.authentic, pe.surrogate):
        np.testing.assert_allclose(s.obs, np.round(a.obs / delta) * delta, atol=1e-6)
