"""Tests for the cross-recipe held-out families (spec 2026-07-15)."""

import numpy as np

from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)  # the frozen organism world


def _authentic_law(vel, a, drag, dt):
    return (1.0 - drag * dt) * np.asarray(vel, float) + np.asarray(a, float) * dt


def test_g_cd_eps_zero_is_authentic_law():
    from itasorl.surrogate_l3_families import make_g_cd
    g = make_g_cd(eps=0.0, params=P)
    rng = np.random.default_rng(0)
    for _ in range(50):
        vel = rng.normal(size=2)
        a = rng.normal(size=2)
        want = _authentic_law(vel, a, 1.5, P.dt)
        got = g(vel, a, drag=None)  # drag ignored by contract
        assert np.array_equal(got, want)


def test_g_cd_eps_nonzero_differs_and_is_biased_decay():
    from itasorl.surrogate_l3_families import make_g_cd
    g = make_g_cd(eps=0.2, params=P)
    vel, a = np.array([0.3, -0.4]), np.array([0.0, 0.0])
    want = _authentic_law(vel, a, 1.5 * 1.2, P.dt)
    got = g(vel, a)
    assert np.array_equal(got, want)
    assert not np.array_equal(got, _authentic_law(vel, a, 1.5, P.dt))


def test_g_cd_requires_uniform_drag_world():
    import pytest
    from itasorl.surrogate_l3_families import make_g_cd
    with pytest.raises(ValueError):
        make_g_cd(eps=0.1, params=WorldParams())  # defaults: k_land=0.20 != k_water=0.60
