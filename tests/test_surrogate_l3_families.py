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


def test_g_rff_deterministic_across_fits():
    from itasorl.surrogate_l3_families import fit_g_rff
    kw = dict(D=16, params=P, n_eps=4, steps=10)  # tiny data: determinism only
    g1, g2 = fit_g_rff(**kw), fit_g_rff(**kw)
    vel, a = np.array([0.2, -0.1]), np.array([0.5, 0.3])
    assert np.array_equal(g1(vel, a), g2(vel, a))


def test_g_rff_cannot_fit_the_linear_law_exactly():
    """World P's authentic law is exactly linear; the cosine basis must leave a
    systematic residual (this is the fingerprint's existence proof)."""
    from itasorl.surrogate_l3_families import fit_g_rff
    g = fit_g_rff(D=16, params=P, n_eps=4, steps=10)
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(100):
        vel, a = rng.normal(size=2), rng.normal(size=2)
        errs.append(np.abs(g(vel, a) - _authentic_law(vel, a, 1.5, P.dt)).max())
    assert max(errs) > 1e-8


def test_g_rff_output_contract():
    from itasorl.surrogate_l3_families import fit_g_rff
    g = fit_g_rff(D=16, params=P, n_eps=4, steps=10)
    out = g(np.array([0.1, 0.2]), np.array([0.3, 0.4]), drag=1.5)  # drag ignored
    assert out.shape == (2,) and out.dtype == np.float64


def test_gate0_candidates_labels_and_callables():
    from itasorl.surrogate_l3_families import gate0_candidates
    rff = list(gate0_candidates("rff", params=P, n_eps=2, steps=5))
    assert [lab for lab, _ in rff] == [
        ("D", 8), ("D", 16), ("D", 32), ("D", 64), ("D", 128)]
    cd = list(gate0_candidates("cd", params=P))
    assert [lab for lab, _ in cd] == [
        ("eps", 0.05), ("eps", 0.1), ("eps", 0.2), ("eps", 0.4), ("eps", 0.8)]
    out = cd[0][1](np.array([0.1, 0.1]), np.array([0.0, 0.0]))
    assert out.shape == (2,)
