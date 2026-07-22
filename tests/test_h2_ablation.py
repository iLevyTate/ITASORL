"""H2 substrate-grounding ablation - unit tests for the graded velocity law.

Covers the GradedGMotion blend used by scripts/run_expH2_ablation.py:
alpha=1 is bit-identical to the frozen GMotion, alpha=0 is the exact authentic
law, and intermediate alpha is the convex interpolation between them.
"""

import numpy as np

from itasorl.surrogate_l3 import GradedGMotion, train_g_motion
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
DT = P.dt  # 0.05 - GradedGMotion must use this to match the world's step


class _StubG:
    """A stand-in for GMotion with a fixed, distinctive output so the blend is checkable
    independently of any trained net."""

    def __init__(self, out):
        self.out = np.asarray(out, float)

    def __call__(self, vel, a, drag=None):
        return self.out.copy()


def _authentic(vel, a, drag, dt=DT):
    vel = np.asarray([vel[0], vel[1]], float)
    a = np.asarray([a[0], a[1]], float)
    return (1.0 - drag * dt) * vel + a * dt


def test_endpoints_and_blend_with_stub():
    g = _StubG([9.0, -9.0])
    vel, a, drag = [0.3, -0.2], [1.0, 0.5], 1.5
    true = _authentic(vel, a, drag)

    g1 = GradedGMotion(g, 1.0, DT)
    g0 = GradedGMotion(g, 0.0, DT)
    ghalf = GradedGMotion(g, 0.5, DT)

    # alpha=1 -> exactly g (bit-identical, the determinism-gate requirement)
    assert np.array_equal(g1(vel, a, drag), g(vel, a, drag))
    # alpha=0 -> exactly the authentic law
    assert np.allclose(g0(vel, a, drag), true)
    # alpha=0.5 -> convex midpoint
    assert np.allclose(ghalf(vel, a, drag), 0.5 * true + 0.5 * np.array([9.0, -9.0]))


def test_blend_is_monotone_convex_in_alpha():
    g = _StubG([5.0, 5.0])
    vel, a, drag = [0.1, 0.4], [-0.3, 0.2], 1.5
    true = _authentic(vel, a, drag)
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    outs = [GradedGMotion(g, al, DT)(vel, a, drag) for al in alphas]
    for al, out in zip(alphas, outs):
        assert np.allclose(out, (1 - al) * true + al * np.array([5.0, 5.0]))
    # each component moves monotonically from the authentic value toward g's value
    comp0 = [o[0] for o in outs]
    assert comp0 == sorted(comp0) or comp0 == sorted(comp0, reverse=True)


def test_alpha1_matches_real_trained_gmotion_bitwise():
    """With a real (small, fast) trained GMotion, GradedGMotion(alpha=1) must reproduce
    the net's output bit-for-bit, and alpha=0 must reproduce the authentic law - the two
    anchors the ablation's integrity gate and L0 anchor rely on."""
    g = train_g_motion(hidden=4, n_eps=4, steps=8, epochs=5, seed=0, params=P)
    rng = np.random.default_rng(0)
    for _ in range(20):
        vel = rng.normal(size=2)
        a = rng.normal(size=2)
        drag = 1.5
        assert np.array_equal(GradedGMotion(g, 1.0, DT)(vel, a, drag), g(vel, a, drag))
        assert np.allclose(GradedGMotion(g, 0.0, DT)(vel, a, drag), _authentic(vel, a, drag))
