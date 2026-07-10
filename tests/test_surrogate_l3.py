"""Tests for the L3 dynamics-level surrogate (learned velocity law G_motion) and its
make_world wiring. The keystone invariant: the authentic world (drift_sigma=0) never
receives G_motion, so authentic dynamics stay byte-identical (guarded in test_world.py);
here we lock the surrogate-side plumbing."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl import experiment_b2 as b2  # noqa: E402
from itasorl.surrogate_l3 import (  # noqa: E402
    GMotion,
    collect_authentic_transitions,
    train_g_motion,
)
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams()


def test_collect_authentic_transitions_shapes():
    X, Y = collect_authentic_transitions(n_eps=6, steps=8, ray_steps=4, seed0=0)
    assert X.ndim == 2 and X.shape[1] == 4          # inputs = [vel_x, vel_y, a_x, a_y]
    assert Y.shape[1] == 2 and X.shape[0] == Y.shape[0] > 0


def test_train_g_motion_callable_and_finite():
    g = train_g_motion(hidden=4, n_eps=10, steps=10, epochs=20, ray_steps=4, seed=0)
    assert isinstance(g, GMotion)
    out = g(np.array([0.1, -0.2]), np.array([0.3, 0.0]), 1.5)   # (vel, a, drag) -> vel_next
    assert out.shape == (2,) and np.isfinite(out).all()


def test_make_world_l3_wiring(monkeypatch):
    def sentinel(vel, a, drag=None):
        return np.zeros(2)
    monkeypatch.setattr(b2, "DRIFT_MODE", "l3")
    monkeypatch.setattr(b2, "_L3_GMOTION", sentinel)
    surr = b2.make_world(P, 0.45, 4)     # surrogate world (drift_sigma > 0)
    auth = b2.make_world(P, 0.0, 4)      # authentic world (drift_sigma = 0)
    assert surr._g_motion is sentinel, "l3 surrogate world must receive the shared G_motion"
    assert auth._g_motion is None, "authentic world (drift_sigma=0) must stay authentic"


def test_setup_l3_surrogate_installs_net(monkeypatch):
    monkeypatch.setattr(b2, "_L3_GMOTION", None)
    b2.setup_l3_surrogate(hidden=4, n_eps=8, steps=8, epochs=10, ray_steps=4, seed=0)
    assert isinstance(b2._L3_GMOTION, GMotion)
