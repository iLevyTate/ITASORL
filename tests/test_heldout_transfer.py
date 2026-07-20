"""Tests for the held-out fingerprint transfer channel (spec:
docs/specs/2026-07-14-l3-heldout-common-garden-probe-design.md).

Synthetic ground truth: a fingerprint-GENERAL signal (same discriminative
direction in train and test pools) must transfer; a fingerprint-SPECIFIC
signal (orthogonal directions) must not."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from itasorl.experiment_b2 import transfer_probe  # noqa: E402


def _pools(rng, n, dim, sig_dim, shift):
    """Two pools of feature vectors separated by `shift` along axis sig_dim."""
    Xa = rng.normal(0, 1, (n, dim))
    Xs = rng.normal(0, 1, (n, dim))
    Xs[:, sig_dim] += shift
    X = np.concatenate([Xa, Xs])
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
    return X, y


def test_transfer_probe_general_signal_transfers():
    rng = np.random.default_rng(0)
    Xtr, ytr = _pools(rng, 80, 16, sig_dim=3, shift=3.0)
    Xte, yte = _pools(rng, 80, 16, sig_dim=3, shift=3.0)   # SAME direction
    assert transfer_probe(Xtr, ytr, Xte, yte) > 0.9


def test_transfer_probe_specific_signal_does_not_transfer():
    rng = np.random.default_rng(1)
    Xtr, ytr = _pools(rng, 80, 16, sig_dim=3, shift=3.0)
    Xte, yte = _pools(rng, 80, 16, sig_dim=11, shift=3.0)  # ORTHOGONAL direction
    assert abs(transfer_probe(Xtr, ytr, Xte, yte) - 0.5) < 0.20


def test_transfer_probe_degenerate_labels_nan():
    rng = np.random.default_rng(2)
    Xtr, ytr = _pools(rng, 20, 8, 0, 2.0)
    Xte = rng.normal(0, 1, (10, 8))
    yte = np.zeros(10, int)                                 # one class only
    assert np.isnan(transfer_probe(Xtr, ytr, Xte, yte))


def test_transfer_probe_return_scores_contract():
    rng = np.random.default_rng(3)
    Xtr, ytr = _pools(rng, 40, 8, sig_dim=2, shift=3.0)
    Xte, yte = _pools(rng, 40, 8, sig_dim=2, shift=3.0)
    auc, yv, pv = transfer_probe(Xtr, ytr, Xte, yte, return_scores=True)
    assert 0.0 <= auc <= 1.0 and pv.shape == (80,) and np.array_equal(yv, yte)


from itasorl.experiment_b2 import (  # noqa: E402
    pooled_readout, transfer_readout, untrained_agent,
)
import itasorl.experiment_b2 as b2  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RS = 4
TINY_G = dict(n_eps=6, steps=10, epochs=3, device="cpu", seed=0)


@pytest.fixture()
def l3_tiny():
    """Install tiny trained + held-out surrogates; restore module state after."""
    saved = (b2.DRIFT_MODE, b2._L3_GMOTION, b2._L3_GMOTION_HELDOUT)
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=2, params=P, **TINY_G)
    b2.setup_l3_heldout_surrogate(hidden=3, params=P, **TINY_G)
    yield
    b2.DRIFT_MODE, b2._L3_GMOTION, b2._L3_GMOTION_HELDOUT = saved


def test_pooled_readout_return_pools_shape():
    agent, norm = untrained_agent(P, 0.0, RS, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    out, (Ha, Hs) = pooled_readout(agent, norm, P, 0.0, n_eps=6, steps=5, ray_steps=RS,
                                   device="cpu", return_pools=True)
    assert isinstance(out, dict) and Ha.ndim == 3 and Hs.ndim == 3
    # default call still returns the bare dict (schema no-op)
    out2 = pooled_readout(agent, norm, P, 0.0, n_eps=6, steps=5, ray_steps=RS, device="cpu")
    assert isinstance(out2, dict) and set(out2) == set(out)


def test_transfer_readout_runs_and_restores_surrogate(l3_tiny):
    agent, norm = untrained_agent(P, 0.45, RS, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    _, (Ha, Hs) = pooled_readout(agent, norm, P, 0.45, n_eps=8, steps=5, ray_steps=RS,
                                 device="cpu", return_pools=True)
    before = b2._L3_GMOTION
    tr = transfer_readout(agent, norm, P, 0.45, Ha, Hs, n_eps=8, steps=5,
                          ray_steps=RS, device="cpu")
    assert b2._L3_GMOTION is before, "trained surrogate must be restored after transfer eval"
    assert set(tr) >= {"transfer_target", "transfer_n_auth", "transfer_n_surr"}
    assert np.isnan(tr["transfer_target"]) or 0.0 <= tr["transfer_target"] <= 1.0


def test_transfer_readout_requires_heldout_installed(l3_tiny):
    b2._L3_GMOTION_HELDOUT = None
    agent, norm = untrained_agent(P, 0.45, RS, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    with pytest.raises(RuntimeError):
        transfer_readout(agent, norm, P, 0.45, np.zeros((5, 5, 8), np.float32),
                         np.zeros((5, 5, 8), np.float32), n_eps=5, steps=5,
                         ray_steps=RS, device="cpu")


def test_transfer_readout_restores_surrogate_on_exception(l3_tiny, monkeypatch):
    agent, norm = untrained_agent(P, 0.45, RS, hidden=8, embed=16, world_model=True,
                                  device="cpu", seed=0)
    before = b2._L3_GMOTION

    def boom(*a, **k):
        raise ValueError("collect_pool failed mid-swap")

    monkeypatch.setattr(b2, "collect_pool", boom)
    with pytest.raises(ValueError):
        transfer_readout(agent, norm, P, 0.45, np.zeros((5, 5, 8), np.float32),
                         np.zeros((5, 5, 8), np.float32), n_eps=5, steps=5,
                         ray_steps=RS, device="cpu")
    assert b2._L3_GMOTION is before, "finally block must restore the trained surrogate"
