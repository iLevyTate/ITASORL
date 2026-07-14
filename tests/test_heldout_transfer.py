"""Tests for the held-out fingerprint transfer channel (spec:
docs/superpowers/specs/2026-07-14-l3-heldout-common-garden-probe-design.md).

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
    assert abs(transfer_probe(Xtr, ytr, Xte, yte) - 0.5) < 0.15


def test_transfer_probe_degenerate_labels_nan():
    rng = np.random.default_rng(2)
    Xtr, ytr = _pools(rng, 20, 8, 0, 2.0)
    Xte = rng.normal(0, 1, (10, 8))
    yte = np.zeros(10, int)                                 # one class only
    assert np.isnan(transfer_probe(Xtr, ytr, Xte, yte))
