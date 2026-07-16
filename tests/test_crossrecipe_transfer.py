"""Synthetic ground truth for the cross-recipe transfer semantics (spec test 1)
plus the no-op contract of the generalized transfer_readout."""

import inspect

import numpy as np

from itasorl.experiment_b import episode_features
from itasorl.experiment_b2 import transfer_probe, transfer_readout


def _pool(rng, n, steps, hid, shift):
    """Episodes (n, steps, hid) of unit noise with a mean shift along `shift`."""
    return rng.normal(size=(n, steps, hid)) + shift


def test_shared_component_transfers_orthogonal_does_not():
    rng = np.random.default_rng(0)
    hid, n, steps = 16, 80, 12
    u = np.zeros(hid); u[0] = 1.0          # shared world-signal direction
    v = np.zeros(hid); v[1] = 1.0          # orthogonal texture
    auth_tr = _pool(rng, n, steps, hid, 0.0)
    surr_tr = _pool(rng, n, steps, hid, 1.5 * u)      # trained fingerprint
    auth_te = _pool(rng, n, steps, hid, 0.0)
    surr_shared = _pool(rng, n, steps, hid, 1.5 * u)  # different family, shared component
    surr_orth = _pool(rng, n, steps, hid, 1.5 * v)    # different family, orthogonal

    Xtr = episode_features(np.concatenate([auth_tr, surr_tr]))
    ytr = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    def score(surr_te):
        Xte = episode_features(np.concatenate([auth_te, surr_te]))
        yte = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        return transfer_probe(Xtr, ytr, Xte, yte)

    assert score(surr_shared) > 0.9
    assert abs(score(surr_orth) - 0.5) < 0.15


def test_transfer_readout_defaults_are_noop():
    """The generalization must not change the default call contract: same
    signature defaults the heldout global and the original seed bases."""
    sig = inspect.signature(transfer_readout)
    assert sig.parameters["heldout"].default is None
    assert sig.parameters["seed_base_auth"].default == 860_000
    assert sig.parameters["seed_base_surr"].default == 870_000
