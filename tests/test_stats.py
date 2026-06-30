"""Unit tests for the inference helpers in stats.py (AUROC CIs + Bayesian ROPE leg).

These are the load-bearing numbers behind the null claim, so they get direct coverage
independent of the (torch-heavy) experiment pipeline. All pure numpy, fast, deterministic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from itasorl.stats import auroc, auroc_ci, equivalence_test, mean_ci, rope_test


def test_auroc_matches_known_cases():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0      # perfect separation
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0      # perfectly reversed
    assert abs(auroc([0, 1], [0.5, 0.5]) - 0.5) < 1e-9           # all ties -> 0.5
    assert math.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))        # one class -> nan


def test_auroc_matches_sklearn_on_random_data():
    roc_auc_score = pytest.importorskip("sklearn.metrics").roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    while len(np.unique(y)) < 2:
        y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + 0.5 * y                          # mild signal + ties-free
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-9


def test_auroc_ci_brackets_and_bounded():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(60), np.ones(60)].astype(int)
    s = np.r_[rng.normal(0.0, 1.0, 60), rng.normal(1.2, 1.0, 60)]   # clearly separable
    pt = auroc(y, s)
    lo, hi = auroc_ci(y, s, n_boot=800, seed=0)
    assert 0.0 <= lo < hi <= 1.0
    assert lo <= pt <= hi                                        # percentile CI brackets the point
    assert lo > 0.5                                             # real signal: CI clears chance


def test_mean_ci_centers_on_mean():
    m, lo, hi = mean_ci([0.49, 0.51, 0.50, 0.48, 0.52], level=0.90, seed=0)
    assert abs(m - 0.5) < 1e-9
    assert lo <= m <= hi


def test_mean_ci_handles_single_value():
    m, lo, hi = mean_ci([0.5])
    assert m == lo == hi == 0.5


def test_rope_accepts_tight_chance_rejects_clear_signal():
    near = rope_test([0.50, 0.49, 0.51, 0.50, 0.50, 0.51, 0.49], rope=(0.45, 0.55), seed=0)
    far = rope_test([0.78, 0.80, 0.82, 0.79, 0.81], rope=(0.45, 0.55), seed=0)
    assert near.accept is True                                  # 95% HDI inside ROPE
    assert far.accept is False                                  # clearly above the ROPE
    assert near.p_in_rope > far.p_in_rope
    assert 0.0 <= near.p_in_rope <= 1.0


def test_rope_and_tost_agree_on_a_chance_sample():
    vals = [0.50, 0.49, 0.51, 0.50, 0.48, 0.52, 0.50, 0.49]
    assert equivalence_test(vals).equivalent is True
    assert rope_test(vals).accept is True
