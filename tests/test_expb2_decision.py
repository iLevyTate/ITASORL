"""decide_h_b2: pre-registered verdict zones and the NaN survivorship guard.

A non-finite pooled target means a seed's pool had too few survivors; the
verdict must read UNINFORMATIVE (broken precondition), never a silent
"NOT met" (which would conflate an apparatus failure with a negative result).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_expB2  # noqa: E402


def test_met_zone():
    ok, zone, n_bad, fin = run_expB2.decide_h_b2([0.70] * 3, [0.50] * 3, [0.50] * 3)
    assert ok and zone.startswith("MET") and n_bad == 0
    assert fin.size == 3


def test_intermediate_zone():
    ok, zone, n_bad, _ = run_expB2.decide_h_b2([0.60] * 3, [0.50] * 3, [0.50] * 3)
    assert not ok and "intermediate" in zone and n_bad == 0


def test_strengthened_negative_zone():
    ok, zone, n_bad, _ = run_expB2.decide_h_b2([0.52] * 3, [0.50] * 3, [0.50] * 3)
    assert not ok and "strengthened negative" in zone and n_bad == 0


def test_nan_cell_forces_uninformative_even_when_finite_seeds_clear_bar():
    surv = [0.85, float("nan"), 0.84]
    ok, zone, n_bad, fin = run_expB2.decide_h_b2(surv, [0.50] * 3, [0.50] * 3)
    assert not ok
    assert "UNINFORMATIVE" in zone
    assert n_bad == 1
    assert fin.size == 2 and np.isfinite(fin).all()


def test_nan_in_baseline_also_uninformative():
    ok, zone, n_bad, _ = run_expB2.decide_h_b2(
        [0.85] * 3, [0.50, float("nan"), 0.50], [0.50] * 3)
    assert not ok and "UNINFORMATIVE" in zone and n_bad == 1


def test_all_nan_survival_uninformative_with_empty_finite():
    ok, zone, n_bad, fin = run_expB2.decide_h_b2(
        [float("nan")] * 3, [0.50] * 3, [0.50] * 3)
    assert not ok and "UNINFORMATIVE" in zone and n_bad == 3
    assert fin.size == 0
