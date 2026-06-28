"""
ITASORL - small statistics helpers for methodologically-tight readouts.

The headline tool here is an EQUIVALENCE test (two one-sided tests, TOST). The L0
control must be shown to sit AT chance, and "we failed to reject a difference from
0.5" is not the same claim as "it is equivalent to 0.5". TOST makes the at-chance
claim positively: it rejects the null of a meaningful difference in favour of
equivalence within a pre-registered margin. (ITASORL.md sec.13 item #2.)

Deliberately dependency-light (numpy + scipy.stats.t if available, else a normal
approximation) so it runs anywhere the rest of the pipeline runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import t as _student_t
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy optional
    _HAVE_SCIPY = False


def _t_sf(tstat: float, df: int) -> float:
    """Upper-tail P(T > tstat). Student-t if scipy is present, else normal approx."""
    if df <= 0:
        return float("nan")
    if _HAVE_SCIPY:
        return float(_student_t.sf(tstat, df))
    # normal approximation to the t survival function
    return float(0.5 * math.erfc(tstat / math.sqrt(2.0)))


@dataclass
class EquivalenceResult:
    mean: float
    margin: float
    lower: float          # (h0 - margin)
    upper: float          # (h0 + margin)
    p_lower: float        # H0: mean <= lower   (one-sided)
    p_upper: float        # H0: mean >= upper   (one-sided)
    p_value: float        # max(p_lower, p_upper) - the TOST p
    equivalent: bool      # p_value < alpha AND the mean lies inside the band
    n: int

    def __str__(self) -> str:
        verdict = "EQUIVALENT to chance" if self.equivalent else "NOT shown equivalent"
        return (f"mean={self.mean:.3f}  band=[{self.lower:.3f},{self.upper:.3f}]  "
                f"TOST p={self.p_value:.4f}  -> {verdict} (n={self.n})")


def equivalence_test(values, h0: float = 0.5, margin: float = 0.05,
                     alpha: float = 0.05) -> EquivalenceResult:
    """TOST equivalence test that `values` are within +/- margin of h0.

    values: a sample of summary statistics (e.g. one AUROC per seed). We test the
    composite null "the true mean is OUTSIDE [h0-margin, h0+margin]" with two
    one-sided t-tests; rejecting both (p<alpha) concludes practical equivalence.

    Returns an EquivalenceResult; `.equivalent` is the gate the L0 control uses.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    lower, upper = h0 - margin, h0 + margin
    mean = float(x.mean())
    if n < 2:
        # Can't estimate variance from <2 points; fall back to a pointwise band check.
        inside = lower <= mean <= upper
        return EquivalenceResult(mean, margin, lower, upper, float("nan"),
                                 float("nan"), float("nan"), bool(inside), n)
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 1e-12
    df = n - 1
    # H0a: mean <= lower  -> reject if mean is significantly ABOVE lower
    t_lower = (mean - lower) / se
    p_lower = _t_sf(t_lower, df)
    # H0b: mean >= upper  -> reject if mean is significantly BELOW upper
    t_upper = (upper - mean) / se
    p_upper = _t_sf(t_upper, df)
    p = max(p_lower, p_upper)
    equivalent = bool(p < alpha and lower <= mean <= upper)
    return EquivalenceResult(mean, margin, lower, upper, p_lower, p_upper, p, equivalent, n)
