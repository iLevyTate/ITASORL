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


# ---------------------------------------------------------------------------
# AUROC uncertainty. A point AUROC says nothing about precision; reviewers of a
# null result will (rightly) ask for an interval. These are dependency-light
# (numpy only) so they run wherever the pipeline runs.
# ---------------------------------------------------------------------------
def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks (1-based), matching scipy.stats.rankdata(method='average')."""
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(a.size, dtype=np.intp)
    inv[sorter] = np.arange(a.size)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], a.size]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auroc(y_true, y_score) -> float:
    """AUROC via the Mann-Whitney U / rank statistic (handles ties). NaN if one class."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(y_score)
    sum_pos = float(ranks[y_true == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auroc_ci(y_true, y_score, level: float = 0.95, n_boot: int = 2000,
             seed: int = 0) -> tuple[float, float]:
    """Stratified bootstrap CI for a single AUROC. Resamples positives and negatives
    with replacement (keeps both classes present) and recomputes the rank-AUROC; no
    model refit, so it is cheap enough to attach to every reported number."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if pos.size == 0 or neg.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([
            pos[rng.integers(0, pos.size, pos.size)],
            neg[rng.integers(0, neg.size, neg.size)],
        ])
        aucs[b] = auroc(y_true[idx], y_score[idx])
    a = (1.0 - level) / 2.0
    return (float(np.nanpercentile(aucs, 100 * a)), float(np.nanpercentile(aucs, 100 * (1 - a))))


def mean_ci(values, level: float = 0.90, n_boot: int = 10000,
            seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI of the across-seed mean. Seeds are the replication unit for a null
    claim (cf. Colas et al., 'How many random seeds?'), so this is the decision-relevant
    interval. Returns (mean, lo, hi)."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    a = (1.0 - level) / 2.0
    return (float(x.mean()), float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a))))


@dataclass
class RopeResult:
    mean: float
    rope: tuple[float, float]
    hdi: tuple[float, float]      # bootstrap percentile interval of the mean
    p_in_rope: float             # P(mean in ROPE) under the bootstrap posterior
    accept: bool                 # 95% interval entirely inside ROPE -> accept equivalence
    n: int

    def __str__(self) -> str:
        verdict = "ACCEPT equivalence" if self.accept else "inconclusive"
        return (f"mean={self.mean:.3f}  ROPE=[{self.rope[0]:.3f},{self.rope[1]:.3f}]  "
                f"95%HDI=[{self.hdi[0]:.3f},{self.hdi[1]:.3f}]  "
                f"P(in ROPE)={self.p_in_rope:.3f}  -> {verdict} (n={self.n})")


def rope_test(values, rope: tuple[float, float] = (0.45, 0.55), level: float = 0.95,
              n_boot: int = 20000, seed: int = 0) -> RopeResult:
    """Bayesian-style equivalence leg (Kruschke HDI+ROPE). A bootstrap posterior over the
    across-seed mean; accept equivalence when the 95% interval lies entirely inside the
    ROPE. Reported alongside TOST - both agreeing is a cheap, large credibility gain for
    a null."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    lo_r, hi_r = rope
    if n < 2:
        m = float(x.mean()) if n else float("nan")
        inside = bool(lo_r <= m <= hi_r) if n else False
        return RopeResult(m, (lo_r, hi_r), (m, m), float(inside), inside, n)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    a = (1.0 - level) / 2.0
    hdi = (float(np.percentile(means, 100 * a)), float(np.percentile(means, 100 * (1 - a))))
    p_in = float(np.mean((means >= lo_r) & (means <= hi_r)))
    accept = bool(hdi[0] >= lo_r and hdi[1] <= hi_r)
    return RopeResult(float(x.mean()), (lo_r, hi_r), hdi, p_in, accept, n)
