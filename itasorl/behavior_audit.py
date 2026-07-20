"""ITASORL - behavior-mediation audit for the pooled world-identity probe.

The L3 positive carries a mediation caveat: the agent behaves differently in
the two worlds, so the state probe could be reading BEHAVIOR rather than a
world-identity representation. This module makes that audit reproducible:

  behavior_only_auroc        can behavior alone decode the world? (the ceiling)
  residual_probe_auroc       per-episode-mean control: probe the state features
                             after removing what episode-mean behavior predicts
  trace_residual_probe_auroc per-timestep control (strictly stronger): remove
                             what the instantaneous behavior trace predicts of
                             every h_t, then probe episode features of residuals
  audit_cell                 run everything on one dumped (drift, seed, agent)
                             cell; per-timestep metrics only when the dump
                             carries traces (bta/bts)

All controls are fit IN-FOLD (train folds only, GroupKFold at episode level,
same estimator family as the headline probe) because in-sample residualization
over-removes: it deflated the audited signal to 0.56-0.63 in the ad hoc run.
Spec: docs/specs/2026-07-12-l3-behavior-audit-design.md.
"""

from __future__ import annotations

import numpy as np

from itasorl.experiment_b import (episode_features, episode_features_full,
                                  probe_auroc)

#: dump trace channel order (matches collect_pool's accumulators). Dumps written
#: before 2026-07-18 carry only the first four channels; position/heading were
#: added after the methodology audit flagged that diverging position paths are
#: an uncontrolled mediator (FINDINGS 10.4 covariate-gap note). All controls
#: below are channel-count agnostic and accept both dump generations.
BEHAVIOR_CHANNELS = ("speed", "energy", "food", "drag", "pos_x", "pos_y", "heading")


def quad_expand(B: np.ndarray) -> np.ndarray:
    """Degree-2 feature map: [B, B^2, pairwise products]. No bias column (the
    downstream estimators fit intercepts)."""
    n = B.shape[1]
    pairs = [(B[:, i] * B[:, j])[:, None] for i in range(n) for j in range(i + 1, n)]
    return np.concatenate([B, B ** 2] + pairs, axis=1)


def behavior_only_auroc(B: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                        nonlinear: bool = False, seed: int = 0, n_splits: int = 5) -> float:
    """Grouped-CV AUROC of behavior features alone -> world label. linear =
    the headline probe family (scaled logistic); nonlinear = random forest,
    which sees variance/interaction codes the linear decoder misses."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from itasorl.experiment_a import grouped_auroc
    if groups is None:
        groups = np.arange(len(y))
    if not nonlinear:
        return grouped_auroc(B, y, groups, n_splits=n_splits)
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(B, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf.fit(B[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(B[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def residual_probe_auroc(X: np.ndarray, B: np.ndarray, y: np.ndarray,
                         groups: np.ndarray | None = None, quad: bool = False,
                         n_splits: int = 5) -> float:
    """Behavior-controlled probe. Inside each CV fold: fit behavior -> feature
    regression on TRAIN only, residualize train AND test with that model, then
    run the standard probe on residuals. Out-of-fold mean AUROC."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if groups is None:
        groups = np.arange(len(y))
    Phi = quad_expand(B) if quad else B
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        reg = make_pipeline(StandardScaler(), LinearRegression())
        reg.fit(Phi[tr], X[tr])
        r_tr = X[tr] - reg.predict(Phi[tr])
        r_te = X[te] - reg.predict(Phi[te])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(r_tr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(r_te)[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _trace_phi(Bt: np.ndarray, quad: bool) -> np.ndarray:
    """Per-timestep regressors: [b_t, b_{t-1} (edge-padded), cummean_{<=t}(b)].
    Contemporaneous b_t is deliberately conservative (h_t precedes it by half a
    step): removing slightly more than causally available makes any SURVIVING
    signal harder to dismiss. quad applies the degree-2 map to b_t only."""
    n, T, C = Bt.shape
    b_prev = np.concatenate([Bt[:, :1], Bt[:, :-1]], axis=1)
    cummean = np.cumsum(Bt, axis=1) / np.arange(1, T + 1, dtype=float)[None, :, None]
    b_now = Bt.reshape(n * T, C)
    if quad:
        b_now = quad_expand(b_now)
    return np.concatenate([b_now, b_prev.reshape(n * T, C),
                           cummean.reshape(n * T, C)], axis=1)


def trace_residual_probe_auroc(H: np.ndarray, Bt: np.ndarray, y: np.ndarray,
                               groups: np.ndarray | None = None, quad: bool = False,
                               n_splits: int = 5) -> float:
    """Per-timestep behavior control. Per fold: fit phi(b)_t -> h_t on the
    TRAIN episodes' pooled timesteps, residualize every h_t with that model,
    rebuild episode features from the residual states, probe as usual."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if groups is None:
        groups = np.arange(len(y))
    n, T, hid = H.shape
    Phi = _trace_phi(np.asarray(Bt, dtype=float), quad)          # (n*T, P)
    Hflat = np.asarray(H, dtype=float).reshape(n * T, hid)
    row_ep = np.repeat(np.arange(n), T)                          # episode id per row
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(n), y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        tr_rows = np.isin(row_ep, tr)
        reg = make_pipeline(StandardScaler(), LinearRegression())
        reg.fit(Phi[tr_rows], Hflat[tr_rows])
        R = (Hflat - reg.predict(Phi)).reshape(n, T, hid)
        F = episode_features(R)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(F[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(F[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def audit_cell(npz: dict, seed: int = 0) -> dict:
    """Behavior-mediation audit of one dumped cell. Returns {} when either
    pool is too small to probe (mirrors the headline's 5-survivor guard)."""
    Ha, Hs = np.asarray(npz["Ha"]), np.asarray(npz["Hs"])
    if len(Ha) < 5 or len(Hs) < 5:
        return {}
    H = np.concatenate([Ha, Hs])
    y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
    X = episode_features(H)
    B = np.stack([np.concatenate([npz["spa"], npz["sps"]]),
                  np.concatenate([npz["ena"], npz["ens"]]),
                  np.concatenate([npz["fda"], npz["fds"]]),
                  np.concatenate([npz["dra"], npz["drs"]])], axis=1)
    if "bta" in npz and np.asarray(npz["bta"]).shape[-1] > 4:
        # post-2026-07-18 dumps: fold the extra trace channels (position/heading)
        # into the episode-mean basis so the control covers the position mediator
        Bt_all = np.concatenate([np.asarray(npz["bta"]), np.asarray(npz["bts"])])
        B = np.concatenate([B, Bt_all[:, :, 4:].mean(axis=1)], axis=1)
    out = {
        "target": probe_auroc(X, y),
        "behavior_only": behavior_only_auroc(B, y),
        "behavior_only_nonlinear": behavior_only_auroc(B, y, nonlinear=True, seed=seed),
        "resid_epmean": residual_probe_auroc(X, B, y),
        "resid_epmean_quad": residual_probe_auroc(X, B, y, quad=True),
    }
    if "bta" in npz and "bts" in npz:
        Bt = np.concatenate([np.asarray(npz["bta"]), np.asarray(npz["bts"])])
        out["behavior_trace_only"] = probe_auroc(episode_features_full(Bt), y)
        out["resid_trace"] = trace_residual_probe_auroc(H, Bt, y)
        out["resid_trace_quad"] = trace_residual_probe_auroc(H, Bt, y, quad=True)
    return out


def aggregate_cells(rows: list[dict], bar: float = 0.65) -> dict:
    """Across-seed aggregation, mirroring the headline reporting: for every
    metric present in any cell, the across-seed mean with a 90% bootstrap CI
    (seeds are the replication unit) plus the count of seeds clearing `bar`.
    NaN cells are ignored."""
    from itasorl.stats import mean_ci
    keys = sorted({k for r in rows for k in r})
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        if not vals:
            continue
        mean, lo, hi = mean_ci(vals)
        agg[k] = {"mean": mean, "lo": lo, "hi": hi, "n_seeds": len(vals),
                  "n_above_bar": int(sum(v >= bar for v in vals))}
    return agg
