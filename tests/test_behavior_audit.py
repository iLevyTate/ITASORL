"""Tests for the behavior-mediation audit (itasorl/behavior_audit.py).

The load-bearing guarantees are the CONTROL PROPERTIES, all on synthetic data
where the ground truth is known:
  - a world signal that reaches the state ONLY through behavior is removed by
    the in-fold behavior control (no false positive);
  - a genuine behavior-independent world direction SURVIVES the control (no
    over-removal, the failure mode of in-sample residualization);
  - a timing-pattern signal invisible to per-episode means is removed by the
    per-timestep control but NOT by the per-episode control (the new control
    is strictly stronger);
  - the audit runs on old-format dumps (no traces) without crashing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from itasorl.behavior_audit import (  # noqa: E402
    aggregate_cells,
    audit_cell,
    behavior_only_auroc,
    residual_probe_auroc,
    trace_residual_probe_auroc,
)
from itasorl.experiment_b import episode_features, probe_auroc  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic generators
# ---------------------------------------------------------------------------

def _epmean_data(k=60, d_beh=4, d_x=24, mediated=True, orthogonal=False, seed=0):
    """Episode-level data. y -> B (behavior) always; X = M @ B + noise, plus an
    optional world direction ORTHOGONAL to the behavior pathway."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(k, int), np.ones(k, int)])
    B = rng.normal(size=(2 * k, d_beh))
    if mediated:
        B = B + 1.5 * y[:, None]                       # behavior separates the worlds
    M = rng.normal(size=(d_beh, d_x)) / np.sqrt(d_beh)
    X = B @ M + 0.3 * rng.normal(size=(2 * k, d_x))
    if orthogonal:
        d = rng.normal(size=d_x)
        d -= M.T @ np.linalg.lstsq(M.T, d, rcond=None)[0]  # project out span(M rows)
        X = X + 2.0 * np.outer(y, d / np.linalg.norm(d))
    return X, B, y


def _trace_data(k=50, T=12, C=4, hid=12, timing=True, orthogonal=False, seed=0):
    """Per-timestep data. Channel 0 ramps UP over the episode in world 1 and DOWN
    in world 0 with IDENTICAL per-episode means, so an episode-mean control sees
    nothing while the state (driven by instantaneous behavior) still separates.
    h_t = W b_t + noise (+ optional orthogonal world direction)."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(k, int), np.ones(k, int)])
    ramp = np.arange(T) / (T - 1.0)
    Bt = 0.3 * rng.normal(size=(2 * k, T, C))
    if timing:
        Bt[:, :, 0] += np.where(y[:, None] == 1, ramp[None, :], ramp[None, ::-1])
    W = rng.normal(size=(C, hid)) / np.sqrt(C)
    H = Bt @ W + 0.1 * rng.normal(size=(2 * k, T, hid))
    if orthogonal:
        d = rng.normal(size=hid)
        d -= W.T @ np.linalg.lstsq(W.T, d, rcond=None)[0]
        H = H + 1.0 * (y[:, None, None] * (d / np.linalg.norm(d))[None, None, :])
    return H.astype(np.float32), Bt, y


# ---------------------------------------------------------------------------
# per-episode control
# ---------------------------------------------------------------------------

def test_behavior_mediated_signal_is_removed():
    """X depends on y ONLY through behavior: the uncontrolled probe decodes the
    world, the in-fold behavior control must drive it back to chance."""
    X, B, y = _epmean_data(mediated=True)
    assert probe_auroc(X, y) > 0.85
    assert residual_probe_auroc(X, B, y) < 0.62


def test_orthogonal_world_signal_survives_control():
    """With behavior UNINFORMATIVE about the world, a world direction orthogonal
    to the behavior pathway must survive the control at full strength - blind
    over-removal is exactly the in-sample failure mode we reject."""
    X, B, y = _epmean_data(mediated=False, orthogonal=True)
    assert residual_probe_auroc(X, B, y) > 0.85


def test_orthogonal_signal_attenuated_but_alive_under_joint_mediation():
    """When behavior ALSO decodes the world, residualization necessarily absorbs
    part of any y-correlated signal through the B->y correlation. The genuine
    orthogonal component must remain well above chance - this bounded
    attenuation is the price of the conservative control and mirrors the
    published 0.752 -> ~0.66 drop."""
    X, B, y = _epmean_data(mediated=True, orthogonal=True)
    assert residual_probe_auroc(X, B, y) > 0.70


def test_pure_noise_stays_at_chance():
    rng = np.random.default_rng(3)
    y = np.concatenate([np.zeros(60, int), np.ones(60, int)])
    X = rng.normal(size=(120, 24))
    B = rng.normal(size=(120, 4))
    assert 0.35 < residual_probe_auroc(X, B, y) < 0.65


def test_quadratic_control_removes_squared_mediation():
    """X reads |B| (a nonlinear function of behavior): the LINEAR control leaves
    signal behind, the quadratic control removes it."""
    rng = np.random.default_rng(4)
    k = 80
    y = np.concatenate([np.zeros(k, int), np.ones(k, int)])
    B = rng.normal(size=(2 * k, 4)) * (1.0 + 1.5 * y[:, None])  # variance-coded world
    M = rng.normal(size=(4, 24)) / 2.0
    X = (B ** 2) @ M + 0.3 * rng.normal(size=(2 * k, 24))
    assert residual_probe_auroc(X, B, y, quad=False) > 0.7
    assert residual_probe_auroc(X, B, y, quad=True) < 0.62


# ---------------------------------------------------------------------------
# behavior-only decode
# ---------------------------------------------------------------------------

def test_behavior_only_linear_and_chance():
    X, B, y = _epmean_data(mediated=True)
    assert behavior_only_auroc(B, y) > 0.85
    rng = np.random.default_rng(5)
    assert 0.35 < behavior_only_auroc(rng.normal(size=(120, 4)), y) < 0.65


def test_behavior_only_nonlinear_sees_variance_code():
    """A variance-coded world label is invisible to the linear decoder but the
    nonlinear (random forest) decoder must find it - mirrors the published
    0.689 linear vs 0.704 nonlinear gap."""
    rng = np.random.default_rng(6)
    k = 80
    y = np.concatenate([np.zeros(k, int), np.ones(k, int)])
    B = rng.normal(size=(2 * k, 4)) * (1.0 + 2.0 * y[:, None])
    assert behavior_only_auroc(B, y) < 0.65
    assert behavior_only_auroc(B, y, nonlinear=True) > 0.75


# ---------------------------------------------------------------------------
# per-timestep control
# ---------------------------------------------------------------------------

def test_timing_signal_defeats_epmean_control_but_not_trace_control():
    """The decisive test: equal episode MEANS, world-dependent temporal ORDER.
    The per-episode control cannot see it; the per-timestep control must
    remove it. This is what makes the new control strictly stronger."""
    H, Bt, y = _trace_data(timing=True)
    X = episode_features(H)
    B = Bt.mean(axis=1)
    assert probe_auroc(X, y) > 0.85                       # signal is real
    assert residual_probe_auroc(X, B, y) > 0.75           # epmean control blind to it
    assert trace_residual_probe_auroc(H, Bt, y) < 0.62    # trace control removes it


def test_orthogonal_world_signal_survives_trace_control():
    """Same no-blind-over-removal property for the per-timestep control:
    behavior traces uninformative about the world, genuine state direction."""
    H, Bt, y = _trace_data(timing=False, orthogonal=True)
    assert trace_residual_probe_auroc(H, Bt, y) > 0.85


def test_orthogonal_signal_attenuated_but_alive_under_trace_mediation():
    H, Bt, y = _trace_data(timing=True, orthogonal=True)
    assert trace_residual_probe_auroc(H, Bt, y) > 0.70


# ---------------------------------------------------------------------------
# audit_cell integration (old and new dump formats)
# ---------------------------------------------------------------------------

def _fake_npz(traces: bool, seed=7):
    rng = np.random.default_rng(seed)
    k, T, hid = 30, 6, 8
    d = {"Ha": rng.normal(size=(k, T, hid)).astype(np.float32),
         "Hs": rng.normal(size=(k, T, hid)).astype(np.float32),
         "spa": rng.random(k), "sps": rng.random(k),
         "ena": rng.random(k), "ens": rng.random(k),
         "fda": rng.random(k), "fds": rng.random(k),
         "dra": np.zeros(k), "drs": rng.random(k),
         "ra": rng.random(k), "rs": rng.random(k),
         "drift_sigma": np.float64(0.45), "steps": np.int64(T)}
    if traces:
        d["bta"] = rng.random((k, T, 4))
        d["bts"] = rng.random((k, T, 4))
    return d


def test_audit_cell_old_format_has_epmean_metrics_only():
    out = audit_cell(_fake_npz(traces=False), seed=0)
    for key in ("target", "behavior_only", "behavior_only_nonlinear",
                "resid_epmean", "resid_epmean_quad"):
        assert key in out and np.isfinite(out[key])
    assert "resid_trace" not in out


def test_audit_cell_new_format_adds_trace_metrics():
    out = audit_cell(_fake_npz(traces=True), seed=0)
    for key in ("resid_trace", "resid_trace_quad", "behavior_trace_only"):
        assert key in out and np.isfinite(out[key])


def test_aggregate_cells_reports_mean_ci_and_bar_count():
    """Across-seed aggregation mirrors the headline reporting: bootstrap mean CI
    plus how many seeds clear the 0.65 bar; NaN cells are ignored, metrics
    missing from every cell are absent from the output."""
    rows = [{"resid_epmean": 0.70}, {"resid_epmean": 0.66},
            {"resid_epmean": 0.60}, {"resid_epmean": float("nan")}]
    agg = aggregate_cells(rows, bar=0.65)
    m = agg["resid_epmean"]
    assert abs(m["mean"] - (0.70 + 0.66 + 0.60) / 3) < 1e-9
    assert m["lo"] <= m["mean"] <= m["hi"]
    assert m["n_seeds"] == 3 and m["n_above_bar"] == 2
    assert "resid_trace" not in agg
