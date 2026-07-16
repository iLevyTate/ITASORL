"""Cross-recipe held-out surrogate families (spec 2026-07-15).

Both families satisfy the GMotion hook contract: callable `(vel, a, drag) ->
vel_next`, drag ignored, numpy-only per world step. They are EVALUATION-ONLY
transfer targets; the training surrogate stays the frozen GMotion MLP.

World-P scope (spec): drag is constant in the frozen organism world, so the
authentic velocity law is exactly linear in (vel, a). A family therefore only
has a fingerprint if it CANNOT represent that linear map (G_rff: cosine basis)
or is deliberately mis-set (G_cd: wrong drag constant).
"""

from __future__ import annotations

import numpy as np


class GConstantDrag:
    """Analytic constant-drag law with a deliberately mis-set constant.
    Degenerate L2-regime by construction; pre-registered as the SECONDARY
    cross-rung channel, never part of the primary decision."""

    def __init__(self, c: float, dt: float) -> None:
        self._c, self._dt = float(c), float(dt)

    def __call__(self, vel, a, drag=None) -> np.ndarray:
        return ((1.0 - self._c * self._dt) * np.asarray(vel, float)
                + np.asarray(a, float) * self._dt)


def make_g_cd(*, eps: float, params) -> GConstantDrag:
    """c = drag0 * (1 + eps) where drag0 is world-P's uniform drag. Refuses
    non-uniform-drag worlds: there the law would need wetness, which the hook
    deliberately cannot see, and the eps=0 identity check would be ill-defined.
    Rejects eps large enough to make the decay coefficient non-positive (c*dt >= 1)."""
    if params.k_land != params.k_water:
        raise ValueError("make_g_cd requires a uniform-drag world (k_land == k_water)")
    c = params.k_land * (1.0 + eps)
    if c * params.dt >= 1.0:
        raise ValueError(f"unstable constant-drag law: c*dt = {c * params.dt:.3f} >= 1")
    return GConstantDrag(c=c, dt=params.dt)


class GRff:
    """Random-Fourier-features ridge velocity law: z(x) = sqrt(2/D) cos(Wx + b)
    on normalized inputs, closed-form ridge readout. Smooth global sinusoidal
    basis + convex fit = a different recipe from the ReLU-MLP GMotion; the
    PRIMARY cross-recipe transfer target."""

    def __init__(self, W, b, Wout, norm, D) -> None:
        self._W = W.astype(np.float32)          # (D, 4)
        self._b = b.astype(np.float32)          # (D,)
        self._Wout = Wout.astype(np.float32)    # (D, 2)
        self._xm, self._xs, self._ym, self._ys = norm
        self._scale = np.float32(np.sqrt(2.0 / D))

    def __call__(self, vel, a, drag=None) -> np.ndarray:
        x = (np.array([vel[0], vel[1], a[0], a[1]], np.float32) - self._xm) / self._xs
        z = self._scale * np.cos(self._W @ x + self._b)
        return (z @ self._Wout * self._ys + self._ym).astype(float)


def fit_g_rff(*, D: int = 32, lam: float = 1e-3, ell: float = 1.0,
              feature_seed: int = 0, n_eps: int = 250, steps: int = 40,
              params=None, ray_steps: int = 5, seed: int = 0) -> GRff:
    """Fit on the same authentic-transition data budget as train_g_motion.
    Frozen defaults per spec: lam=1e-3, ell=1.0 on normalized inputs,
    feature_seed=0. Difficulty knob: D (feature count)."""
    from .surrogate_l3 import collect_authentic_transitions
    X, Y = collect_authentic_transitions(n_eps=n_eps, steps=steps, params=params,
                                         ray_steps=ray_steps, seed0=seed)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    rng = np.random.default_rng(feature_seed)
    W = rng.normal(0.0, 1.0 / ell, size=(D, 4)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, size=D).astype(np.float32)
    Z = (np.sqrt(2.0 / D) * np.cos(((X - xm) / xs) @ W.T + b)).astype(np.float64)
    A = Z.T @ Z + lam * np.eye(D, dtype=np.float64)
    Wout = np.linalg.solve(A, Z.T @ ((Y - ym) / ys).astype(np.float64))
    return GRff(W, b, Wout, (xm, xs, ym, ys), D)


RFF_SWEEP = (8, 16, 32, 64, 128)          # spec: ascending, freeze FIRST in-band
CD_SWEEP = (0.05, 0.1, 0.2, 0.4, 0.8)     # spec: coarse grid, then bisect


def gate0_candidates(family: str, *, params, sweep=None, **fit_kwargs):
    """Yield ((knob_name, knob_value), g) pairs for the gate-0 sweep.
    `sweep` overrides the frozen default grid (sorted ascending so the
    freeze-FIRST-in-band selection rule stays well-defined).
    fit_kwargs pass through to fit_g_rff (test-size overrides)."""
    if family == "rff":
        for D in sorted(sweep) if sweep is not None else RFF_SWEEP:
            yield ("D", int(D)), fit_g_rff(D=int(D), params=params, **fit_kwargs)
    elif family == "cd":
        for eps in sorted(sweep) if sweep is not None else CD_SWEEP:
            yield ("eps", float(eps)), make_g_cd(eps=float(eps), params=params)
    else:
        raise ValueError(f"unknown family: {family!r}")
