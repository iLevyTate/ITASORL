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
    deliberately cannot see, and the eps=0 identity check would be ill-defined."""
    if params.k_land != params.k_water:
        raise ValueError("make_g_cd requires a uniform-drag world (k_land == k_water)")
    return GConstantDrag(c=params.k_land * (1.0 + eps), dt=params.dt)
