"""ITASORL - Experiment C gate 1: world-only-exploitability (world_spec / prereg sec. 7).

Gate 1 certifies the fitness coupling itself, BEFORE any selection is run: is
world identity worth knowing? We answer with a scripted momentum-to-target
controller rather than a learned policy, so the measurement isolates the single
primitive that separates the worlds - the velocity update at
``patch_of_earth.py:177`` - with no training noise and bit-for-bit determinism.

  * ``scripted_reach_payoff`` runs a constant forward thrust from rest toward a
    straight-ahead target and scores ``-final distance``. Because the two worlds
    share terrain/ecology seeds and differ only in the drag law, the payoff gap
    at fixed thrust is a pure read of the velocity primitive.
  * ``value_of_world_identity`` sweeps thrust in each world and reports the
    ORACLE-minus-BLIND gap: the extra payoff a world-conditional controller earns
    over the single best world-blind thrust. That gap IS the value of knowing the
    world (a latent-MDP regret), so it is exactly what selection could exploit.

The control layout exploits an exact mechanical fact: from ``vel=0`` the first
integrated step is ``a*dt`` regardless of drag, so a from-rest short-horizon
reach is world-invariant BY CONSTRUCTION for the ar1/regime surrogates. Under the
L3 learned map that invariance is not guaranteed, so gate 1 MEASURES the control
gap (bootstrap CI must include 0) rather than assuming it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .experiment_b2 import _seeds
from .patch_of_earth import PatchOfEarthV0
from .world import WorldParams


@dataclass(frozen=True)
class Layout:
    """A scripted-reach geometry: how far ahead the target sits and for how many
    steps the controller drives toward it. ``horizon`` is what turns drag (world
    identity) on or off - a long reach lets velocity build so the drag law bites
    (treatment), a short from-rest reach stays in the drag-invariant regime
    (control)."""
    reach_range: float
    horizon: int
    name: str = ""


def scripted_reach_payoff(
    world: PatchOfEarthV0,
    thrust: float,
    layout: Layout,
    *,
    seed: int,
    start: tuple[float, float] = (0.3, 0.5),
) -> float:
    """Payoff = ``-‖final_pos - target‖`` for a constant forward thrust from rest.

    Resets ``world`` on ``seed`` (so terrain/ecology are fixed), then overrides the
    agent to a known from-rest pose (``pos=start``, ``heading=0`` -> +x, ``vel=0``)
    while LEAVING the per-episode drag regime (``_drift_w``) intact. The target sits
    ``reach_range`` straight ahead; the controller applies ``[thrust,0,0,0,0]`` for
    ``horizon`` steps. Deterministic given the seed."""
    world.reset(_seeds(seed))
    world.pos = np.array(start, dtype=float)
    world.heading = 0.0
    world.vel = np.zeros(2)
    d = np.array([np.cos(world.heading), np.sin(world.heading)])
    target = world.pos + layout.reach_range * d
    action = np.array([float(thrust), 0.0, 0.0, 0.0, 0.0], dtype=float)
    for _ in range(layout.horizon):
        world.step(action)
    return -float(np.linalg.norm(world.pos - target))


def value_of_world_identity(
    layout: Layout,
    *,
    seed: int,
    drift_sigma: float,
    drift_mode: str = "regime",
    thrusts: np.ndarray | None = None,
    params: WorldParams | None = None,
    ray_steps: int = 5,
    g_motion: Callable | None = None,
) -> dict[str, Any]:
    """Oracle-minus-blind payoff gap for a layout: the value of knowing the world.

    Sweeps ``thrusts`` in an authentic world (``drift_sigma=0``) and a surrogate
    world (``drift_sigma>0``) that share ``seed``/``params`` and thus differ ONLY
    in the drag law. The ORACLE picks the best thrust per world; the BLIND
    controller must pick one thrust that is best on average across both. The gap
    ``oracle - blind`` is non-negative (Jensen) and is > 0 exactly when the
    world-optimal thrusts diverge. Under ``drift_mode="l3"`` pass the frozen
    ``g_motion`` to certify against the real learned map. Deterministic given the
    seed."""
    thrusts = np.linspace(0.0, 1.0, 21) if thrusts is None else np.asarray(thrusts, dtype=float)
    auth = PatchOfEarthV0(params or WorldParams(), drift_sigma=0.0, drift_mode=drift_mode)
    surr = PatchOfEarthV0(params or WorldParams(), drift_sigma=drift_sigma, drift_mode=drift_mode)
    auth.ray_steps = surr.ray_steps = ray_steps
    if drift_mode == "l3" and g_motion is not None:
        surr._g_motion = g_motion  # surrogate uses the learned velocity law; authentic never does

    auth_curve = np.array([scripted_reach_payoff(auth, t, layout, seed=seed) for t in thrusts])
    surr_curve = np.array([scripted_reach_payoff(surr, t, layout, seed=seed) for t in thrusts])
    avg = 0.5 * (auth_curve + surr_curve)
    oracle = 0.5 * (float(auth_curve.max()) + float(surr_curve.max()))
    blind = float(avg.max())
    return {
        "name": layout.name,
        "thrusts": thrusts,
        "auth_curve": auth_curve,
        "surr_curve": surr_curve,
        "oracle": oracle,
        "blind": blind,
        "gap": oracle - blind,
        "best_auth_thrust": float(thrusts[int(auth_curve.argmax())]),
        "best_surr_thrust": float(thrusts[int(surr_curve.argmax())]),
        "best_blind_thrust": float(thrusts[int(avg.argmax())]),
    }


def reach_normalized_gap(gap: float, reach_range: float) -> float:
    """Value-of-world-identity gap as a fraction of the reach distance.

    The raw ``gap`` from ``value_of_world_identity`` is in payoff (distance) units,
    so it scales with how far the target sits and cannot be compared across layouts
    of different reach. Dividing by ``reach_range`` yields a scale-free steepness -
    the fraction of the intended reach a world-blind controller forfeits by not
    knowing the world - so a payoff-steepness sweep can rank layouts fairly.
    ``nan`` for a non-positive reach (nothing to normalize against)."""
    if reach_range <= 0.0:
        return float("nan")
    return gap / reach_range


def steepness_sweep(
    reaches: Sequence[float],
    horizons: Sequence[int],
    mean_gap_of: Callable[["Layout"], float],
    *,
    baseline_gap: float,
) -> dict[str, Any]:
    """Grid the value-of-world-identity gap over a (reach x horizon) layout lattice.

    ``mean_gap_of`` maps a ``Layout`` to its mean gap across evaluation seeds (injected
    so the expensive L3 map lives in the caller, not here). Each cell also carries the
    reach-normalized gap and the gap as a multiple of ``baseline_gap`` (the pilot's
    primary-treatment gap, ~0.0023) so a candidate config can be read as 'Nx the flat
    pilot'. Ranked TWO ways on purpose: ``best_by_gap`` (raw payoff, what the fitness
    margin cares about) and ``best_by_norm_gap`` (scale-free steepness, which guards
    against a gap that is large only because the reach is far). Deterministic given a
    deterministic ``mean_gap_of``."""
    cells = []
    for r in reaches:
        for h in horizons:
            lay = Layout(reach_range=float(r), horizon=int(h), name=f"r{r}_h{h}")
            gap = float(mean_gap_of(lay))
            cells.append({
                "reach": float(r),
                "horizon": int(h),
                "name": lay.name,
                "gap": gap,
                "norm_gap": reach_normalized_gap(gap, float(r)),
                "ratio_to_baseline": gap / baseline_gap if baseline_gap > 0.0 else float("nan"),
            })
    return {
        "cells": cells,
        "best_by_gap": max(cells, key=lambda c: c["gap"]),
        "best_by_norm_gap": max(cells, key=lambda c: c["norm_gap"]),
        "baseline_gap": float(baseline_gap),
    }


def _bootstrap_ci90(gaps: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    """Seed-level 90% percentile CI of the mean gap (prereg sec. 10). Resamples the
    per-seed gaps with replacement; deterministic given ``rng``."""
    n = len(gaps)
    means = np.array([rng.choice(gaps, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def gate1_exploitability(
    *,
    treatment: Layout,
    control: Layout,
    seeds: list[int],
    drift_sigma: float,
    drift_mode: str = "regime",
    thrusts: np.ndarray | None = None,
    params: WorldParams | None = None,
    ray_steps: int = 5,
    g_motion: Callable | None = None,
    margin: float = 0.005,
    tol: float = 0.005,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Gate 1 (docs/PREREGISTRATION_C.md sec. 7): is the payoff world-only-exploitable?

    Measures the value-of-world-identity gap for the TREATMENT and the CONTROL
    layout across ``seeds`` and adjudicates with a seed-level 90% bootstrap CI:

      * treatment PASSES when its gap CI lower bound clears ``margin`` (world
        identity is exploitable - a world-blind controller leaves payoff on the
        table),
      * control PASSES when its gap CI upper bound stays below ``tol`` (world
        identity is fitness-neutral by construction),
      * gate 1 passes only if BOTH hold.

    For the real certification pass ``drift_mode="l3"`` and the frozen ``g_motion``;
    the ``regime`` default is the cheap machinery stand-in. Deterministic given
    ``rng`` (the bootstrap) since the gaps themselves are seed-deterministic."""
    rng = np.random.default_rng(0) if rng is None else rng
    kw = dict(drift_sigma=drift_sigma, drift_mode=drift_mode, thrusts=thrusts,
              params=params, ray_steps=ray_steps, g_motion=g_motion)
    treat_gaps = np.array([value_of_world_identity(treatment, seed=s, **kw)["gap"] for s in seeds])
    ctrl_gaps = np.array([value_of_world_identity(control, seed=s, **kw)["gap"] for s in seeds])

    t_lo, t_hi = _bootstrap_ci90(treat_gaps, n_boot, rng)
    c_lo, c_hi = _bootstrap_ci90(ctrl_gaps, n_boot, rng)
    passes_treatment = t_lo > margin
    passes_control = c_hi < tol
    return {
        "treatment_gaps": treat_gaps.tolist(),
        "control_gaps": ctrl_gaps.tolist(),
        "treatment_gap_mean": float(treat_gaps.mean()),
        "control_gap_mean": float(ctrl_gaps.mean()),
        "treatment_ci90": [t_lo, t_hi],
        "control_ci90": [c_lo, c_hi],
        "margin": margin,
        "tol": tol,
        "n_boot": n_boot,
        "n_seeds": len(seeds),
        "passes_treatment": bool(passes_treatment),
        "passes_control": bool(passes_control),
        "passes_gate1": bool(passes_treatment and passes_control),
    }
