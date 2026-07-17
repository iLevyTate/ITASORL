"""ITASORL - Experiment C milestone-1 pilot (docs/PREREGISTRATION_C.md sec. 13.1).

Determinism + selection-works, CPU/GPU-light. NOT an emergence run: it verifies
the machinery only -

  * gate 2 (selection-works): does mean lifetime foraging fitness MOVE across
    generations under threshold-triggered reproduction? Swept over a small set of
    mutation sigmas - this is the one calibration the prereg leaves open (sec. 9),
    tuned to put the per-generation fitness gain in a workable band.
  * determinism: a fixed seed reproduces the whole fitness series to the bit.
  * panel-in-loop: the fixed common-garden detection panel runs at gen 0 and the
    final generation, proving it executes inside the loop and reports survival
    SEPARATELY from detection (sec. 5). Detection VALUES are not a result here.

The surrogate is the cheap default DRIFT_MODE (ar1), a stand-in sufficient for a
machinery/selection check; the real emergence pilot (milestone 3) swaps in the
frozen L3 momentum fingerprint and the treatment/control payoff geometry.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time
from functools import partial

import numpy as np
import torch

from itasorl.agent_ac import RecurrentActorCritic
from itasorl.experiment_b2 import DRIFT_MODE
from itasorl.experiment_c import common_garden_panel, mixed_world_fitness
from itasorl.neuroevolution import evolve
from itasorl.patch_of_earth import PatchOfEarthV0


def build_population(n, *, embed, hidden, seed0):
    w = PatchOfEarthV0()
    obs, act = w.obs_spec.size, w.action_spec.size
    pop = []
    for i in range(n):
        torch.manual_seed(seed0 + i)
        pop.append(RecurrentActorCritic(obs, act, embed=embed, hidden=hidden, world_model=False))
    return pop


def fitness_series(pop, *, generations, threshold, sigma, drift_sigma, n_eps, max_steps,
                   seed_base, rng_seed):
    fit = partial(mixed_world_fitness, drift_sigma=drift_sigma, n_eps_per_world=n_eps,
                  max_steps=max_steps, seed_base=seed_base)
    final_pop, history = evolve(pop, fit, generations=generations, threshold=threshold,
                                sigma=sigma, rng=np.random.default_rng(rng_seed))
    return final_pop, [round(r["mean_fitness"], 8) for r in history], history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.03, 0.08])
    ap.add_argument("--threshold", type=float, default=-0.05,
                    help="absolute lifetime-return threshold to reproduce (NOT a quantile)")
    ap.add_argument("--drift-sigma", type=float, default=0.02)
    ap.add_argument("--n-eps", type=int, default=3, help="episodes per world per lifetime")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--embed", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=320_000)
    ap.add_argument("--panel-pairs", type=int, default=12)
    ap.add_argument("--json", default="fullruns/expC_milestone1/pilot.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[expC m1] N={args.n} G={args.generations} sigmas={args.sigmas} "
          f"thr={args.threshold} drift={args.drift_sigma} eps={args.n_eps} "
          f"steps={args.max_steps} embed={args.embed} hidden={args.hidden}", flush=True)

    results = {}
    best_sigma, best_delta, best_final_pop = None, -np.inf, None
    for sigma in args.sigmas:
        pop = build_population(args.n, embed=args.embed, hidden=args.hidden, seed0=500)
        ts = time.time()
        final_pop, series, _ = fitness_series(
            pop, generations=args.generations, threshold=args.threshold, sigma=sigma,
            drift_sigma=args.drift_sigma, n_eps=args.n_eps, max_steps=args.max_steps,
            seed_base=args.seed_base, rng_seed=0)
        delta = series[-1] - series[0]
        results[f"sigma={sigma}"] = {"mean_fitness_series": series,
                                     "gen0": series[0], "final": series[-1], "delta": delta}
        print(f"[expC m1] sigma={sigma}: gen0={series[0]:.4f} final={series[-1]:.4f} "
              f"delta={delta:+.4f}  ({time.time()-ts:.0f}s)", flush=True)
        if delta > best_delta:
            best_sigma, best_delta, best_final_pop = sigma, delta, final_pop

    # determinism: rerun the best sigma from a fresh population, assert bit-identical series
    pop = build_population(args.n, embed=args.embed, hidden=args.hidden, seed0=500)
    _, series2, _ = fitness_series(
        pop, generations=args.generations, threshold=args.threshold, sigma=best_sigma,
        drift_sigma=args.drift_sigma, n_eps=args.n_eps, max_steps=args.max_steps,
        seed_base=args.seed_base, rng_seed=0)
    bit_repro = (series2 == results[f"sigma={best_sigma}"]["mean_fitness_series"])
    print(f"[expC m1] determinism (sigma={best_sigma}): bit-reproducible={bit_repro}", flush=True)

    # panel-in-loop snapshot: gen 0 (fresh pop) vs the best sigma's evolved final pop
    pop0 = build_population(args.n, embed=args.embed, hidden=args.hidden, seed0=500)
    panel_kw = dict(drift_sigma=args.drift_sigma, n_pairs=args.panel_pairs, prefix_steps=20,
                    tail_steps=24, seed_base=930_000)
    panel_gen0 = common_garden_panel(pop0, 0, **panel_kw)
    panel_final = common_garden_panel(best_final_pop, args.generations, **panel_kw)
    print(f"[expC m1] panel gen0: det={panel_gen0['cg_tail_target']:.3f} "
          f"L0={panel_gen0['l0_auroc']:.3f} surv={panel_gen0['survival']}", flush=True)
    print(f"[expC m1] panel final: det={panel_final['cg_tail_target']:.3f} "
          f"L0={panel_final['l0_auroc']:.3f} surv={panel_final['survival']}", flush=True)

    gate2_pass = best_delta > 0.0
    out = {
        "config": vars(args),
        "drift_mode": DRIFT_MODE,
        "results": results,
        "best_sigma": best_sigma,
        "best_delta": best_delta,
        "gate2_fitness_moves": bool(gate2_pass),
        "determinism_bit_reproducible": bool(bit_repro),
        "panel_gen0": panel_gen0,
        "panel_final": panel_final,
        "wall_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[expC m1] DONE in {out['wall_seconds']}s -> {args.json}", flush=True)
    print(f"[expC m1] GATE2 fitness-moves={gate2_pass} (best sigma={best_sigma}, "
          f"delta={best_delta:+.4f}); determinism={bit_repro}", flush=True)


if __name__ == "__main__":
    main()
