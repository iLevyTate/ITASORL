"""ITASORL - Experiment C milestone-3 EMERGENCE pilot (docs/PREREGISTRATION_C.md sec. 13.3).

The first run that actually asks Experiment C's question: does Darwinian selection
build a PERSISTENT, heritable world-identity detector where within-life learning (the
L3 arc) found only a reactive one? Two arms, paired per lineage:

  * TREATMENT: the frozen sparse/far SURVIVAL_FOOD layout. Pellets sit beyond `reach`,
    so a policy must build velocity and coast to forage - which makes the ONE primitive
    separating the worlds (the velocity law, patch_of_earth.py:174-177) fitness-relevant.
    De-risk measured world-sensitivity ~0.32 of foraging return here.
  * CONTROL: a dense/near, world-INVARIANT layout (food_override n_pellets=80, reach=0.25).
    Food is within reach at spawn, so foraging needs no coasting and the velocity law is
    irrelevant - detecting the world buys ~0 fitness (de-risk sensitivity ~0.038, 8.4x
    smaller). "The control is the claim" (sec. 11): only the treatment-minus-control gain
    licenses a selection claim.

DESIGN CHOICES (flagged; the one non-obvious call is the threshold):
  - Paired lineages: for each lineage seed the two arms share the gen-0 population, the
    world seed_base, AND the mutation/selection rng seed, so the ONLY difference is the
    food layout (the selection pressure). Variance-reducing and confound-tight.
  - MATCHED SELECTION INTENSITY (not in the prereg letter; a defensible pilot call): the
    treatment forages at a loss (mean return ~-1.5) and the control at a profit (~+0.94),
    so a single ABSOLUTE reproduction threshold would make selection bite hard in one arm
    and not at all in the other - confounding "detection is useful" with "selection was
    stronger". Instead each arm's fixed threshold is the `--q` quantile (default median)
    of ITS OWN gen-0 fitness, so both arms start at the same qualifier fraction. Still a
    FIXED threshold (not per-generation truncation), matching the pre-registered mechanism
    (itasorl/neuroevolution.reproduce). Surface this to the user before the confirmatory run.
  - No within-life gradient step (Option P): a generational detection gain is inherited,
    not learned.
  - The common-garden DETECTION panel is the SAME fixed yardstick (frozen layout) for both
    arms every measurement, so the arms are comparable and survival never leaks into
    detection (sec. 5). Measured at gen 0 (shared) and the final generation (per arm).

This is a PILOT: 3 lineage seeds give a wide (df=2) contrast CI by design, so a null
`emergence_claim` boolean is expected - the raw per-seed pieces are what inform the
sec. 8/11 scale-or-redesign decision, not the boolean.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import copy
import json
import os
import time
from functools import partial

import numpy as np
import torch

import itasorl.experiment_b2 as b2
from itasorl.agent_ac import RecurrentActorCritic
from itasorl.experiment_c import (common_garden_panel, emergence_contrast,
                                   mixed_world_fitness)
from itasorl.neuroevolution import evolve
from itasorl.patch_of_earth import PatchOfEarthV0
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)  # gate-0 / organism world


def build_population(n, *, embed, hidden, seed0):
    w = PatchOfEarthV0()
    obs, act = w.obs_spec.size, w.action_spec.size
    pop = []
    for i in range(n):
        torch.manual_seed(seed0 + i)
        pop.append(RecurrentActorCritic(obs, act, embed=embed, hidden=hidden, world_model=False))
    return pop


def run_arm(pop0, food_override, *, generations, sigma, drift_sigma, n_eps, max_steps,
            seed_base, rng_seed, quantile, device="cpu"):
    """Evolve one arm from a COPY of the shared gen-0 population. The fixed
    reproduction threshold is the `quantile` of this arm's gen-0 fitness (matched
    selection intensity, see module docstring). Returns final pop + fitness series."""
    fit_kw = dict(drift_sigma=drift_sigma, n_eps_per_world=n_eps, max_steps=max_steps,
                  seed_base=seed_base, food_override=food_override, params=P, device=device)
    gen0_fit = mixed_world_fitness(pop0, **fit_kw)
    threshold = float(np.quantile(gen0_fit, quantile))
    fit = partial(mixed_world_fitness, **fit_kw)
    final_pop, history = evolve(copy.deepcopy(pop0), fit, generations=generations,
                                threshold=threshold, sigma=sigma,
                                rng=np.random.default_rng(rng_seed))
    series = [round(r["mean_fitness"], 8) for r in history]
    return final_pop, threshold, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="population / carrying capacity")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="lineage seeds")
    ap.add_argument("--sigma", type=float, default=0.03, help="mutation sigma (milestone-1)")
    ap.add_argument("--q", type=float, default=0.5, help="gen-0 fitness quantile -> threshold")
    ap.add_argument("--drift-sigma", type=float, default=1.0, help="l3 mode: installs map only")
    ap.add_argument("--n-eps", type=int, default=2, help="episodes per world per lifetime")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--embed", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--l3-hidden", type=int, default=8, help="frozen L3 capacity (gate-0 headline)")
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--ctrl-n-pellets", type=int, default=80)
    ap.add_argument("--ctrl-reach", type=float, default=0.25)
    ap.add_argument("--panel-pairs", type=int, default=110)
    ap.add_argument("--panel-prefix", type=int, default=20)
    ap.add_argument("--panel-tail", type=int, default=24)
    ap.add_argument("--panel-seed-base", type=int, default=930_000)
    ap.add_argument("--base-seed-base", type=int, default=320_000)
    ap.add_argument("--json", default="fullruns/expC_milestone3/emergence_pilot.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device for agent forward passes. The world sim is numpy on "
                         "CPU either way, so cuda helps only if the nets are large "
                         "enough to amortize transfer overhead; benchmark one seed "
                         "both ways before committing to a device.")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    t0 = time.time()
    control_food = {"n_pellets": args.ctrl_n_pellets, "reach": args.ctrl_reach}
    print(f"[expC m3] N={args.n} G={args.generations} seeds={args.seeds} sigma={args.sigma} "
          f"q={args.q} n_eps={args.n_eps} steps={args.max_steps} "
          f"embed={args.embed} hidden={args.hidden}", flush=True)
    print(f"[expC m3] treatment=frozen SURVIVAL_FOOD  control={control_food}", flush=True)

    # frozen L3 map, bit-identical to gate-0 / the organism run
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=args.l3_hidden, seed=args.l3_seed, params=P)
    print(f"[expC m3] frozen L3 map installed ({time.time()-t0:.0f}s)", flush=True)

    # params=P: the fitness lifetimes and the detection panel must run on the SAME
    # world the frozen L3 map was trained on, or the auth-vs-surrogate contrast
    # includes the whole P-vs-default parameter gap instead of the velocity law
    # (and the "world" field below would be false provenance).
    panel_kw = dict(drift_sigma=args.drift_sigma, n_pairs=args.panel_pairs,
                    prefix_steps=args.panel_prefix, tail_steps=args.panel_tail,
                    seed_base=args.panel_seed_base, params=P, device=args.device)
    arm_kw = dict(generations=args.generations, sigma=args.sigma, drift_sigma=args.drift_sigma,
                  n_eps=args.n_eps, max_steps=args.max_steps, quantile=args.q,
                  device=args.device)

    per_seed = []
    gen0_aurocs, final_t_aurocs, final_c_aurocs = [], [], []
    for s in args.seeds:
        ts = time.time()
        pop_seed = 500 + s * 50
        seed_base = args.base_seed_base + s * 10_000
        pop0 = build_population(args.n, embed=args.embed, hidden=args.hidden, seed0=pop_seed)
        for a_ in pop0:
            a_.to(args.device)

        panel_gen0 = common_garden_panel(pop0, 0, **panel_kw)  # shared by both arms
        final_t, thr_t, series_t = run_arm(pop0, None, seed_base=seed_base, rng_seed=s, **arm_kw)
        final_c, thr_c, series_c = run_arm(pop0, control_food, seed_base=seed_base, rng_seed=s, **arm_kw)
        panel_t = common_garden_panel(final_t, args.generations, **panel_kw)
        panel_c = common_garden_panel(final_c, args.generations, **panel_kw)

        a0 = panel_gen0["cg_tail_target"]
        at = panel_t["cg_tail_target"]
        ac = panel_c["cg_tail_target"]
        gen0_aurocs.append(a0)
        final_t_aurocs.append(at)
        final_c_aurocs.append(ac)
        per_seed.append({
            "seed": s, "threshold_treat": thr_t, "threshold_ctrl": thr_c,
            "auroc_gen0": a0, "auroc_final_treat": at, "auroc_final_ctrl": ac,
            "fit_series_treat": series_t, "fit_series_ctrl": series_c,
            "fit_delta_treat": round(series_t[-1] - series_t[0], 6),
            "fit_delta_ctrl": round(series_c[-1] - series_c[0], 6),
            "l0_gen0": panel_gen0["l0_auroc"], "l0_final_treat": panel_t["l0_auroc"],
            "l0_final_ctrl": panel_c["l0_auroc"],
            "survival_gen0": panel_gen0["survival"],
            "survival_final_treat": panel_t["survival"], "survival_final_ctrl": panel_c["survival"],
        })
        print(f"[expC m3] seed {s}: AUROC gen0={a0:.3f} treat={at:.3f} ctrl={ac:.3f} "
              f"| fit d_treat={series_t[-1]-series_t[0]:+.3f} d_ctrl={series_c[-1]-series_c[0]:+.3f} "
              f"| thr_t={thr_t:.3f} thr_c={thr_c:.3f}  ({time.time()-ts:.0f}s)", flush=True)

    est = emergence_contrast(gen0_aurocs, final_t_aurocs, final_c_aurocs,
                             rng=np.random.default_rng(0))
    print(f"[expC m3] CONTRAST mean={est['mean_contrast']:+.4f} "
          f"t_CI90=[{est['t_ci90'][0]:+.4f},{est['t_ci90'][1]:+.4f}] "
          f"boot_CI90=[{est['boot_ci90'][0]:+.4f},{est['boot_ci90'][1]:+.4f}]", flush=True)
    print(f"[expC m3] mean final treat AUROC={est['mean_final_treat_auroc']:.3f}  "
          f"ci_excl_0={est['ci_excludes_zero']} sesoi={est['meets_sesoi']} "
          f"floor={est['meets_auroc_floor']} -> CLAIM={est['emergence_claim']}", flush=True)

    # determinism: rerun the first seed's treatment arm, assert bit-identical fitness series
    s0 = args.seeds[0]
    pop0 = build_population(args.n, embed=args.embed, hidden=args.hidden, seed0=500 + s0 * 50)
    for a_ in pop0:
        a_.to(args.device)
    _, _, series_repro = run_arm(pop0, None, seed_base=args.base_seed_base + s0 * 10_000,
                                 rng_seed=s0, **arm_kw)
    bit_repro = (series_repro == per_seed[0]["fit_series_treat"])
    gate2_treat = any(d["fit_delta_treat"] > 0 for d in per_seed)
    gate2_ctrl = any(d["fit_delta_ctrl"] > 0 for d in per_seed)
    print(f"[expC m3] determinism (seed {s0} treat): bit-reproducible={bit_repro}  "
          f"gate2 fitness-moves treat={gate2_treat} ctrl={gate2_ctrl}", flush=True)

    out = {
        "world": "P(k_land=1.5, k_water=1.5, gravity=0.4)",
        "surrogate": f"frozen L3 G_motion (hidden={args.l3_hidden}, seed={args.l3_seed})",
        "config": vars(args), "control_food": control_food,
        "per_seed": per_seed,
        "estimand": est,
        "gate2_fitness_moves_treat": bool(gate2_treat),
        "gate2_fitness_moves_ctrl": bool(gate2_ctrl),
        "determinism_bit_reproducible": bool(bit_repro),
        "wall_seconds": round(time.time() - t0, 1),
    }
    d = os.path.dirname(args.json)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[expC m3] DONE in {out['wall_seconds']}s -> {args.json}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
