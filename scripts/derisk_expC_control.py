"""ITASORL - Experiment C control-layout de-risk (docs/PREREGISTRATION_C.md sec. 11).

Before spending milestone-3 pilot compute we check the load-bearing design
assumption: the CONTROL arm's dense/near pellet layout must make world identity
FITNESS-NEUTRAL (detecting the world buys no foraging return), while the TREATMENT
arm's frozen sparse/far layout keeps it world-coupled. If the control is not
actually world-invariant the treatment-minus-control contrast is uninterpretable
("the control is the claim", sec. 11).

Two cheap, deterministic probes against the frozen L3 map on world P (the gate-0 /
gate-1 recipe: train_g_motion hidden=8, seed=0):

  (1) GEOMETRY: mean distance from the agent to its nearest pellet at reset, over
      many world seeds. The dense/near control should sit within `reach` (eat-able
      with little travel) while the sparse/far treatment forces sustained travel.
      Coasting is the ONLY thing that exposes the one primitive separating the
      worlds (the velocity update, patch_of_earth.py:174-177), so a no-coast layout
      is world-blind by construction.

  (2) FITNESS SENSITIVITY: for a fixed population, the per-policy gap between
      AUTHENTIC (drift_sigma=0, analytic law) and SURROGATE (frozen L3 velocity
      map) foraging return, using the exact two-leg rollout `mixed_world_fitness`
      runs. control sensitivity must be << treatment sensitivity (ideally ~0). A
      non-zero TREATMENT gap on the same population is its own engagement proof - if
      both were ~0 the population would be inert and the probe inconclusive, not a
      demonstration of world-invariance.

INTERPRETATION: a small control gap is a NECESSARY (conservative) condition, not a
selection result - it says "for these policies the world does not change foraging
outcome", which is sufficient for "detection buys no fitness". A large control gap
routes to a sec. 8 control-layout redesign, never to a silent run.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
import torch

import itasorl.experiment_b2 as b2
from itasorl.agent_ac import RecurrentActorCritic
from itasorl.experiment_b2 import (RunningNorm, _food_potential, _seeds,
                                    collect_episodes_ac, make_world)
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


def geometry_probe(food_override, *, seeds, ray_steps):
    """Mean distance to the nearest pellet at reset over `seeds` (authentic world;
    pellet placement is world-independent so one world suffices)."""
    dists = []
    for s in seeds:
        w = make_world(P, 0.0, ray_steps, food_override)
        w.reset(_seeds(s))
        dists.append(-_food_potential(w))  # _food_potential returns -nearest-distance
    return {"mean_nearest": float(np.mean(dists)), "reach": float(make_world(P, 0.0, ray_steps, food_override).reach)}


def fitness_sensitivity(pop, food_override, *, drift_sigma, n_eps, max_steps, seed_base, ray_steps):
    """Per-policy |authentic return - surrogate return|. Faithful to the two-leg
    rollout in mixed_world_fitness: one frozen RunningNorm per agent reused across
    legs, deterministic mode actions, no norm updates."""
    per_gap, auth_m, surr_m, lens = [], [], [], []
    for agent in pop:
        agent.train(False)
        n = RunningNorm(agent.obs_dim)  # frozen at init (update_norm=False) -> identity-ish
        legs = {}
        for label, dsig in (("auth", 0.0), ("surr", drift_sigma)):
            b = collect_episodes_ac(agent, n, P, dsig, n_eps=n_eps, max_steps=max_steps,
                                    device="cpu", seed_base=seed_base, ray_steps=ray_steps,
                                    deterministic=True, update_norm=False,
                                    food_override=food_override)
            legs[label] = float(np.asarray(b["ret"]).mean())
            lens.append(float(np.asarray(b["lengths"]).mean()))
        per_gap.append(abs(legs["auth"] - legs["surr"]))
        auth_m.append(legs["auth"])
        surr_m.append(legs["surr"])
    return {"mean_abs_gap": float(np.mean(per_gap)), "max_abs_gap": float(np.max(per_gap)),
            "mean_auth": float(np.mean(auth_m)), "mean_surr": float(np.mean(surr_m)),
            "mean_len": float(np.mean(lens)), "per_gap": [round(x, 6) for x in per_gap]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="population size (shared across arms)")
    ap.add_argument("--embed", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--l3-hidden", type=int, default=8, help="frozen L3 capacity (gate-0 headline)")
    ap.add_argument("--l3-seed", type=int, default=0)
    ap.add_argument("--drift-sigma", type=float, default=1.0, help="l3-mode: installs the map only")
    ap.add_argument("--n-eps", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--seed-base", type=int, default=320_000)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--geom-seeds", type=int, default=200)
    # control = dense/near (world-invariant); treatment = frozen SURVIVAL_FOOD (None)
    ap.add_argument("--ctrl-n-pellets", type=int, default=80)
    ap.add_argument("--ctrl-reach", type=float, default=0.25)
    ap.add_argument("--json", default="fullruns/expC_derisk/control_layout.json")
    args = ap.parse_args()

    t0 = time.time()
    control_food = {"n_pellets": args.ctrl_n_pellets, "reach": args.ctrl_reach}
    print(f"[derisk] world P  frozen L3 hidden={args.l3_hidden} seed={args.l3_seed}  "
          f"N={args.n} steps={args.max_steps} eps={args.n_eps}", flush=True)
    print(f"[derisk] treatment=frozen SURVIVAL_FOOD={b2.SURVIVAL_FOOD}  control={control_food}",
          flush=True)

    # install the frozen L3 map exactly as gate-0 / the organism run does
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=args.l3_hidden, seed=args.l3_seed, params=P)
    print(f"[derisk] frozen L3 map installed ({time.time()-t0:.0f}s)", flush=True)

    geom_seeds = list(range(args.seed_base, args.seed_base + args.geom_seeds))
    geom_treat = geometry_probe(None, seeds=geom_seeds, ray_steps=args.ray_steps)
    geom_ctrl = geometry_probe(control_food, seeds=geom_seeds, ray_steps=args.ray_steps)
    print(f"[derisk] GEOMETRY treatment: mean_nearest={geom_treat['mean_nearest']:.4f} "
          f"reach={geom_treat['reach']:.3f}", flush=True)
    print(f"[derisk] GEOMETRY control:   mean_nearest={geom_ctrl['mean_nearest']:.4f} "
          f"reach={geom_ctrl['reach']:.3f}", flush=True)

    fs_kw = dict(drift_sigma=args.drift_sigma, n_eps=args.n_eps, max_steps=args.max_steps,
                 seed_base=args.seed_base, ray_steps=args.ray_steps)
    pop = build_population(args.n, embed=args.embed, hidden=args.hidden, seed0=500)
    fit_treat = fitness_sensitivity(pop, None, **fs_kw)
    pop = build_population(args.n, embed=args.embed, hidden=args.hidden, seed0=500)  # same pop
    fit_ctrl = fitness_sensitivity(pop, control_food, **fs_kw)
    print(f"[derisk] FITNESS treatment: mean|auth-surr|={fit_treat['mean_abs_gap']:.5f} "
          f"max={fit_treat['max_abs_gap']:.5f} auth={fit_treat['mean_auth']:.4f} "
          f"surr={fit_treat['mean_surr']:.4f} len={fit_treat['mean_len']:.1f}", flush=True)
    print(f"[derisk] FITNESS control:   mean|auth-surr|={fit_ctrl['mean_abs_gap']:.5f} "
          f"max={fit_ctrl['max_abs_gap']:.5f} auth={fit_ctrl['mean_auth']:.4f} "
          f"surr={fit_ctrl['mean_surr']:.4f} len={fit_ctrl['mean_len']:.1f}", flush=True)

    engaged = fit_treat["mean_abs_gap"] > 1e-6
    control_invariant = fit_ctrl["mean_abs_gap"] < fit_treat["mean_abs_gap"] * 0.25
    verdict = ("PASS" if (engaged and control_invariant)
               else "INCONCLUSIVE(inert pop)" if not engaged else "FAIL(control still coupled)")
    print(f"[derisk] VERDICT={verdict}  engaged={engaged} "
          f"control<<treatment={control_invariant}", flush=True)

    out = {
        "world": "P(k_land=1.5, k_water=1.5, gravity=0.4)",
        "surrogate": f"frozen L3 G_motion (hidden={args.l3_hidden}, seed={args.l3_seed})",
        "config": vars(args), "control_food": control_food,
        "geometry": {"treatment": geom_treat, "control": geom_ctrl},
        "fitness_sensitivity": {"treatment": fit_treat, "control": fit_ctrl},
        "engaged": bool(engaged), "control_world_invariant": bool(control_invariant),
        "verdict": verdict, "wall_seconds": round(time.time() - t0, 1),
    }
    d = os.path.dirname(args.json)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[derisk] DONE in {out['wall_seconds']}s -> {args.json}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
