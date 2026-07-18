"""ITASORL - Experiment C gate-1 payoff-steepness sweep against the frozen L3 map.

The milestone-3 emergence pilot came back a pre-registered null (docs/FINDINGS.md
sec. 13): under selection the treatment lineage did not build a persistent, heritable
world-detector beyond its gen-0 reactive signal. The gate-1 L3 certification had already
flagged WHY this was at risk - the scripted world-identity value gap at the primary
treatment layout (reach=0.25, horizon=30) was only ~0.0023, i.e. the payoff surface is
nearly flat near its optimum, so a world-blind forager forfeits almost nothing by
ignoring the velocity law.

Before committing to a full prereg sec. 8 redesign (a fresh, expensive cycle), this
script cheaply de-risks it: it grids the scripted-oracle value-of-world-identity gap
over a (reach x horizon) lattice against the SAME frozen L3 map (train_g_motion on the
gate-0 recipe: hidden=8, seed=0, world P), looking for a layout where the gap is
MATERIALLY larger than the flat 0.0023 pilot baseline. A steeper cell would say 'a
redesign with harsher coasting / longer horizons could restore a real fitness incentive
to detect'; a lattice that stays flat everywhere would say 'the scripted controller
cannot express the coupling, route to a richer-controller redesign instead'.

This is a scripted-oracle probe (no evolution, no panels), so it is cheap. It reuses the
one-sided-lower-bound caveat from run_expC_gate1.py: a large cell PROVES exploitability,
but a flat lattice is only a lower bound - a recurrent forager could still extract more
than a constant-thrust reach can. The sweep is descriptive input to the redesign
go/no-go, never a silent change of world.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time

import numpy as np

from itasorl.experiment_c_gate1 import (
    Layout,
    reach_normalized_gap,
    steepness_sweep,
    value_of_world_identity,
)
from itasorl.surrogate_l3 import train_g_motion
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)  # the run_expB2 / gate-0 organism world
PILOT_BASELINE_GAP = 0.0023  # primary treatment (reach=0.25, h=30) gap from the L3 certification


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8, help="frozen L3 capacity (gate-0 headline)")
    ap.add_argument("--seed", type=int, default=0, help="L3 training seed (gate-0 bit-identical)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(7100, 7106)),
                    help="per-cell layout evaluation seeds (kept small - the sweep is "
                         "descriptive). Deliberately DISJOINT from run_expC_gate1's "
                         "certification seeds (7000-7009): certifying a sweep-selected "
                         "layout on the seeds it was selected on would inherit the "
                         "selection maximum (winner's curse). The recorded 2026-07-17 "
                         "sweep used 7000-7005, before this guard.")
    ap.add_argument("--reaches", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.5, 0.75])
    ap.add_argument("--horizons", type=int, nargs="+", default=[20, 30, 40, 60, 80])
    ap.add_argument("--ctrl-reach", type=float, default=0.05)
    ap.add_argument("--ctrl-horizon", type=int, default=1)
    ap.add_argument("--n-thrusts", type=int, default=21)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--baseline-gap", type=float, default=PILOT_BASELINE_GAP)
    ap.add_argument("--json", default="fullruns/expC_gate1_sweep/steepness.json")
    args = ap.parse_args()

    t0 = time.time()
    thrusts = np.linspace(0.0, 1.0, args.n_thrusts)
    n_cells = len(args.reaches) * len(args.horizons)
    print(f"[expC sweep] world P (k=1.5, gravity=0.4)  frozen L3 hidden={args.hidden} "
          f"seed={args.seed}  {n_cells} cells x {len(args.seeds)} seeds", flush=True)

    # the frozen L3 map - bit-identical to the gate-0 / organism-run surrogate.
    g = train_g_motion(hidden=args.hidden, seed=args.seed, params=P)
    print(f"[expC sweep] frozen L3 map trained ({time.time()-t0:.0f}s)", flush=True)

    l3_kw = dict(drift_sigma=1.0, drift_mode="l3", thrusts=thrusts, params=P,
                 ray_steps=args.ray_steps, g_motion=g)

    def mean_gap_of(lay: Layout) -> float:
        return float(np.mean([value_of_world_identity(lay, seed=s, **l3_kw)["gap"]
                              for s in args.seeds]))

    result = steepness_sweep(args.reaches, args.horizons, mean_gap_of,
                             baseline_gap=args.baseline_gap)

    # determinism: a second full sweep must reproduce every cell gap to the bit.
    result2 = steepness_sweep(args.reaches, args.horizons, mean_gap_of,
                              baseline_gap=args.baseline_gap)
    bit_repro = ([c["gap"] for c in result["cells"]] == [c["gap"] for c in result2["cells"]])

    # from-rest control floor: analytically world-invariant, so the gap should stay ~0.
    ctrl = Layout(reach_range=args.ctrl_reach, horizon=args.ctrl_horizon, name="control")
    ctrl_gap = mean_gap_of(ctrl)

    for c in result["cells"]:
        print(f"[expC sweep]   reach={c['reach']:.2f} h={c['horizon']:>3d}  "
              f"gap={c['gap']:.5f}  norm={c['norm_gap']:.4f}  "
              f"x{c['ratio_to_baseline']:.1f} baseline", flush=True)
    bg = result["best_by_gap"]
    bn = result["best_by_norm_gap"]
    print(f"[expC sweep] BEST raw gap: reach={bg['reach']:.2f} h={bg['horizon']} "
          f"gap={bg['gap']:.5f} (x{bg['ratio_to_baseline']:.1f} the {args.baseline_gap} baseline)",
          flush=True)
    print(f"[expC sweep] BEST steepness: reach={bn['reach']:.2f} h={bn['horizon']} "
          f"norm={bn['norm_gap']:.4f} gap={bn['gap']:.5f}", flush=True)
    print(f"[expC sweep] control floor gap={ctrl_gap:.6f} "
          f"(reach={args.ctrl_reach}, h={args.ctrl_horizon})  determinism={bit_repro}", flush=True)

    out = {
        "world": "P(k_land=1.5, k_water=1.5, gravity=0.4)",
        "surrogate": f"frozen L3 G_motion (hidden={args.hidden}, seed={args.seed})",
        "config": vars(args),
        "baseline_gap": args.baseline_gap,
        "cells": result["cells"],
        "best_by_gap": bg,
        "best_by_norm_gap": bn,
        "control_floor": {"reach": args.ctrl_reach, "horizon": args.ctrl_horizon,
                          "gap": ctrl_gap, "norm_gap": reach_normalized_gap(ctrl_gap, args.ctrl_reach)},
        "determinism_bit_reproducible": bool(bit_repro),
        "wall_seconds": round(time.time() - t0, 1),
    }
    d = os.path.dirname(args.json)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[expC sweep] DONE in {out['wall_seconds']}s -> {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
