"""ITASORL - Experiment C gate 1: world-only-exploitability against the frozen L3 map.

Gate 1 (docs/PREREGISTRATION_C.md sec. 7) certifies the fitness coupling BEFORE any
selection is run: is world identity worth knowing? We answer with the scripted
momentum-to-target oracle (itasorl.experiment_c_gate1) rather than a learned policy,
so the measurement isolates the single primitive that separates the worlds - the
velocity update at patch_of_earth.py:177 - with no training noise and bit-for-bit
determinism.

The surrogate here is the REAL frozen L3 map (surrogate_l3.train_g_motion on the same
recipe gate 0 certified: hidden=8, seed=0, world P), NOT the cheap `regime` stand-in
the unit tests use. Authentic worlds keep the analytic law; only the surrogate's
velocity update is replaced by the learned G_motion, so the oracle-minus-blind payoff
gap is exactly the value of the L3 fingerprint for a momentum-to-target forager.

Pre-specified layouts, fixed a priori (no post-hoc layout selection - that would be
fishing, docs/ITASORL.md researcher-degrees-of-freedom note):
  treatment = a straight-ahead reach long enough for velocity to build so the
              learned-vs-analytic velocity difference bites (reach=0.25, horizon=30).
  control   = a from-rest horizon-1 reach; analytically world-invariant, but under
              the learned map that invariance is MEASURED not assumed (sec. 7).

A horizon sweep over the treatment is reported as DESCRIPTIVE context (how the gap
grows as velocity accumulates); the PASS/FAIL adjudication uses ONLY the single
pre-specified primary treatment layout, so there is no multiple-comparisons leak.

INTERPRETATION (read before trusting a null): the scripted constant-thrust oracle is
a conservative, ONE-SIDED probe. A treatment gap CI above `margin` proves world
identity is fitness-exploitable (a world-blind controller leaves payoff on the table);
a near-zero gap is only a LOWER BOUND - it does not prove no coupling, because a
constant-thrust single-pellet controller cannot express what a recurrent forager
could. A weak result therefore routes to the prereg sec. 8 redesign discussion, never
to a silent change of world.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time

import numpy as np

from itasorl.experiment_c_gate1 import Layout, gate1_exploitability, value_of_world_identity
from itasorl.surrogate_l3 import train_g_motion
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)  # the run_expB2 / gate-0 organism world


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8, help="frozen L3 capacity (gate-0 headline)")
    ap.add_argument("--seed", type=int, default=0, help="L3 training seed (gate-0 bit-identical)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(7000, 7010)), help="per-seed layout evaluation seeds")
    ap.add_argument("--treat-reach", type=float, default=0.25)
    ap.add_argument("--treat-horizon", type=int, default=30, help="PRIMARY treatment horizon")
    ap.add_argument("--ctrl-reach", type=float, default=0.05)
    ap.add_argument("--ctrl-horizon", type=int, default=1)
    ap.add_argument("--horizon-sweep", type=int, nargs="+", default=[10, 20, 30, 40],
                    help="descriptive-only treatment horizons")
    ap.add_argument("--n-thrusts", type=int, default=21)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--ray-steps", type=int, default=5)
    ap.add_argument("--json", default="fullruns/expC_gate1/l3_certification.json")
    args = ap.parse_args()

    t0 = time.time()
    thrusts = np.linspace(0.0, 1.0, args.n_thrusts)
    print(f"[expC gate1] world P (k=1.5, gravity=0.4)  frozen L3 hidden={args.hidden} "
          f"seed={args.seed}  seeds={len(args.seeds)}", flush=True)

    # the frozen L3 map - bit-identical to the gate-0 / organism-run surrogate.
    g = train_g_motion(hidden=args.hidden, seed=args.seed, params=P)
    print(f"[expC gate1] frozen L3 map trained ({time.time()-t0:.0f}s)", flush=True)

    treat = Layout(reach_range=args.treat_reach, horizon=args.treat_horizon, name="treatment")
    ctrl = Layout(reach_range=args.ctrl_reach, horizon=args.ctrl_horizon, name="control")
    l3_kw = dict(drift_sigma=1.0, drift_mode="l3", thrusts=thrusts, params=P,
                 ray_steps=args.ray_steps, g_motion=g)

    # PRIMARY decision: the single pre-specified treatment vs the from-rest control.
    gate = gate1_exploitability(treatment=treat, control=ctrl, seeds=args.seeds,
                                margin=args.margin, tol=args.tol, n_boot=args.n_boot,
                                rng=np.random.default_rng(0), **l3_kw)
    # determinism: a second call must reproduce the per-seed gaps to the bit.
    gate2 = gate1_exploitability(treatment=treat, control=ctrl, seeds=args.seeds,
                                 margin=args.margin, tol=args.tol, n_boot=args.n_boot,
                                 rng=np.random.default_rng(0), **l3_kw)
    bit_repro = (gate["treatment_gaps"] == gate2["treatment_gaps"]
                 and gate["control_gaps"] == gate2["control_gaps"])

    print(f"[expC gate1] TREATMENT gap mean={gate['treatment_gap_mean']:.5f} "
          f"CI90={[round(x, 5) for x in gate['treatment_ci90']]} "
          f"-> passes={gate['passes_treatment']} (margin={args.margin})", flush=True)
    print(f"[expC gate1] CONTROL   gap mean={gate['control_gap_mean']:.5f} "
          f"CI90={[round(x, 5) for x in gate['control_ci90']]} "
          f"-> passes={gate['passes_control']} (tol={args.tol})", flush=True)
    print(f"[expC gate1] GATE1 {'PASS' if gate['passes_gate1'] else 'not-passed'} "
          f"determinism={bit_repro}", flush=True)

    # DESCRIPTIVE horizon sweep: how the treatment gap grows as velocity accumulates.
    sweep = {}
    for h in args.horizon_sweep:
        lay = Layout(reach_range=args.treat_reach, horizon=h, name=f"treat_h{h}")
        res = [value_of_world_identity(lay, seed=s, **l3_kw) for s in args.seeds]
        gaps = np.array([r["gap"] for r in res])
        argmax_div = int(sum(r["best_auth_thrust"] != r["best_surr_thrust"] for r in res))
        sweep[f"h={h}"] = {"gap_mean": float(gaps.mean()), "gap_min": float(gaps.min()),
                           "gap_max": float(gaps.max()),
                           "argmax_divergence_seeds": argmax_div, "n_seeds": len(args.seeds)}
        print(f"[expC gate1]   sweep h={h}: gap mean={gaps.mean():.5f} "
              f"min={gaps.min():.5f} max={gaps.max():.5f} "
              f"argmax-divergent={argmax_div}/{len(args.seeds)}", flush=True)

    out = {
        "world": "P(k_land=1.5, k_water=1.5, gravity=0.4)",
        "surrogate": f"frozen L3 G_motion (hidden={args.hidden}, seed={args.seed})",
        "config": vars(args),
        "primary_treatment": {"reach": args.treat_reach, "horizon": args.treat_horizon},
        "control": {"reach": args.ctrl_reach, "horizon": args.ctrl_horizon},
        "gate": gate,
        "determinism_bit_reproducible": bool(bit_repro),
        "descriptive_horizon_sweep": sweep,
        "wall_seconds": round(time.time() - t0, 1),
    }
    d = os.path.dirname(args.json)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[expC gate1] DONE in {out['wall_seconds']}s -> {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
