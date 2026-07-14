"""
ITASORL - Experiment A oracle calibration for the L3 rung (gate 0), on the
ACTUAL organism world.

Why this exists: the hidden=4 organism run (fullruns/l3_h4_traces, 2026-07-13)
failed its gates (untrained floor 0.891, reward leak clean in 0/10 seeds,
engagement 30%). Root cause: hidden=4's in-band status (oracle 0.940) was frozen
from the pre-bugfix calibration on the WRONG world at sigma=0.05; after the
world-params fix, gate 0 was re-frozen only at hidden=8 and the floor sweep
covered capacities 8-32 (docs/PREREGISTRATION_L3.md sec.12). This script closes
that hole: it validates BOTH halves of gate 0 per capacity, on world P, before
any organism run is launched.

Per capacity it reports:
  oracle side   - residual-oracle AUROC (must land in [0.85, 0.95]) and the
                  mechanical leakage battery (length/metadata ~0.5), via the
                  same generate_l3_pairs/run_experiment_a_l3 as the original
                  calibration, at the frozen sigma_meas=0.02.
  organism side - the untrained (mechanical-floor) pooled target at drift 0.45,
                  built EXACTLY as run_expB2.py builds its untrained arm
                  (hidden=96, embed=64, world_model=True, ray_steps=5,
                  pool n_eps=110, steps=24). Must sit within 0.1 of 0.5.

Selection rule (frozen in advance, prereg sec.9 fallback (a) - bisection
between the bracketing capacities 4 and 8): the SECOND capacity is the smallest
h in {4..7} with oracle in-band AND mechanical leakage clean AND floor within
tolerance. hidden=8 is excluded (it is the headline capacity; sec.11 wants a
second artifact type) and instead serves as a REGRESSION check: it must
reproduce the frozen gate-0 values (oracle ~0.928 in-band, floor ~0.48).

Usage:  python scripts/run_expA_l3.py --json fullruns/l3_gate0_recal/calibration.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os

import numpy as np

import itasorl.experiment_b2 as b2
from itasorl.experiment_a_l3 import generate_l3_pairs, run_experiment_a_l3
from itasorl.experiment_b2 import default_device, pooled_readout, untrained_agent
from itasorl.surrogate_l3 import train_g_motion
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)  # the run_expB2.py organism world
SIGMA_MEAS = 0.02   # frozen gate-0 sensor-noise floor (PREREGISTRATION_L3.md sec.12)
BAND = (0.85, 0.95)  # oracle-detectability band (sec.7 gate 0)
FLOOR_TOL = 0.10    # untrained pooled target must be within this of 0.5
DRIFT = 0.45        # organism surrogate strength (run_expB2 default grid)
H8_ORACLE_REF = 0.928   # frozen gate-0 regression values at hidden=8 (sec.12)
H8_FLOOR_REF = 0.483


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hiddens", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--floor-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-pairs", type=int, default=60)
    ap.add_argument("--branch", type=int, default=30)
    ap.add_argument("--json", type=str, default=None,
                    help="write the calibration table + selection to this path")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return ap.parse_args()


def main():
    a = cfg()
    dev = default_device() if a.device == "auto" else a.device
    b2.DRIFT_MODE = "l3"
    print(f"L3 gate-0 recalibration on world P (the run_expB2 organism world)  device={dev}")
    print(f"  oracle: sigma_meas={SIGMA_MEAS} band=[{BAND[0]:.2f},{BAND[1]:.2f}] "
          f"n_pairs={a.n_pairs} branch={a.branch}")
    print(f"  floor : untrained pooled target @ drift={DRIFT}, seeds={a.floor_seeds}, "
          f"pass if |target-0.5| < {FLOOR_TOL}")
    rows = []
    for h in a.hiddens:
        # Same training seed/world as run_expB2's setup_l3_surrogate call, so the
        # organism run at the selected capacity uses a bit-identical G_motion.
        g = train_g_motion(hidden=h, device=dev, seed=0, params=P)
        eps = generate_l3_pairs(g, n_pairs=a.n_pairs, branch=a.branch, seed0=3000, params=P)
        oa = run_experiment_a_l3(eps, sigma_meas=SIGMA_MEAS, seed=0)
        oracle = float(oa["oracle_auroc"])
        in_band = BAND[0] <= oracle <= BAND[1]
        b2._L3_GMOTION = g  # install the SAME net for the organism-side floor check
        floors, leaks = [], []
        for s in a.floor_seeds:
            agent, norm = untrained_agent(P, DRIFT, ray_steps=5, hidden=96, embed=64,
                                          world_model=True, device=dev, seed=s)
            pool = pooled_readout(agent, norm, P, DRIFT, n_eps=110, steps=24,
                                  ray_steps=5, device=dev, seed=s)
            floors.append(float(pool["target"]))
            leaks.append(float(pool["pool_reward_leak"]))
        floor = float(np.mean(floors))
        floor_ok = abs(floor - 0.5) < FLOOR_TOL
        row = {"hidden": h, "oracle_auroc": oracle, "in_band": in_band,
               "mech_leak_pass": bool(oa["leakage_pass"]),
               "oracle_reward_leak": float(oa["reward_leak"]),
               "floor": floor, "floor_per_seed": floors, "floor_ok": floor_ok,
               "pool_reward_leak_per_seed": leaks,
               "passes_gate0": bool(in_band and oa["leakage_pass"] and floor_ok)}
        rows.append(row)
        print(f"  hidden={h}: oracle={oracle:.3f} in_band={in_band} "
              f"mech_leak_pass={row['mech_leak_pass']} "
              f"(oracle reward_leak={row['oracle_reward_leak']:.3f}, legit for this rung)  "
              f"floor={floor:.3f} per-seed={[f'{x:.3f}' for x in floors]} floor_ok={floor_ok} "
              f"pool_reward_leak={[f'{x:.3f}' for x in leaks]} "
              f"-> gate0 {'PASS' if row['passes_gate0'] else 'FAIL'}", flush=True)

    # Regression check: hidden=8 must reproduce the frozen gate-0 values.
    h8 = next((r for r in rows if r["hidden"] == 8), None)
    regression_ok = None
    if h8 is not None:
        regression_ok = (abs(h8["oracle_auroc"] - H8_ORACLE_REF) < 0.05
                         and abs(h8["floor"] - H8_FLOOR_REF) < FLOOR_TOL)
        print(f"\nregression check @ hidden=8: oracle {h8['oracle_auroc']:.3f} "
              f"(ref {H8_ORACLE_REF}), floor {h8['floor']:.3f} (ref {H8_FLOOR_REF}) "
              f"-> {'OK' if regression_ok else 'FAILED'}")

    candidates = [r for r in rows if r["hidden"] != 8 and r["passes_gate0"]]
    selected = min((r["hidden"] for r in candidates), default=None)
    if regression_ok is False:
        print("REGRESSION FAILED: the hidden=8 row does not reproduce the frozen "
              "gate-0 values; do NOT trust this table or launch an organism run.")
        selected = None
    if selected is not None:
        print(f"\nSELECTED second capacity: hidden={selected} (smallest gate0-passing "
              "candidate below the headline capacity)")
    else:
        print("\nNO second capacity passes gate 0 in the scanned range; do not launch "
              "an organism run (see prereg sec.9 fallback (b): data-budget knob).")

    if a.json:
        d = os.path.dirname(a.json)
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {"world": "P(k_land=1.5, k_water=1.5, gravity=0.4)",
                   "sigma_meas": SIGMA_MEAS, "band": list(BAND),
                   "floor_tol": FLOOR_TOL, "drift": DRIFT,
                   "floor_seeds": a.floor_seeds, "rows": rows,
                   "regression_ok": regression_ok, "selected_hidden": selected}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved {a.json}")
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
