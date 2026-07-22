"""H2 substrate-grounding ablation - A1 graded-seam sweep, READOUT-ONLY.

Spec: docs/specs/2026-07-21-h2-substrate-grounding-ablation-design.md (A1 only).

Reuses the saved fullruns/l3_h8_heldout agents; NO training. For each alpha in the
frozen grid, the surrogate world's velocity law is the convex blend
`(1-alpha)*authentic + alpha*G_hidden8` (itasorl.surrogate_l3.GradedGMotion), injected
via the b2._L3_GMOTION hook; the standard pooled_readout then scores world identity.

Order of operations (pre-registered):
  0. INTEGRITY GATE (alpha=1): the graded law == the plain hidden=8 GMotion, so the
     regenerated pools must bit-match the saved dumps and the drift-0.45 survival mean
     must equal 0.752 (3 dp). Any mismatch aborts - investigate, never paper over.
  1. L0 ANCHOR (alpha=0): the graded law == the authentic law, so the surrogate pool is
     authentic-vs-authentic; the survival mean must be equivalent to the chance floor
     (ROPE [0.45, 0.55]).
  2. SWEEP: per alpha, per drift-0.45 agent, pooled_readout -> target. Aggregate per-arm
     per-alpha mean + t-CI90; report the survival collapse curve and its monotonicity
     (Spearman rho of survival mean vs alpha; substrate-grounded => rho ~ +1, i.e. the
     signal rises with the seam and collapses toward the floor as it is removed).

Usage:
    python scripts/run_expH2_ablation.py \\
        --agents-dir fullruns/l3_h8_heldout/agents \\
        --states-dir fullruns/l3_h8_heldout/states \\
        --out-dir fullruns/expH2_ablation --device cpu
    python scripts/run_expH2_ablation.py --agents-dir ... --states-dir ... \\
        --out-dir ... --quick        # smoke: seed 0, alphas {0,1}, tiny pools, no bit compare
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import re

import numpy as np

import itasorl.experiment_b2 as b2
from itasorl.experiment_b2 import default_device, load_agent_bundle, pooled_readout, setup_l3_surrogate
from itasorl.stats import rope_test, t_ci90
from itasorl.surrogate_l3 import GradedGMotion
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)   # frozen organism world
PUBLISHED_TARGET = 0.752                                 # drift-0.45 survival mean at alpha=1
DRIFT = 0.45                                             # the headline evaluation cell
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)               # FROZEN grid (spec section 9)
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def parse_agent_filename(name: str) -> tuple[float, int, str]:
    m = AGENT_RE.search(name)
    if not m:
        raise ValueError(f"unrecognized agent filename: {name}")
    return float(m.group(1)), int(m.group(2)), m.group(3)


def pools_match(Ha_saved, Hs_saved, Ha_new, Hs_new) -> bool:
    return bool(np.array_equal(Ha_saved, Ha_new) and np.array_equal(Hs_saved, Hs_new))


def spearman_rho(x, y) -> float:
    """Rank correlation without a scipy dependency (distinct values assumed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["survival", "untrained", "predictor"],
                    choices=("survival", "untrained", "predictor"))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--quick", action="store_true",
                    help="smoke: seed 0, alphas {0,1}, tiny pools, no bit compare")
    return ap.parse_args()


def main():
    a = cfg()
    dev = default_device() if a.device == "auto" else a.device
    if a.device == "cuda" and dev != "cuda":
        raise SystemExit("--device cuda requested but CUDA unavailable")
    os.makedirs(a.out_dir, exist_ok=True)
    b2.DRIFT_MODE = "l3"
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)
    alphas = (0.0, 1.0) if a.quick else ALPHAS

    # The base surrogate MUST be the bit-identical hidden=8 GMotion so alpha=1 reproduces
    # the saved dumps. setup installs it into b2._L3_GMOTION; capture it, then per alpha
    # overwrite the hook with the graded blend around this same base net.
    setup_l3_surrogate(hidden=8, seed=0, params=P, device=dev)
    g_base = b2._L3_GMOTION
    print(f"device={dev}  arms={a.arms}  alphas={alphas}  n_eps={n_eps} steps={steps} quick={a.quick}")

    cells = sorted(f for f in os.listdir(a.agents_dir) if AGENT_RE.search(f))
    cells = [c for c in cells if parse_agent_filename(c)[0] == DRIFT
             and parse_agent_filename(c)[2] in a.arms]
    if a.quick:
        cells = [c for c in cells if "_s0_" in c]
    if not cells:
        raise SystemExit(f"no drift-{DRIFT} agents for arms {a.arms} in {a.agents_dir}")

    rows = []
    integrity = {"checked": False, "pools_bit_match": None, "survival_mean_alpha1": None,
                 "published_target": PUBLISHED_TARGET}
    # Validate the determinism gate FIRST (alpha=1 == plain hidden=8 GMotion) so a bad
    # injection aborts in one pass, before the expensive sweep. Then the remaining alphas.
    alphas_ordered = ([1.0] + [x for x in alphas if x != 1.0]) if 1.0 in alphas else list(alphas)
    for alpha in alphas_ordered:
        b2._L3_GMOTION = GradedGMotion(g_base, alpha, P.dt)
        bit_match_all = True
        for name in cells:
            drift, seed, arm = parse_agent_filename(name)
            agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), dev)
            out, (Ha, Hs) = pooled_readout(agent, norm, P, drift, n_eps=n_eps, steps=steps,
                                           device=dev, seed=seed, return_pools=True)
            # INTEGRITY GATE at alpha=1: bit-match the saved dumps (determinism check #5).
            if not a.quick and alpha == 1.0:
                dump = os.path.join(a.states_dir, f"states_d{drift:.2f}_s{seed}_{arm}.npz")
                if os.path.exists(dump):
                    saved = np.load(dump)
                    if not pools_match(saved["Ha"], saved["Hs"], Ha, Hs):
                        raise SystemExit(
                            f"INTEGRITY GATE FAILED: alpha=1 pools differ from {dump}. "
                            f"Do not proceed; investigate (device? norm? G retrain?).")
                else:
                    bit_match_all = False   # no dump to compare (e.g. arm not dumped)
            rows.append({"alpha": alpha, "seed": seed, "arm": arm,
                         "target": float(out["target"]),
                         "target_lo": float(out["target_lo"]), "target_hi": float(out["target_hi"]),
                         "pool_leak_clean": bool(out["pool_leak_clean"]),
                         "deaths_auth": int(out["deaths_auth"]), "deaths_surr": int(out["deaths_surr"])})
        done = [r["target"] for r in rows if r["alpha"] == alpha and r["arm"] == "survival"]
        print(f"  alpha={alpha:.2f}: survival targets " +
              " ".join(f"{t:.3f}" for t in done) + (f"  mean={np.mean(done):.3f}" if done else ""))
        # Abort the sweep immediately if the alpha=1 determinism gate misses 0.752.
        if not a.quick and alpha == 1.0 and "survival" in a.arms:
            mean_a1 = round(float(np.mean(done)), 3) if done else None
            integrity.update({"checked": True, "pools_bit_match": bit_match_all,
                              "survival_mean_alpha1": mean_a1})
            if mean_a1 != PUBLISHED_TARGET:
                raise SystemExit(f"INTEGRITY GATE FAILED: alpha=1 survival mean {mean_a1} "
                                 f"!= published {PUBLISHED_TARGET}. Aborting before the sweep.")
            print(f"integrity gate PASSED: alpha=1 survival mean {mean_a1} == {PUBLISHED_TARGET} "
                  f"(determinism check #5)")

    # ---- aggregate ----------------------------------------------------------
    agg = {"drift": DRIFT, "alphas": list(alphas), "arms": a.arms, "quick": a.quick,
           "n_eps": n_eps, "steps": steps}
    per_alpha = {}     # arm -> [mean per alpha] for the curve/monotonicity
    for arm in a.arms:
        per_alpha[arm] = []
        for alpha in alphas:
            vals = [r["target"] for r in rows if r["alpha"] == alpha and r["arm"] == arm]
            v = np.asarray(vals, float)
            m = float(v.mean()) if v.size else float("nan")
            lo, hi = t_ci90(v)
            key = f"{arm}_a{alpha:.2f}"
            agg[f"{key}_per_seed"] = [round(float(x), 4) for x in v]
            agg[f"{key}_mean"] = round(m, 4)
            agg[f"{key}_tci90"] = [round(lo, 4), round(hi, 4)]
            agg[f"{key}_n_ge_065"] = int((v >= 0.65).sum())
            per_alpha[arm].append(m)

    if "survival" in a.arms:
        sm = per_alpha["survival"]
        agg["survival_curve"] = [round(x, 4) for x in sm]
        agg["survival_monotonicity_rho"] = round(spearman_rho(list(alphas), sm), 4)
        # L0 anchor at alpha=0: survival equivalent to the chance floor?
        a0 = [r["target"] for r in rows if r["alpha"] == 0.0 and r["arm"] == "survival"]
        rr = rope_test(a0, rope=(0.45, 0.55))
        agg["l0_anchor_alpha0"] = {"mean": round(rr.mean, 4), "hdi": [round(rr.hdi[0], 4), round(rr.hdi[1], 4)],
                                   "p_in_rope": round(rr.p_in_rope, 4), "accept_equiv": bool(rr.accept)}
    agg["integrity"] = integrity

    with open(os.path.join(a.out_dir, "cells.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(a.out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print("wrote", os.path.join(a.out_dir, "aggregate.json"))
    if "survival" in a.arms and not a.quick:
        print("survival collapse curve (alpha -> mean target):")
        for al, m in zip(alphas, per_alpha["survival"]):
            print(f"  alpha={al:.2f}  target={m:.3f}")
        print(f"monotonicity (Spearman rho, alpha vs survival mean) = {agg['survival_monotonicity_rho']}")
        print(f"L0 anchor (alpha=0 survival): {agg['l0_anchor_alpha0']}")


if __name__ == "__main__":
    main()
