"""L3 cross-recipe transfer probe, READOUT-ONLY (spec 2026-07-15).

Reuses the saved fullruns/l3_h8_heldout agents. Order of operations, all
pre-registered:

1. INTEGRITY GATE: for every saved agent, regenerate the standard pools with
   the ORIGINAL seed bases (800_000 auth / 850_000 surr via pooled_readout)
   and require bit-identical Ha/Hs against the saved state dumps. The
   drift-0.45 survival per-seed targets must average to the published 0.752
   (3 dp). Any mismatch aborts the run: investigate, never paper over.
2. TRANSFER: per drift-0.45 cell per arm per gate-passing family, score the
   frozen probe (fit on the regenerated standard pools) on a fresh authentic
   pool vs the family pool. Frozen seed bases: rff 880_000/890_000,
   cd 940_000/950_000 (distinct from every original base).

No training anywhere. run_expB2.py is not imported.

Usage:
    python scripts/run_l3_crossrecipe.py \\
        --agents-dir fullruns/l3_h8_heldout/agents \\
        --states-dir fullruns/l3_h8_heldout/states \\
        --families rff cd --device cuda \\
        --rff-json fullruns/l3_crossrecipe/gate0_rff.json \\
        --cd-json fullruns/l3_crossrecipe/gate0_cd.json \\
        --out-dir fullruns/l3_crossrecipe
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import re

import numpy as np

import itasorl.experiment_b2 as b2
from itasorl.experiment_b2 import (default_device, load_agent_bundle,
                                   pooled_readout, setup_l3_surrogate,
                                   transfer_readout)
from itasorl.surrogate_l3_families import fit_g_rff, make_g_cd
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)   # frozen organism world
PUBLISHED_TARGET = 0.752                                 # drift-0.45 survival mean
SEED_BASES = {"rff": (880_000, 890_000), "cd": (940_000, 950_000)}
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def parse_agent_filename(name: str) -> tuple[float, int, str]:
    m = AGENT_RE.search(name)
    if not m:
        raise ValueError(f"unrecognized agent filename: {name}")
    return float(m.group(1)), int(m.group(2)), m.group(3)


def rename_transfer_keys(out: dict, family: str) -> dict:
    return {k.replace("transfer_", f"transfer_{family}_", 1): v for k, v in out.items()}


def selected_knob(path: str, family: str):
    with open(path) as f:
        blob = json.load(f)
    sel = blob.get("selected")
    if sel is None:
        return None
    return sel["D"] if family == "rff" else sel["eps"]


def pools_match(Ha_saved, Hs_saved, Ha_new, Hs_new) -> bool:
    return bool(np.array_equal(Ha_saved, Ha_new) and np.array_equal(Hs_saved, Hs_new))


def build_family(family: str, knob):
    if family == "rff":
        return fit_g_rff(D=int(knob), params=P)
    return make_g_cd(eps=float(knob), params=P)


def decision_rule_pass(survival_mean: float, untrained_mean: float) -> bool:
    """The frozen primary decision rule (spec 2026-07-15): survival >= 0.65
    AND survival > untrained floor + 0.05, evaluated on per-arm means."""
    return bool(survival_mean >= 0.65 and survival_mean > untrained_mean + 0.05)


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", nargs="+", choices=("rff", "cd"), default=["rff", "cd"])
    ap.add_argument("--rff-json", default=None, help="gate-0 calibration JSON for rff")
    ap.add_argument("--cd-json", default=None, help="gate-0 calibration JSON for cd")
    ap.add_argument("--rff-d", type=int, default=None, help="override knob (quick/smoke only)")
    ap.add_argument("--cd-eps", type=float, default=None, help="override knob (quick/smoke only)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--quick", action="store_true",
                    help="smoke mode: seed 0 only, tiny pools, no bit compare")
    return ap.parse_args()


def main():
    a = cfg()
    if not a.quick and (a.rff_d is not None or a.cd_eps is not None):
        raise SystemExit("--rff-d/--cd-eps are smoke-only overrides; a full run must "
                         "take its knobs from gate-0 JSONs (--rff-json/--cd-json)")
    dev = default_device() if a.device == "auto" else a.device
    if a.device == "cuda" and dev != "cuda":
        raise SystemExit("--device cuda requested but CUDA unavailable")
    os.makedirs(a.out_dir, exist_ok=True)
    b2.DRIFT_MODE = "l3"
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)

    # The training surrogate must be the bit-identical hidden=8 GMotion so the
    # regenerated pools can match the saved dumps.
    setup_l3_surrogate(hidden=8, seed=0, params=P, device=dev)

    knobs = {}
    for fam in a.families:
        override = a.rff_d if fam == "rff" else a.cd_eps
        jpath = a.rff_json if fam == "rff" else a.cd_json
        if override is not None:
            knobs[fam] = override
        elif jpath is not None:
            k = selected_knob(jpath, fam)
            if k is None:
                print(f"family {fam}: DROPPED at gate 0 (selected=None); recorded, skipped")
                continue
            knobs[fam] = k
        else:
            raise SystemExit(f"family {fam}: need --{fam}-json or a knob override")
    families = {fam: build_family(fam, k) for fam, k in knobs.items()}
    print(f"families={knobs}  device={dev}  n_eps={n_eps} steps={steps} quick={a.quick}")

    cells = sorted(f for f in os.listdir(a.agents_dir) if AGENT_RE.search(f))
    if a.quick:
        cells = [c for c in cells if "_s0_" in c]

    # ---- phase 1: integrity gate over every reloaded agent -----------------
    survival_targets_045 = []
    train_pools = {}     # (drift, seed, arm) -> (Ha, Hs) for the transfer fits
    for name in cells:
        drift, seed, arm = parse_agent_filename(name)
        agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), dev)
        out, (Ha, Hs) = pooled_readout(agent, norm, P, drift, n_eps=n_eps,
                                       steps=steps, device=dev, seed=seed,
                                       return_pools=True)
        dump = os.path.join(a.states_dir, f"states_d{drift:.2f}_s{seed}_{arm}.npz")
        if not a.quick:
            saved = np.load(dump)
            if not pools_match(saved["Ha"], saved["Hs"], Ha, Hs):
                raise SystemExit(f"INTEGRITY GATE FAILED: regenerated pools differ "
                                 f"from {dump}. Do not proceed; investigate "
                                 f"(device mismatch? norm state? G retrain?).")
        if drift > 0 and arm == "survival":
            survival_targets_045.append(float(out["target"]))
        train_pools[(drift, seed, arm)] = (Ha, Hs)
        print(f"  integrity ok: {name}  target={out['target']:.3f}")
    if not a.quick:
        mean_t = round(float(np.mean(survival_targets_045)), 3)
        if mean_t != PUBLISHED_TARGET:
            raise SystemExit(f"INTEGRITY GATE FAILED: drift-0.45 survival mean "
                             f"{mean_t} != published {PUBLISHED_TARGET}")
        print(f"integrity gate PASSED: survival mean {mean_t} == {PUBLISHED_TARGET} "
              f"(determinism check #4)")

    # ---- phase 2: cross-recipe transfer ------------------------------------
    results = []
    for (drift, seed, arm), (Ha, Hs) in sorted(train_pools.items()):
        if drift == 0.0:
            continue                       # transfer is defined at drift 0.45
        agent, norm = load_agent_bundle(
            os.path.join(a.agents_dir, f"agent_d{drift:.2f}_s{seed}_{arm}.pt"), dev)
        row = {"drift": drift, "seed": seed, "arm": arm}
        for fam, g in families.items():
            sb_auth, sb_surr = SEED_BASES[fam]
            dump = os.path.join(a.out_dir,
                                f"states_d{drift:.2f}_s{seed}_{arm}_{fam}transfer.npz")
            out = transfer_readout(agent, norm, P, drift, Ha, Hs, n_eps=n_eps,
                                   steps=steps, device=dev, seed=seed,
                                   dump_path=dump, heldout=g,
                                   seed_base_auth=sb_auth, seed_base_surr=sb_surr)
            row.update(rename_transfer_keys(out, fam))
        results.append(row)
        print(f"  transfer d{drift:.2f} s{seed} {arm}: " + "  ".join(
            f"{fam}={row.get(f'transfer_{fam}_target', float('nan')):.3f}"
            for fam in families))

    # ---- aggregate ----------------------------------------------------------
    agg = {"knobs": knobs, "quick": a.quick, "n_eps": n_eps, "steps": steps,
           "published_target_check": None if a.quick else PUBLISHED_TARGET}
    for fam in families:
        for arm in ("untrained", "predictor", "survival"):
            vals = [r.get(f"transfer_{fam}_target") for r in results if r["arm"] == arm
                    and np.isfinite(r.get(f"transfer_{fam}_target", float("nan")))]
            if vals:
                v = np.asarray(vals, float)
                agg[f"{fam}_{arm}_per_seed"] = [round(float(x), 4) for x in v]
                agg[f"{fam}_{arm}_mean"] = round(float(v.mean()), 4)
                agg[f"{fam}_{arm}_n_ge_065"] = int((v >= 0.65).sum())
        sm, um = agg.get(f"{fam}_survival_mean"), agg.get(f"{fam}_untrained_mean")
        if sm is not None and um is not None:
            agg[f"{fam}_rule_pass"] = decision_rule_pass(sm, um)
            agg[f"{fam}_rule_margin"] = round(sm - max(0.65, um + 0.05), 4)
    with open(os.path.join(a.out_dir, "cells.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(a.out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print("wrote", os.path.join(a.out_dir, "aggregate.json"))


if __name__ == "__main__":
    main()
