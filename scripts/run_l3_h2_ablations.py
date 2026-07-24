"""L3 H2 substrate-grounding ablations, READOUT-ONLY (spec 2026-07-22).

Two channels against the saved fullruns/l3_h8_heldout agents. No training.

1. INTEGRITY GATE: regenerate standard pools with the original seed bases
   (800_000 / 850_000) against the bit-identical rebuilt hidden=8 GMotion;
   require bit-identical Ha/Hs vs saved dumps and drift-0.45 survival mean
   0.752 (3 dp). Abort on mismatch.
2. CHANNEL gn (PRIMARY): score the frozen world-identity direction on a
   gate-calibrated unstructured Gaussian-jitter surrogate (G_gn). Seed bases
   960_000 / 970_000.
3. CHANNEL ladder (SECONDARY): same frozen direction on same-recipe GMotion
   at hidden {16, 32, 64}. Seed bases per capacity (spec). Capacities that
   fail mech_leak_pass or floor_ok in the ladder gate-0 JSON are dropped.

Usage:
    python scripts/run_l3_h2_ablations.py \\
        --agents-dir fullruns/l3_h8_heldout/agents \\
        --states-dir fullruns/l3_h8_heldout/states \\
        --channels gn ladder --device cuda \\
        --gn-json fullruns/l3_h2_ablations/gate0_gn.json \\
        --ladder-json fullruns/l3_h2_ablations/gate0_ladder.json \\
        --out-dir fullruns/l3_h2_ablations
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
from itasorl.surrogate_l3 import train_g_motion
from itasorl.surrogate_l3_families import make_g_gn
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)   # frozen organism world
PUBLISHED_TARGET = 0.752                                 # drift-0.45 survival mean
LADDER_HIDDENS = (16, 32, 64)
SEED_BASES = {
    "gn": (960_000, 970_000),
    "h16": (980_000, 990_000),
    "h32": (1_000_000, 1_010_000),
    "h64": (1_020_000, 1_030_000),
}
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def parse_agent_filename(name: str) -> tuple[float, int, str]:
    m = AGENT_RE.search(name)
    if not m:
        raise ValueError(f"unrecognized agent filename: {name}")
    return float(m.group(1)), int(m.group(2)), m.group(3)


def rename_transfer_keys(out: dict, tag: str) -> dict:
    return {k.replace("transfer_", f"transfer_{tag}_", 1): v for k, v in out.items()}


def pools_match(Ha_saved, Hs_saved, Ha_new, Hs_new) -> bool:
    return bool(np.array_equal(Ha_saved, Ha_new) and np.array_equal(Hs_saved, Hs_new))


def selected_gn_sigma(path: str):
    with open(path) as f:
        blob = json.load(f)
    sel = blob.get("selected")
    if sel is None:
        return None
    return sel["sigma_v"]


def ladder_capacities(path: str) -> list[tuple[int, float]]:
    """Return (hidden, oracle_auroc) for ladder rows that pass mech_leak + floor.
    Composite passes_gate0 (which requires the in-band check) is ignored."""
    with open(path) as f:
        blob = json.load(f)
    out = []
    for row in blob.get("rows", []):
        h = row.get("hidden")
        if h not in LADDER_HIDDENS:
            continue
        if not (row.get("mech_leak_pass") and row.get("floor_ok")):
            continue
        out.append((int(h), float(row["oracle_auroc"])))
    return out


def decision_rule_pass(survival_mean: float, untrained_mean: float) -> bool:
    """Frozen positive rule: survival >= 0.65 AND > untrained + 0.05."""
    return bool(survival_mean >= 0.65 and survival_mean > untrained_mean + 0.05)


def gn_verdict(survival_mean: float, untrained_mean: float) -> str:
    """Frozen Channel-1 adjudication table (spec 2026-07-22)."""
    clears_bar = survival_mean >= 0.65
    clears_floor = survival_mean > untrained_mean + 0.05
    if clears_bar and (not clears_floor) and untrained_mean >= 0.60:
        return "UNINFORMATIVE"
    if clears_bar and clears_floor:
        return "H2_NEGATIVE"
    if (not clears_bar) and clears_floor:
        return "PARTIAL"
    return "H2_SUPPORTED"


def ladder_promotion_pass(h16_mean: float, h64_mean: float) -> bool:
    """Frozen promotion rule: only evaluated when gn drops or is UNINFORMATIVE."""
    return bool(h64_mean < 0.65 and h16_mean > h64_mean + 0.05)


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--channels", nargs="+", choices=("gn", "ladder"),
                    default=["gn", "ladder"])
    ap.add_argument("--gn-json", default=None, help="gate-0 calibration JSON for gn")
    ap.add_argument("--ladder-json", default=None,
                    help="gate-0 calibration JSON for mlp hidden 16/32/64")
    ap.add_argument("--gn-sigma", type=float, default=None,
                    help="override sigma_v (quick/smoke only)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--quick", action="store_true",
                    help="smoke mode: seed 0 only, tiny pools, no bit compare")
    return ap.parse_args()


def main():
    a = cfg()
    if not a.quick and a.gn_sigma is not None:
        raise SystemExit("--gn-sigma is a smoke-only override; a full run must "
                         "take its knob from --gn-json")
    dev = default_device() if a.device == "auto" else a.device
    if a.device == "cuda" and dev != "cuda":
        raise SystemExit("--device cuda requested but CUDA unavailable")
    os.makedirs(a.out_dir, exist_ok=True)
    b2.DRIFT_MODE = "l3"
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)

    # Training surrogate must be the bit-identical hidden=8 GMotion so the
    # regenerated pools can match the saved dumps.
    setup_l3_surrogate(hidden=8, seed=0, params=P, device=dev)

    heldouts = {}          # tag -> callable g
    ladder_oracles = {}    # "h16" -> oracle_auroc
    gn_dropped = False
    if "gn" in a.channels:
        sigma = None
        if a.gn_sigma is not None:
            sigma = a.gn_sigma
        elif a.gn_json is not None:
            sigma = selected_gn_sigma(a.gn_json)
            if sigma is None:
                print("channel gn: DROPPED at gate 0 (selected=None); recorded")
                gn_dropped = True
        else:
            raise SystemExit("channel gn: need --gn-json or --gn-sigma")
        if sigma is not None:
            heldouts["gn"] = make_g_gn(sigma_v=float(sigma), params=P, seed=0)
            print(f"  gn sigma_v={sigma}")
    if "ladder" in a.channels:
        if a.ladder_json is None and not a.quick:
            raise SystemExit("channel ladder: need --ladder-json")
        if a.ladder_json is not None:
            caps = ladder_capacities(a.ladder_json)
        else:
            # quick smoke: train tiny-budget Gs without a gate-0 JSON
            caps = [(h, float("nan")) for h in LADDER_HIDDENS]
        dropped = []
        for h, oracle in caps:
            tag = f"h{h}"
            heldouts[tag] = train_g_motion(hidden=h, device=dev, seed=0, params=P)
            ladder_oracles[tag] = oracle
            print(f"  ladder {tag}: oracle={oracle}")
        if a.ladder_json is not None:
            kept = {h for h, _ in caps}
            for h in LADDER_HIDDENS:
                if h not in kept:
                    dropped.append(h)
            if dropped:
                print(f"  ladder DROPPED capacities (floor/leak fail): {dropped}")

    print(f"channels={list(heldouts)}  device={dev}  n_eps={n_eps} steps={steps} "
          f"quick={a.quick}")

    cells = sorted(f for f in os.listdir(a.agents_dir) if AGENT_RE.search(f))
    if a.quick:
        cells = [c for c in cells if "_s0_" in c]

    # ---- phase 1: integrity gate -------------------------------------------
    survival_targets_045 = []
    train_pools = {}
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
              f"(determinism check #5)")

    # ---- phase 2: ablation transfer ----------------------------------------
    results = []
    for (drift, seed, arm), (Ha, Hs) in sorted(train_pools.items()):
        if drift == 0.0:
            continue
        agent, norm = load_agent_bundle(
            os.path.join(a.agents_dir, f"agent_d{drift:.2f}_s{seed}_{arm}.pt"), dev)
        row = {"drift": drift, "seed": seed, "arm": arm}
        for tag, g in heldouts.items():
            sb_auth, sb_surr = SEED_BASES[tag]
            if hasattr(g, "reseed"):
                g.reseed(sb_surr + seed)
            dump = os.path.join(
                a.out_dir, f"states_d{drift:.2f}_s{seed}_{arm}_{tag}transfer.npz")
            out = transfer_readout(agent, norm, P, drift, Ha, Hs, n_eps=n_eps,
                                   steps=steps, device=dev, seed=seed,
                                   dump_path=dump, heldout=g,
                                   seed_base_auth=sb_auth, seed_base_surr=sb_surr)
            row.update(rename_transfer_keys(out, tag))
        results.append(row)
        print(f"  transfer d{drift:.2f} s{seed} {arm}: " + "  ".join(
            f"{tag}={row.get(f'transfer_{tag}_target', float('nan')):.3f}"
            for tag in heldouts))

    # ---- aggregate ----------------------------------------------------------
    agg = {"quick": a.quick, "n_eps": n_eps, "steps": steps,
           "published_target_check": None if a.quick else PUBLISHED_TARGET,
           "gn_dropped_at_gate0": gn_dropped,
           "ladder_oracles": ladder_oracles}
    if "gn" in heldouts:
        agg["gn_sigma_v"] = float(getattr(heldouts["gn"], "_sigma_v"))
    for tag in heldouts:
        for arm in ("untrained", "predictor", "survival"):
            vals = [r.get(f"transfer_{tag}_target") for r in results if r["arm"] == arm
                    and np.isfinite(r.get(f"transfer_{tag}_target", float("nan")))]
            if vals:
                v = np.asarray(vals, float)
                agg[f"{tag}_{arm}_per_seed"] = [round(float(x), 4) for x in v]
                agg[f"{tag}_{arm}_mean"] = round(float(v.mean()), 4)
                agg[f"{tag}_{arm}_n_ge_065"] = int((v >= 0.65).sum())
    if "gn" in heldouts:
        sm, um = agg.get("gn_survival_mean"), agg.get("gn_untrained_mean")
        if sm is not None and um is not None:
            agg["gn_rule_pass"] = decision_rule_pass(sm, um)
            agg["gn_rule_margin"] = round(sm - max(0.65, um + 0.05), 4)
            agg["gn_verdict"] = gn_verdict(sm, um)

    # Promotion rule: only if gn dropped or landed UNINFORMATIVE.
    promote = gn_dropped or agg.get("gn_verdict") == "UNINFORMATIVE"
    agg["ladder_promotion_eligible"] = promote
    if promote and "h16" in heldouts and "h64" in heldouts:
        h16 = agg.get("h16_survival_mean")
        h64 = agg.get("h64_survival_mean")
        if h16 is not None and h64 is not None:
            agg["ladder_promotion_pass"] = ladder_promotion_pass(h16, h64)
            agg["ladder_promotion_margin"] = round(h16 - (h64 + 0.05), 4)

    with open(os.path.join(a.out_dir, "cells.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(a.out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print("wrote", os.path.join(a.out_dir, "aggregate.json"))
    if "gn_verdict" in agg:
        print(f"gn_verdict={agg['gn_verdict']}  gn_rule_pass={agg.get('gn_rule_pass')}")
    if "ladder_promotion_pass" in agg:
        print(f"ladder_promotion_pass={agg['ladder_promotion_pass']}")


if __name__ == "__main__":
    main()
