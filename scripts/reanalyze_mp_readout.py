"""
ITASORL - re-score of the SECONDARY matched-pair (mp) channel with the FIXED
pair-grouped estimator, from regenerated rollouts of the saved agent bundles.

Why this exists (FINDINGS sec. 10.6 banner; PREREGISTRATION_L3 sec. 12,
2026-07-19 entry): the heldout runs' mp_target numbers were recorded on pre-fix
code (probe_auroc = singleton CV groups) under the ASSUMPTION that pair counts
60/25 co-locate matched-pair twins in GroupKFold, making the numbers fold-safe.
That assumption was flagged "confirm, do not assume" - and it is FALSE under the
sklearn in use (1.5.2): GroupKFold splits twins at both pair counts, and the
committed drift-0.00 mp floors carry the bias signature (~0.13 instead of ~0.5,
bit-identical L0 twins read toward AUROC 0).

Unlike the cg channel there are no saved mp state dumps, but the mp rollout is
deterministic from the saved agent bundles (fixed seed_base=700_000), verified
by reproducing a stored pre-fix cell value bit-for-bit. This script therefore:

  1. regenerates each cell's matched-pair rollout from `agents/agent_*.pt`,
  2. re-scores the OLD way (singleton groups) as a per-cell determinism gate -
     it must equal the stored `mp.target` exactly or the cell is flagged,
  3. re-scores with the FIXED pair-grouped probe (the corrected number).

Run ON THE MACHINE THAT HOLDS the gitignored fullruns/ archives:

    python scripts/reanalyze_mp_readout.py fullruns/l3_h8_heldout --l3-hidden 8 \
        --json fullruns/l3_h8_heldout/mp_rescore.json
    python scripts/reanalyze_mp_readout.py fullruns/l3_h7_heldout --l3-hidden 7 \
        --json fullruns/l3_h7_heldout/mp_rescore.json

The mp channel is secondary/demoted (FINDINGS 9, 11): no pre-registered decision
rides on it, so this re-score corrects the record rather than any verdict.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

import itasorl.experiment_b2 as b2
from itasorl.experiment_a import grouped_auroc
from itasorl.experiment_b2 import (
    _episode_feature,
    default_device,
    load_agent_bundle,
    matched_pair_recurrent_rollout,
    probe_world_identity,
)
from itasorl.world import WorldParams

FNAME = re.compile(r"agent_d(?P<drift>[-\d.]+)_s(?P<seed>\d+)_(?P<agent>\w+)\.pt$")

# The heldout runs' frozen mp config (run_expB2.py defaults, full/non-quick).
MP_PAIRS, MP_PREFIX, MP_BRANCH, RAY_STEPS = 60, 20, 24, 5
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


def stored_mp_target(bundle: str, drift: str, seed: int) -> dict:
    """Stored per-agent mp targets from the run's committed cell file, if present."""
    path = os.path.join(bundle, "cells", f"cell_d{drift}_s{seed}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        cell = json.load(fh)["cell"]
    return {g: cell["agents"][g]["mp"]["target"] for g in cell["agents"]}


def rescore_cell(path: str, drift: float, probe_seed: int, device: str) -> dict:
    agent, norm = load_agent_bundle(path, device)
    a_eps, s_eps = matched_pair_recurrent_rollout(
        agent, norm, P, drift, n_pairs=MP_PAIRS, prefix_steps=MP_PREFIX,
        branch_steps=MP_BRANCH, ray_steps=RAY_STEPS, device=device, seed_base=700_000)
    X = np.stack([_episode_feature(e["H"]) for e in a_eps + s_eps])
    y = np.array([e["label"] for e in a_eps + s_eps])
    old = grouped_auroc(X, y, np.arange(len(y)))          # pre-fix: singleton groups
    fixed = probe_world_identity(a_eps, s_eps, seed=probe_seed)["target"]
    return {"mp_old": float(old), "mp_fixed": float(fixed), "n_pairs": len(a_eps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="heldout fullruns bundle (holds agents/ and cells/)")
    ap.add_argument("--l3-hidden", type=int, required=True,
                    help="G_motion capacity the bundle trained against (8 forward, 7 reverse)")
    ap.add_argument("--json", default=None, help="optional path for the re-score JSON")
    a = ap.parse_args()

    device = default_device()
    b2.DRIFT_MODE = "l3"
    b2.setup_l3_surrogate(hidden=a.l3_hidden, device=device, seed=0, params=P)

    files = sorted(glob.glob(os.path.join(a.bundle, "agents", "agent_*.pt")))
    if not files:
        raise SystemExit(f"no agents/agent_*.pt found in {a.bundle}")

    cells, mismatches = [], []
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    print(f"Re-scoring {len(files)} mp cells from {a.bundle} "
          f"(l3 hidden={a.l3_hidden}, device={device})\n")
    for f in files:
        m = FNAME.search(os.path.basename(f))
        if not m:
            print(f"  skip (unparsed name): {os.path.basename(f)}")
            continue
        drift, seed, agent = m["drift"], int(m["seed"]), m["agent"]
        r = rescore_cell(f, float(drift), seed, device)
        stored = stored_mp_target(a.bundle, drift, seed).get(agent)
        # determinism gate: the singleton-group re-score must reproduce the stored
        # pre-fix value exactly, or the regeneration cannot be trusted for this cell
        repro = stored is not None and abs(r["mp_old"] - stored) < 1e-12
        if stored is not None and not repro:
            mismatches.append((drift, seed, agent))
        row = {"drift": drift, "seed": seed, "agent": agent,
               "mp_stored": stored, "reproduces_stored": repro, **r}
        cells.append(row)
        agg[(drift, agent)].append(row)
        print(f"  d={drift} s={seed} {agent:10s} stored={stored if stored is not None else float('nan'):.3f} "
              f"old={r['mp_old']:.3f} ({'OK' if repro else 'MISMATCH'})  "
              f"fixed={r['mp_fixed']:.3f}  delta={r['mp_fixed'] - r['mp_old']:+.3f}", flush=True)

    print("\n============  MEAN over seeds (per drift x agent)  ============")
    summary = {}
    for (drift, agent) in sorted(agg):
        rows = agg[(drift, agent)]
        old = np.array([r["mp_old"] for r in rows], float)
        fx = np.array([r["mp_fixed"] for r in rows], float)
        summary[f"d{drift}_{agent}"] = {
            "n_seeds": len(rows),
            "mp_old_mean": float(old.mean()),
            "mp_fixed_mean": float(fx.mean()),
            "mp_fixed_per_seed": [round(float(v), 4) for v in fx],
            "all_reproduce_stored": all(r["reproduces_stored"] for r in rows),
        }
        print(f"  d={drift} {agent:10s} (n={len(rows)})  old={old.mean():.3f}  "
              f"fixed={fx.mean():.3f}  delta={(fx - old).mean():+.3f}")

    print("\n============  DECISION READOUTS  ============")
    d0 = next((d for (d, _) in agg if float(d) == 0.0), None)
    if d0 is not None:
        print(f"L0 mp floors at drift {d0} (stored/pre-fix read ~0.13; fixed must sit near 0.5):")
        for agent in sorted({g for (dd, g) in agg if dd == d0}):
            s = summary[f"d{d0}_{agent}"]
            ok = 0.4 <= s["mp_fixed_mean"] <= 0.6
            print(f"  {agent:10s} old={s['mp_old_mean']:.3f} -> fixed={s['mp_fixed_mean']:.3f}"
                  f"  {'OK (chance band)' if ok else 'STILL OFF-CHANCE - investigate'}")
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} cells did NOT reproduce their stored pre-fix "
              f"value - their corrections are untrusted: {mismatches}")
    else:
        print("\nDeterminism gate: every cell reproduced its stored pre-fix mp_target "
              "exactly (singleton-group re-score == committed value).")

    if a.json:
        d = os.path.dirname(a.json)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"bundle": a.bundle, "l3_hidden": a.l3_hidden,
                       "estimator": "probe_world_identity (pair-grouped, post-1633bca)",
                       "mp_config": {"n_pairs": MP_PAIRS, "prefix_steps": MP_PREFIX,
                                     "branch_steps": MP_BRANCH, "ray_steps": RAY_STEPS},
                       "cells": cells, "aggregate": summary}, fh, indent=1, default=float)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
