"""
ITASORL - offline re-analysis of dumped B-v2 recurrent states (NO GPU, NO training).

`run_expB2.py --dump-states DIR` persists the raw recurrent states of every
(drift, seed, agent) cell as `states_d<drift>_s<seed>_<agent>.npz`. This script
reloads them and recomputes the pooled world-identity probe under three feature
sets, so we can ask whether the world-identity signal lives in a VOLATILITY
signature the pre-registered LEVEL probe discards - without re-running a GPU sweep.

    target       LEVEL features       [mean h, final h]   (the pre-registered headline)
    target_var   DISPERSION features  [std h, mean|delta h|]
    target_full  LEVEL ++ DISPERSION

Reports per-probe selectivity = target - shuffled (same label permutation on the
same feature set), which cancels the constant probe-bias offset that pushes some
seeds above 0.5 at drift=0.

Usage:  python scripts/reanalyze_expB2_states.py runs/pr1_quick
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

from itasorl.experiment_b import (episode_features, episode_features_full,
                                  episode_features_var, probe_auroc)

FNAME = re.compile(r"states_d(?P<drift>[-\d.]+)_s(?P<seed>\d+)_(?P<agent>\w+)\.npz$")


def _nanmean(rows: list[dict], k: str) -> float:
    """Mean over seeds ignoring NaN/missing, returning NaN for an all-empty slice
    (avoids numpy's 'Mean of empty slice' warning when e.g. ceiling_drag is undefined
    at drift 0)."""
    vals = [r[k] for r in rows if k in r and not np.isnan(r[k])]
    return float(np.mean(vals)) if vals else float("nan")


def probe_cell(npz: dict, seed: int) -> dict:
    """Recompute the three-feature-set pooled probe from one dumped cell."""
    Ha, Hs = npz["Ha"], npz["Hs"]
    if len(Ha) < 5 or len(Hs) < 5:
        return {}
    H = np.concatenate([Ha, Hs])
    y = np.concatenate([np.zeros(len(Ha)), np.ones(len(Hs))]).astype(int)
    feats = {"target": episode_features(H), "target_var": episode_features_var(H),
             "target_full": episode_features_full(H)}
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(y)                       # shared across feature sets
    out = {}
    for name, X in feats.items():
        auc = probe_auroc(X, y)
        out[name] = auc
        out["sel_" + name] = auc - probe_auroc(X, y_perm)
    drs = npz["drs"]
    if len(Hs) >= 10 and float(np.ptp(drs)) > 1e-6:
        out["ceiling_drag"] = probe_auroc(episode_features(Hs), (drs > np.median(drs)).astype(int))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("states_dir", help="directory of states_*.npz written by --dump-states")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.states_dir, "states_*.npz")))
    if not files:
        raise SystemExit(f"no states_*.npz found in {a.states_dir}")

    # group per-cell metrics by (drift, agent) so we can average over seeds
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    print(f"Re-analyzing {len(files)} dumped cells from {a.states_dir}\n")
    for f in files:
        m = FNAME.search(os.path.basename(f))
        if not m:
            print(f"  skip (unparsed name): {os.path.basename(f)}")
            continue
        drift, seed, agent = m["drift"], int(m["seed"]), m["agent"]
        with np.load(f) as npz:
            cell = dict(npz)
        if "Ha" not in cell:  # heldout-eval sibling dumps (_h7transfer/_cg) lack pool keys
            print(f"  skip (heldout sibling dump): {os.path.basename(f)}")
            continue
        res = probe_cell(cell, seed)
        if not res:
            print(f"  skip (too few survivors): {os.path.basename(f)}")
            continue
        agg[(drift, agent)].append(res)
        print(f"  d={drift} s={seed} {agent:10s} "
              f"target={res['target']:.3f} var={res['target_var']:.3f} "
              f"full={res['target_full']:.3f}  "
              f"sel(L={res['sel_target']:+.3f} V={res['sel_target_var']:+.3f} "
              f"F={res['sel_target_full']:+.3f})  "
              f"ceiling_drag={res.get('ceiling_drag', float('nan')):.3f}")

    print("\n============  MEAN over seeds (per drift x agent)  ============")
    for (drift, agent) in sorted(agg):
        rows = agg[(drift, agent)]
        print(f"  d={drift} {agent:10s} (n={len(rows)})  "
              f"target={_nanmean(rows, 'target'):.3f} var={_nanmean(rows, 'target_var'):.3f} "
              f"full={_nanmean(rows, 'target_full'):.3f}  "
              f"sel(L={_nanmean(rows, 'sel_target'):+.3f} V={_nanmean(rows, 'sel_target_var'):+.3f} "
              f"F={_nanmean(rows, 'sel_target_full'):+.3f})  "
              f"ceiling_drag={_nanmean(rows, 'ceiling_drag'):.3f}")

    # headline: does any survival readout at the strongest drift cross the 0.65 bar?
    drifts = sorted({d for (d, _) in agg}, key=float)
    if drifts:
        dmax = drifts[-1]
        rows = agg.get((dmax, "survival"), [])
        if rows:
            tv = float(np.nanmean([r["target_var"] for r in rows]))
            tf = float(np.nanmean([r["target_full"] for r in rows]))
            tl = float(np.nanmean([r["target"] for r in rows]))
            hit = max(tv, tf) >= 0.65
            print(f"\nSurvival @ drift {dmax}: level={tl:.3f} var={tv:.3f} full={tf:.3f} (bar 0.65)")
            print("  -> " + ("VOLATILITY-ENCODED: level probe was mis-specified"
                             if hit else "no volatility encoding either; strengthens the null"))


if __name__ == "__main__":
    main()
