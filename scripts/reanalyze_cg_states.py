"""
ITASORL - offline re-score of dumped common-garden tails with the FIXED cg_probe
(no GPU, no training).

Why this exists (FINDINGS sec. 13.C and the sec. 10.6 correction): the heldout
runs behind the "reactive, not persistent" reading were scored with the pre-fix
cg_probe, whose per-episode CV groups let GroupKFold split matched-pair twins
across folds and bias AUROC toward 0 whenever the surviving pair count was not a
multiple of 5. The raw tails were dumped per cell as
`states_d<drift>_s<seed>_<agent>_cg.npz` (keys: auth, surr; shape
(n_pairs, tail_steps, hidden)), so the corrected numbers are a pure CPU
recompute - no retraining.

Run this ON THE MACHINE THAT HOLDS the gitignored fullruns/ archives, e.g.:

    python scripts/reanalyze_cg_states.py fullruns/l3_h8_heldout/states \
        --json fullruns/l3_h8_heldout/cg_rescore.json
    python scripts/reanalyze_cg_states.py fullruns/l3_h7_heldout/states \
        --json fullruns/l3_h7_heldout/cg_rescore.json

It re-scores every cell with the fixed estimator (same probe seed convention as
the original run: cg_probe(..., seed=<cell seed>)), prints per-cell and
per-(drift x agent) aggregate tables with the OLD committed values alongside
where known, and flags the two decision-relevant readouts:

  - the drift-0.00 cg floors (must sit near 0.5 now; they read 0.001-0.27 under
    the biased estimator - the smoking gun),
  - the strongest-drift survival cg_tail / cg_latetail vs the 0.65 bar.
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

from itasorl.experiment_b2 import cg_probe

# non-greedy agent token so "survival_cg" parses as agent="survival"
FNAME = re.compile(r"states_d(?P<drift>[-\d.]+)_s(?P<seed>\d+)_(?P<agent>\w+?)_cg\.npz$")


def rescore_cell(path: str, seed: int) -> dict:
    with np.load(path) as npz:
        auth, surr = npz["auth"], npz["surr"]
    # cg_probe consumes lists of per-pair (T, H) tails; same seed convention as
    # the original run (run_expB2.evaluate_agent: cg_probe(at, st, seed=seed)).
    return cg_probe(list(auth), list(surr), seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("states_dir", help="directory holding states_*_cg.npz dumps")
    ap.add_argument("--json", default=None, help="optional path for the re-score JSON")
    ap.add_argument("--bar", type=float, default=0.65, help="pre-registered encoding bar")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.states_dir, "states_*_cg.npz")))
    if not files:
        raise SystemExit(f"no states_*_cg.npz found in {a.states_dir}")

    cells = []
    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    print(f"Re-scoring {len(files)} common-garden cells from {a.states_dir} "
          f"with the FIXED pair-grouped cg_probe\n")
    for f in files:
        m = FNAME.search(os.path.basename(f))
        if not m:
            print(f"  skip (unparsed name): {os.path.basename(f)}")
            continue
        drift, seed, agent = m["drift"], int(m["seed"]), m["agent"]
        r = rescore_cell(f, seed)
        row = {"drift": drift, "seed": seed, "agent": agent, **r}
        cells.append(row)
        agg[(drift, agent)].append(r)
        print(f"  d={drift} s={seed} {agent:10s} n_pairs={r['cg_n_pairs']:3d} "
              f"(n%5={r['cg_n_pairs'] % 5})  tail={r['cg_tail_target']:.3f} "
              f"[{r['cg_tail_lo']:.3f},{r['cg_tail_hi']:.3f}]  "
              f"latetail={r['cg_latetail_target']:.3f}")

    print("\n============  MEAN over seeds (per drift x agent)  ============")
    summary = {}
    for (drift, agent) in sorted(agg):
        rows = agg[(drift, agent)]
        tails = np.array([r["cg_tail_target"] for r in rows], float)
        lates = np.array([r["cg_latetail_target"] for r in rows], float)
        pairs = np.array([r["cg_n_pairs"] for r in rows], int)
        key = f"d{drift}_{agent}"
        summary[key] = {
            "n_seeds": len(rows),
            "cg_tail_mean": float(np.nanmean(tails)),
            "cg_tail_per_seed": [round(float(t), 4) for t in tails],
            "cg_latetail_mean": float(np.nanmean(lates)),
            "cg_latetail_per_seed": [round(float(t), 4) for t in lates],
            "cg_n_pairs_min": int(pairs.min()), "cg_n_pairs_max": int(pairs.max()),
        }
        print(f"  d={drift} {agent:10s} (n={len(rows)})  "
              f"tail={np.nanmean(tails):.3f}  latetail={np.nanmean(lates):.3f}  "
              f"pairs[{pairs.min()},{pairs.max()}]")

    # decision-relevant readouts
    drifts = sorted({d for (d, _) in agg}, key=float)
    print("\n============  DECISION READOUTS  ============")
    if "0.00" in {d for (d, _) in agg} or any(float(d) == 0.0 for (d, _) in agg):
        d0 = next(d for d in drifts if float(d) == 0.0)
        print(f"L0 floors at drift {d0} (biased estimator read 0.001-0.27; fixed must sit near 0.5):")
        for agent in sorted({g for (dd, g) in agg if dd == d0}):
            s = summary[f"d{d0}_{agent}"]
            ok = 0.4 <= s["cg_tail_mean"] <= 0.6
            print(f"  {agent:10s} tail={s['cg_tail_mean']:.3f} latetail={s['cg_latetail_mean']:.3f}"
                  f"  -> {'OK (chance band)' if ok else 'STILL OFF-CHANCE - investigate before citing'}")
    dmax = drifts[-1]
    key = f"d{dmax}_survival"
    if key in summary:
        s = summary[key]
        print(f"\nSurvival @ strongest drift {dmax} (old committed: tail 0.557 / latetail 0.492 at h8):")
        print(f"  corrected tail     = {s['cg_tail_mean']:.3f}  (bar {a.bar})")
        print(f"  corrected latetail = {s['cg_latetail_mean']:.3f}")
        # Frozen rule (PREREGISTRATION_L3 sec. 12, 2026-07-14 entry): survival
        # cg_tail_target >= bar AND > untrained + 0.05. The late-tail is the
        # separate persistence-decay diagnostic, never an OR-able pass channel.
        # (2026-07-18 audit fix: the earlier verdict line keyed on
        # max(tail, latetail) and omitted the untrained clause.)
        ukey = f"d{dmax}_untrained"
        tail = s["cg_tail_mean"]
        if not np.isfinite(tail):
            print("  -> INSUFFICIENT DATA: survival tail is NaN "
                  "(every seed under the 5-pair guard); no verdict")
        elif ukey not in summary or not np.isfinite(summary[ukey]["cg_tail_mean"]):
            print(f"  -> untrained aggregate unavailable; absolute clause "
                  f"{'reaches' if tail >= a.bar else 'misses'} the bar, but the "
                  "frozen rule needs the '> untrained + 0.05' clause - "
                  "adjudicate manually")
        else:
            floor = summary[ukey]["cg_tail_mean"]
            clause1, clause2 = tail >= a.bar, tail > floor + 0.05
            print(f"  untrained floor    = {floor:.3f}  (margin clause needs > {floor + 0.05:.3f})")
            if clause1 and clause2:
                verdict = ("PERSISTENT-signal candidate: corrected tail passes BOTH frozen "
                           "clauses - the 'reactive, not persistent' reading does NOT "
                           "survive as recorded")
            elif clause1:
                verdict = ("bar reached but the untrained+0.05 margin clause FAILS - "
                           "the frozen rule still adjudicates NEGATIVE (reactive reading "
                           "survives, now on a valid measurement)")
            else:
                verdict = ("below the bar - the reactive reading survives the estimator "
                           "fix (now on a valid measurement)")
            print(f"  -> {verdict}")
        print(f"  (latetail {s['cg_latetail_mean']:.3f} is the persistence-decay "
              "diagnostic, reported alongside, not part of the pass rule)")

    if a.json:
        d = os.path.dirname(a.json)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"states_dir": a.states_dir, "estimator": "cg_probe (pair-grouped, post-1633bca)",
                       "cells": cells, "aggregate": summary}, fh, indent=1, default=float)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
