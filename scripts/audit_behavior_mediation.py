"""ITASORL - offline behavior-mediation audit of dumped pooled states (NO GPU).

The pooled world-identity positive could be reading the agent's BEHAVIOR
(it moves and forages differently across worlds) rather than a world-identity
representation. This script makes that audit reproducible: for every dumped
(drift, seed, agent) cell it reports

    target                   the uncontrolled headline probe (sanity anchor)
    behavior_only[/nonlinear]  can per-episode behavior means decode the world?
    resid_epmean[/quad]      state probe after the IN-FOLD per-episode control
    behavior_trace_only      (new dumps only) richer behavior ceiling
    resid_trace[/quad]       (new dumps only) per-timestep control, strictly
                             stronger than the per-episode one

and the across-seed aggregation (mean, 90% CI, seeds clearing the bar).
Old dumps (no bta/bts traces) get the per-episode metrics only.

Usage:  python scripts/audit_behavior_mediation.py fullruns/l3_n10_audited/states
        [--json out.json] [--bar 0.65]
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

from itasorl.behavior_audit import aggregate_cells, audit_cell

FNAME = re.compile(r"states_d(?P<drift>[-\d.]+)_s(?P<seed>\d+)_(?P<agent>\w+)\.npz$")

COLUMNS = ("target", "behavior_only", "behavior_only_nonlinear",
           "resid_epmean", "resid_epmean_quad",
           "behavior_trace_only", "resid_trace", "resid_trace_quad")


def _fmt(res: dict, key: str) -> str:
    return f"{res[key]:.3f}" if key in res and np.isfinite(res[key]) else "  -  "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("states_dir", help="directory of states_*.npz written by --dump-states")
    ap.add_argument("--bar", type=float, default=0.65,
                    help="pre-registered AUROC bar for the seed count (default 0.65)")
    ap.add_argument("--json", default=None, help="also write per-cell + aggregate results here")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.states_dir, "states_*.npz")))
    if not files:
        raise SystemExit(f"no states_*.npz found in {a.states_dir}")

    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    cells = []
    print(f"Behavior-mediation audit of {len(files)} dumped cells from {a.states_dir}")
    print("(in-fold controls; in-sample residualization over-removes and is not used)\n")
    for f in files:
        m = FNAME.search(os.path.basename(f))
        if not m:
            print(f"  skip (unparsed name): {os.path.basename(f)}")
            continue
        drift, seed, agent = m["drift"], int(m["seed"]), m["agent"]
        with np.load(f) as npz:
            res = audit_cell(dict(npz), seed=seed)
        if not res:
            print(f"  skip (too few survivors): {os.path.basename(f)}")
            continue
        agg[(drift, agent)].append(res)
        cells.append({"drift": drift, "seed": seed, "agent": agent, **res})
        print(f"  d={drift} s={seed} {agent:10s} "
              f"target={_fmt(res, 'target')} beh={_fmt(res, 'behavior_only')}"
              f"/{_fmt(res, 'behavior_only_nonlinear')} "
              f"resid={_fmt(res, 'resid_epmean')}/{_fmt(res, 'resid_epmean_quad')} "
              f"trace_beh={_fmt(res, 'behavior_trace_only')} "
              f"resid_t={_fmt(res, 'resid_trace')}/{_fmt(res, 'resid_trace_quad')}")

    summary = {}
    print(f"\n=====  across-seed mean [90% CI]  (n seeds >= {a.bar:.2f})  =====")
    for (drift, agent) in sorted(agg):
        rows = agg[(drift, agent)]
        stats = aggregate_cells(rows, bar=a.bar)
        summary[f"d={drift} {agent}"] = stats
        print(f"\n  d={drift} {agent} (n={len(rows)})")
        for key in COLUMNS:
            if key not in stats:
                continue
            s = stats[key]
            print(f"    {key:24s} {s['mean']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]  "
                  f"({s['n_above_bar']}/{s['n_seeds']})")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"cells": cells, "aggregate": summary, "bar": a.bar}, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
