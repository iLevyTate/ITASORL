"""Promote the held-out probe results to a committed artifact.

The held-out / common-garden numbers in FINDINGS 10.6 come from the (gitignored)
run bundle `fullruns/l3_h8_heldout`. This script extracts the decision-relevant
per-seed values from that bundle's cell files and writes a compact, committed
summary JSON with provenance, so every published 10.6 number is traceable
in-repo.

Usage:
    python scripts/promote_heldout_artifact.py \
        --run-dir fullruns/l3_h8_heldout \
        --out artifacts/expB2/heldout_l3_h8_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itasorl.behavior_audit import aggregate_cells  # noqa: E402

BAR = 0.65

POOL_KEYS = {  # source key in cell["agents"][arm]["pool"] -> summary row key
    "target": "pool_target",
    "pool_reward_leak": "pool_reward_leak",
    "pool_leak_clean": "pool_leak_clean",
    "deaths_auth": "deaths_auth",
    "deaths_surr": "deaths_surr",
    "n_auth": "n_auth",
    "n_surr": "n_surr",
}
HELDOUT_KEYS = ["transfer_target", "cg_tail_target", "cg_latetail_target",
                "transfer_deaths_auth", "transfer_deaths_surr",
                "transfer_n_auth", "transfer_n_surr", "cg_n_pairs"]
AGG_METRICS = ["pool_target", "pool_reward_leak",
               "transfer_target", "cg_tail_target", "cg_latetail_target"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="fullruns/l3_h8_heldout")
    ap.add_argument("--out", default="artifacts/expB2/heldout_l3_h8_summary.json")
    args = ap.parse_args()

    cell_files = sorted(glob.glob(os.path.join(args.run_dir, "cells", "cell_*.json")))
    if not cell_files:
        print(f"ERROR: no cell files under {args.run_dir}/cells", file=sys.stderr)
        return 1

    fingerprints, commits, rows = set(), set(), []
    for path in cell_files:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        fingerprints.add(data["fingerprint"])
        commits.add(data["git_commit"])
        cell = data["cell"]
        for agent, payload in cell["agents"].items():
            row = {"drift": f"{float(cell['drift']):.2f}",
                   "seed": cell["seed"], "agent": agent}
            for src, dst in POOL_KEYS.items():
                if src in payload.get("pool", {}):
                    row[dst] = payload["pool"][src]
            for k in HELDOUT_KEYS:
                if k in payload.get("heldout", {}):
                    row[k] = payload["heldout"][k]
            rows.append(row)

    if len(fingerprints) != 1:
        print(f"ERROR: mixed fingerprints in bundle: {sorted(fingerprints)}", file=sys.stderr)
        return 1

    aggregate = {}
    drifts = sorted({r["drift"] for r in rows})
    agents = sorted({r["agent"] for r in rows})
    for d in drifts:
        for a in agents:
            sub = [r for r in rows if r["drift"] == d and r["agent"] == a]
            numeric = [{k: v for k, v in r.items()
                        if k in AGG_METRICS and isinstance(v, (int, float))}
                       for r in sub]
            agg = aggregate_cells(numeric, bar=BAR)
            if agg:
                aggregate[f"d={float(d):.2f} {a}"] = agg

    out = {
        "source_run": args.run_dir.replace("\\", "/"),
        "fingerprint": sorted(fingerprints)[0],
        "git_commit": sorted(commits),
        "generated_by": "scripts/promote_heldout_artifact.py",
        "bar": BAR,
        "cells": rows,
        "aggregate": aggregate,
    }
    d = os.path.dirname(args.out)
    if d:  # bare filename -> dirname "" -> makedirs would raise after all cells parsed
        os.makedirs(d, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}  ({len(rows)} rows, fingerprint {sorted(fingerprints)[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
