"""Promote the H2 substrate-grounding ablation (A1) to a committed artifact.

The A1 graded-seam sweep numbers in FINDINGS come from the (gitignored) run
bundle `fullruns/expH2_ablation/aggregate.json`. This script extracts the
decision-relevant per-seed values (the survival and untrained pooled targets at
each alpha, the collapse curve, its monotonicity, and the alpha=0 L0 anchor) and
writes a compact, committed summary JSON with provenance, so every published H2
number is recomputable in-repo by scripts/audit_stats_recheck.py.

Like the Exp C bundle there is no per-cell fingerprint hash; the run's identity is
the frozen world P and the frozen L3 hidden=8 surrogate, recorded verbatim,
alongside the promote-time git HEAD (which is also the run's code version, since
the runner and this promotion sit on the same commit).

Usage:
    python scripts/promote_expH2_summary.py \
        --run fullruns/expH2_ablation/aggregate.json \
        --out artifacts/expH2/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git optional at promote time
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fullruns/expH2_ablation/aggregate.json")
    ap.add_argument("--out", default="artifacts/expH2/summary.json")
    args = ap.parse_args()

    with open(args.run, encoding="utf-8") as fh:
        run = json.load(fh)

    alphas = run["alphas"]
    arms = run["arms"]
    cells: dict[str, dict[str, dict]] = {}
    for arm in arms:
        cells[arm] = {}
        for al in alphas:
            key = f"{arm}_a{al:.2f}"
            cells[arm][f"a{al:.2f}"] = {
                "alpha": al,
                "per_seed": run[f"{key}_per_seed"],
                "mean": run[f"{key}_mean"],
                "tci90": run[f"{key}_tci90"],
                "n_ge_065": run[f"{key}_n_ge_065"],
            }

    head = git_head()
    out = {
        "source_run": args.run.replace("\\", "/"),
        "world": "WorldParams(k_land=1.5, k_water=1.5, gravity=0.4) [P]",
        "surrogate": ("L3 GMotion hidden=8 seed=0 trained on P; A1 seam neutralization = "
                      "GradedGMotion, convex blend (1-alpha)*authentic_law + alpha*G"),
        "git_commit_at_run": head,
        "git_commit_at_promotion": head,
        "generated_by": "scripts/promote_expH2_summary.py",
        "bars": {"auroc_floor": 0.65, "l0_rope": [0.45, 0.55]},
        "config": {"drift": run["drift"], "alphas": alphas, "arms": arms,
                   "n_eps": run["n_eps"], "steps": run["steps"], "n_seeds": len(cells[arms[0]]["a1.00"]["per_seed"]),
                   "device": "cuda"},
        "survival_curve": run["survival_curve"],
        "survival_monotonicity_rho": run["survival_monotonicity_rho"],
        "l0_anchor_alpha0": run["l0_anchor_alpha0"],
        "integrity": run["integrity"],
        "cells": cells,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, args.out)
    curve = out["survival_curve"]
    print(f"wrote {args.out}  (survival curve {curve[0]:.3f}->{curve[-1]:.3f}, "
          f"rho={out['survival_monotonicity_rho']}, L0 accept={out['l0_anchor_alpha0']['accept_equiv']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
