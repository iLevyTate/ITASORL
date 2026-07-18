"""Promote the Experiment C emergence pilot to a committed artifact.

The milestone-3 emergence numbers in FINDINGS come from the (gitignored) run
bundle `fullruns/expC_milestone3/emergence_pilot.json`. This script extracts the
decision-relevant per-seed values (the AUROC deltas that feed the emergence
contrast, plus the fitness-move and separate-survival series that back the
mechanism read) and writes a compact, committed summary JSON with provenance, so
every published Exp C number is recomputable in-repo by
scripts/audit_stats_recheck.py.

Unlike the L3 bundles there is no per-cell fingerprint hash; the run's identity
is the frozen world P and the frozen L3 surrogate, both recorded verbatim from
the source, alongside the promote-time git HEAD.

Usage:
    python scripts/promote_expC_summary.py \
        --run fullruns/expC_milestone3/emergence_pilot.json \
        --out artifacts/expC/emergence_pilot_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

CONFIG_KEYS = ["n", "generations", "seeds", "sigma", "q", "drift_sigma",
               "n_eps", "max_steps", "embed", "hidden", "l3_hidden", "l3_seed",
               "panel_pairs", "panel_prefix", "panel_tail"]


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git optional at promote time
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fullruns/expC_milestone3/emergence_pilot.json")
    ap.add_argument("--out", default="artifacts/expC/emergence_pilot_summary.json")
    args = ap.parse_args()

    with open(args.run, encoding="utf-8") as fh:
        run = json.load(fh)

    cells = []
    for ps in sorted(run["per_seed"], key=lambda r: r["seed"]):
        sg = ps["survival_gen0"]
        st = ps["survival_final_treat"]
        sc = ps["survival_final_ctrl"]
        cells.append({
            "seed": ps["seed"],
            "gen0_auroc": ps["auroc_gen0"],
            "final_treat_auroc": ps["auroc_final_treat"],
            "final_ctrl_auroc": ps["auroc_final_ctrl"],
            "fit_delta_treat": ps["fit_delta_treat"],
            "fit_delta_ctrl": ps["fit_delta_ctrl"],
            "death_rate_auth_gen0": sg["death_rate_auth"],
            "death_rate_auth_final_treat": st["death_rate_auth"],
            "death_rate_auth_final_ctrl": sc["death_rate_auth"],
            "death_rate_surr_gen0": sg["death_rate_surr"],
        })

    est = run["estimand"]
    out = {
        "source_run": args.run.replace("\\", "/"),
        "world": run["world"],
        "surrogate": run["surrogate"],
        "control_food": run["control_food"],
        # provenance, disambiguated (2026-07-18 audit): the commit the RUN
        # executed on comes from the run JSON itself (runner records it since
        # the same audit); the promote-time commit is stamped separately and
        # must never be read as the run's code version.
        "git_commit_at_run": run.get("git_commit_at_run", "unknown (pre-audit run JSON)"),
        "git_commit_at_promotion": git_head(),
        "generated_by": "scripts/promote_expC_summary.py",
        "bars": {"sesoi": 0.05, "auroc_floor": 0.65},
        "config": {k: run["config"][k] for k in CONFIG_KEYS if k in run["config"]},
        "cells": cells,
        "estimand": {
            "delta_treat": est["delta_treat"],
            "delta_ctrl": est["delta_ctrl"],
            "contrast": est["contrast"],
            "mean_contrast": est["mean_contrast"],
            "t_ci90": est["t_ci90"],
            "boot_ci90": est["boot_ci90"],
            "mean_final_treat_auroc": est["mean_final_treat_auroc"],
            "n_seeds": est["n_seeds"],
            "ci_excludes_zero": est["ci_excludes_zero"],
            "meets_sesoi": est["meets_sesoi"],
            "meets_auroc_floor": est["meets_auroc_floor"],
            "emergence_claim": est["emergence_claim"],
        },
        "gates": {
            # full sec.-7 battery (post-invalidation runs record it; the invalid
            # pilot carried only gate 2 + determinism, which are kept for compat)
            **run.get("gates", {}),
            "gates_pass_all": run.get("gates_pass_all"),
            "routing": run.get("routing"),
            "gate2_fitness_moves_treat": run["gate2_fitness_moves_treat"],
            "gate2_fitness_moves_ctrl": run["gate2_fitness_moves_ctrl"],
            "determinism_bit_reproducible": run["determinism_bit_reproducible"],
        },
        "wall_seconds": run["wall_seconds"],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}  ({len(cells)} seeds, claim={out['estimand']['emergence_claim']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
