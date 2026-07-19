"""Promote a corrected common-garden re-score to a committed artifact.

FINDINGS 10.6's common-garden channel was first scored with the pre-fix
pair-splitting `cg_probe` estimator (biased toward AUROC 0; see section 13.C).
`scripts/reanalyze_cg_states.py` re-scores the saved `_cg.npz` tail dumps with
the fixed estimator and writes a (gitignored) `cg_rescore.json` under the run
bundle. This script lifts the decision-relevant aggregate out of that re-score
and writes a compact, committed summary JSON with provenance, so the corrected
10.6 numbers and the frozen-rule adjudication are recomputable in-repo by
scripts/audit_stats_recheck.py.

The frozen rule (2026-07-14/15) is applied here verbatim on the strongest-drift
survival cell: cg_channel_pass = (survival tail >= 0.65) AND
(survival tail > untrained tail + 0.05). The drift-0.00 floors are carried as a
sanity read (they must sit near 0.5 under the fixed estimator).

Usage:
    python scripts/promote_cg_rescore.py \
        --rescore fullruns/l3_h8_heldout/cg_rescore.json \
        --out artifacts/expB2/heldout_l3_h8_cg_rescore.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

BAR = 0.65
MARGIN = 0.05
DRIFT = "0.45"  # strongest drift, the decision cell
AGENTS = ["survival", "predictor", "untrained"]


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git optional at promote time
        return "unknown"


def _cell(agg: dict, drift: str, agent: str) -> dict:
    a = agg[f"d{drift}_{agent}"]
    return {
        "cg_tail_mean": a["cg_tail_mean"],
        "cg_tail_per_seed": a["cg_tail_per_seed"],
        "cg_latetail_mean": a["cg_latetail_mean"],
        "cg_n_pairs_min": a["cg_n_pairs_min"],
        "cg_n_pairs_max": a["cg_n_pairs_max"],
        "n_seeds": a["n_seeds"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default="fullruns/l3_h8_heldout/cg_rescore.json")
    ap.add_argument("--out", default="artifacts/expB2/heldout_l3_h8_cg_rescore.json")
    args = ap.parse_args()

    with open(args.rescore, encoding="utf-8") as fh:
        rs = json.load(fh)
    agg = rs["aggregate"]

    strong = {a: _cell(agg, DRIFT, a) for a in AGENTS}
    floor = {a: agg[f"d0.00_{a}"]["cg_tail_mean"] for a in AGENTS}

    surv = strong["survival"]["cg_tail_mean"]
    untr = strong["untrained"]["cg_tail_mean"]
    pass_bar = surv >= BAR
    pass_margin = surv > untr + MARGIN
    floor_ok = all(abs(v - 0.5) <= 0.05 for v in floor.values())

    out = {
        "source_rescore": args.rescore.replace("\\", "/"),
        "states_dir": rs["states_dir"].replace("\\", "/"),
        "estimator": rs["estimator"],
        "git_commit_at_promotion": git_head(),
        "generated_by": "scripts/promote_cg_rescore.py",
        "bar": BAR,
        "margin": MARGIN,
        "drift": DRIFT,
        "strong_drift": strong,
        "floor_drift0": floor,
        "adjudication": {
            "survival_cg_tail": surv,
            "untrained_cg_tail": untr,
            "margin_threshold": untr + MARGIN,
            "cg_tail_pass_bar": pass_bar,
            "cg_tail_pass_margin": pass_margin,
            "cg_channel_pass": pass_bar and pass_margin,
            "floor_near_chance": floor_ok,
        },
    }

    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}  (survival cg_tail={surv:.4f}, "
          f"cg_channel_pass={out['adjudication']['cg_channel_pass']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
