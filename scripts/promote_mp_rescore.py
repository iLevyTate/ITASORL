"""Promote a corrected matched-pair (mp) re-score to a committed artifact.

The heldout runs' secondary mp channel was recorded pre-fix under the
assumption that pair count 60 co-locates twins in GroupKFold (FINDINGS 10.6
banner: "EXPECTED to be fold-safe ... should be confirmed (not assumed)").
`scripts/reanalyze_mp_readout.py` performed that confirmation and it FAILED:
twins were split under the sklearn in use, exactly as in the cg channel
(section 13.C). This script lifts the corrected aggregate out of the
(gitignored) `mp_rescore.json` and commits it with provenance.

No pre-registered decision rides on the mp channel (it is demoted to a
detectability index, FINDINGS 9/11), so this promotion corrects the record,
not a verdict. The committed gates are: every cell's singleton-group re-score
reproduced its stored pre-fix value exactly (determinism), and the drift-0.00
floors sit at chance under the fixed estimator.

Usage:
    python scripts/promote_mp_rescore.py \
        --rescore fullruns/l3_h8_heldout/mp_rescore.json \
        --out artifacts/expB2/heldout_l3_h8_mp_rescore.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

AGENTS = ["survival", "predictor", "untrained"]
DRIFT = "0.45"  # strongest drift; drift-0.00 is the floor read


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git optional at promote time
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default="fullruns/l3_h8_heldout/mp_rescore.json")
    ap.add_argument("--out", default="artifacts/expB2/heldout_l3_h8_mp_rescore.json")
    args = ap.parse_args()

    with open(args.rescore, encoding="utf-8") as fh:
        rs = json.load(fh)
    agg = rs["aggregate"]

    strong = {a: agg[f"d{DRIFT}_{a}"] for a in AGENTS}
    floor = {a: agg[f"d0.00_{a}"]["mp_fixed_mean"] for a in AGENTS}
    determinism_ok = all(v["all_reproduce_stored"] for v in agg.values())
    floor_ok = all(abs(v - 0.5) <= 0.05 for v in floor.values())

    out = {
        "source_rescore": args.rescore.replace("\\", "/"),
        "bundle": rs["bundle"].replace("\\", "/"),
        "l3_hidden": rs["l3_hidden"],
        "estimator": rs["estimator"],
        "mp_config": rs["mp_config"],
        "git_commit_at_promotion": git_head(),
        "generated_by": "scripts/promote_mp_rescore.py",
        "drift": DRIFT,
        "strong_drift": strong,
        "floor_drift0_fixed": floor,
        "gates": {
            "all_cells_reproduce_stored_prefix_value": determinism_ok,
            "floor_near_chance": floor_ok,
        },
        "note": ("secondary/demoted channel - detectability index, no frozen rule; "
                 "corrects the 'fold-safe' assumption recorded 2026-07-19"),
    }

    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, args.out)
    surv = strong["survival"]
    print(f"wrote {args.out}  (survival mp {surv['mp_old_mean']:.3f} -> "
          f"{surv['mp_fixed_mean']:.3f}, determinism={determinism_ok}, floors_ok={floor_ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
