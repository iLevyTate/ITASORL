"""Promote Experiment A/B and B-v3 numbers to committed summary artifacts.

The L1/L2 oracle ceilings (FINDINGS 2), the Experiment B incidental-detection
arc (FINDINGS 3), and the B-v3 n=10 / capacity-ceiling results (FINDINGS 7.1)
were previously traceable only to gitignored `fullruns/` bundles. This script
extracts the decision-relevant numbers from those bundles into compact
committed JSONs with provenance.

Sources (local run bundles, gitignored):
    fullruns/06302026            full e2e pass (commit 4c16be6): expA/expB steps
    fullruns/kstep_rerun_20260713.log   recorded k-step rerun (FINDINGS 3.3)
    fullruns/07062026            B-v3 regime n=10 (survival 0.610)
    fullruns/07092026            sysid-aux capacity ceiling n=10 (0.596)

Usage:
    python scripts/promote_ab_summaries.py
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from itasorl.stats import mean_ci  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(path: str):
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return json.load(fh)


def provenance(bundle: str) -> dict:
    m = load(f"fullruns/{bundle}/manifest.json")
    return {"source_run": f"fullruns/{bundle}", "run_id": m.get("run_id"),
            "git_commit": m.get("git_commit"), "quick": m.get("quick"),
            "environment": m.get("environment"),
            "generated_by": "scripts/promote_ab_summaries.py"}


def write(path: str, doc: dict) -> None:
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    tmp = full + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, full)
    print(f"wrote {path}")


def parse_kstep_rerun(log_path: str) -> list[dict]:
    pat = re.compile(r"open_horizon=\s*(\d+):\s+drift0\.45 target=([\d.]+).([\d.]+)"
                     r"\s+control target=([\d.]+).([\d.]+)")
    rows = []
    with open(os.path.join(ROOT, log_path), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                rows.append({"open_horizon": int(m.group(1)),
                             "drift_045_target_mean": float(m.group(2)),
                             "drift_045_target_std": float(m.group(3)),
                             "control_target_mean": float(m.group(4)),
                             "control_target_std": float(m.group(5))})
    return rows


def log_sections(bundle: str) -> dict[str, str]:
    """Split a bundle's combined.log into per-step chunks ('STEP <name>' headers)."""
    path = os.path.join(ROOT, f"fullruns/{bundle}/combined.log")
    sections: dict[str, list[str]] = {}
    current = "_preamble"
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.match(r"STEP (\w+)", line)
            if m:
                current = m.group(1)
                sections[current] = []
            else:
                sections.setdefault(current, []).append(line)
    return {k: "".join(v) for k, v in sections.items()}


def parse_mean_std_table(text: str, pattern: str) -> list[dict]:
    rows = []
    for m in re.finditer(pattern, text):
        rows.append({"drift": float(m.group(1)), "mean": float(m.group(2)),
                     "std": float(m.group(3))})
    return rows


def pool_summary(bundle: str, drift: str = "0.45") -> dict:
    res = load(f"fullruns/{bundle}/artifacts/expB2_results.json")
    out = {}
    for agent in ("untrained", "predictor", "survival"):
        vals = [float(v) for v in res[drift][agent]["pool_target"]]
        mean, lo, hi = mean_ci(vals)  # seed-level bootstrap, 90%
        out[agent] = {"pool_target_per_seed": vals, "mean": mean,
                      "boot90_lo": lo, "boot90_hi": hi, "n_seeds": len(vals)}
    return out


def main() -> int:
    # -- Experiment A: oracle detectability ceilings (L1, L2) -----------------
    expa = provenance("06302026")
    expa["expA_l1"] = load("fullruns/06302026/steps/expA_l1.json")
    expa["expA_l2"] = load("fullruns/06302026/steps/expA_l2.json")
    write("artifacts/expA/summary.json", expa)

    # -- Experiment B: incidental-detection arc (L2) --------------------------
    expb = provenance("06302026")
    for step in ("expB_full", "expB_surprise", "expB_kstep",
                 "expB_nonlinear", "expB_gap"):
        expb[step] = load(f"fullruns/06302026/steps/{step}.json")
    rerun = parse_kstep_rerun("fullruns/kstep_rerun_20260713.log")
    if not rerun:
        print("ERROR: could not parse kstep rerun log", file=sys.stderr)
        return 1
    expb["expB_kstep_rerun_20260713"] = {
        "source": "fullruns/kstep_rerun_20260713.log",
        "note": "recorded rerun backing the FINDINGS 3.3 correction",
        "horizons": rerun,
    }
    # the step JSONs drop the across-seed stds; recover them from the run log
    secs = log_sections("06302026")
    expb["expB_surprise"]["drift_sweep_with_std"] = parse_mean_std_table(
        secs.get("expB_surprise", ""),
        r"drift=([\d.]+)\s+surprise-probe AUROC = ([\d.]+) . ([\d.]+)")
    expb["expB_nonlinear"]["drift_sweep_with_std"] = parse_mean_std_table(
        secs.get("expB_nonlinear", ""),
        r"drift=([\d.]+):\s+target=([\d.]+).([\d.]+)\s")
    write("artifacts/expB/summary.json", expb)

    # -- B-v3 regime, n = 10 (FINDINGS 7.1: survival 0.610 [0.585, 0.634]) ----
    bv3 = provenance("07062026")
    bv3["flags"] = load("fullruns/07062026/b2_flags.json")
    bv3["pooled_target_drift045"] = pool_summary("07062026")
    write("artifacts/expB2/bv3_n10_summary.json", bv3)

    # -- sysid-aux capacity ceiling, n = 10 (FINDINGS 7.1: 0.596 [0.577,0.616])
    ceil = provenance("07092026")
    ceil["flags"] = load("fullruns/07092026/b2_flags.json")
    ceil["pooled_target_drift045"] = pool_summary("07092026")
    write("artifacts/expB2/sysid_ceiling_n10_summary.json", ceil)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
