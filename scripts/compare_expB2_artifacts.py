"""
Compare Experiment B-v2 JSON outputs side-by-side (no GPU).

Usage (from repo root):
    python scripts/compare_expB2_artifacts.py
    python scripts/compare_expB2_artifacts.py --run fullruns/06302026
    python scripts/compare_expB2_artifacts.py --run fullruns/06302026/artifacts/expB2_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "artifacts" / "expB2" / "expB2_results.json"
DEFAULT_LAB = ROOT / "artifacts" / "expB2" / "expB2_results_confirmatory_n3.json"
SESOI = 0.65
DRIFT_KEY = "0.45"
AGENT = "survival"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_run_json(run: Path | None) -> Path | None:
    if run is None:
        ptr = ROOT / "results" / "LATEST_RUN.txt"
        if not ptr.is_file():
            return None
        run_dir = Path(ptr.read_text().strip())
        candidate = run_dir / "artifacts" / "expB2_results.json"
        return candidate if candidate.is_file() else None
    run = run.resolve()
    if run.is_file():
        return run
    candidate = run / "artifacts" / "expB2_results.json"
    return candidate if candidate.is_file() else None


def _survival_at_drift(data: dict, drift: str = DRIFT_KEY) -> dict:
    block = data.get(drift, {}).get(AGENT, {})
    targets = block.get("pool_target") or []
    lo = block.get("pool_target_lo")
    hi = block.get("pool_target_hi")
    drag = block.get("pool_ceiling_drag")
    out: dict = {"seeds": targets}
    if targets:
        vals = [float(x) for x in targets if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            out["mean"] = mean
            out["std"] = math.sqrt(var)
            out["n"] = len(vals)
            out["vs_sesoi"] = mean - SESOI
            out["meets_sesoi"] = mean >= SESOI
    if lo and hi and len(lo) == len(targets):
        out["ci_per_seed"] = list(zip(lo, hi))
    if drag:
        drag_vals = [float(x) for x in drag if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if drag_vals:
            out["drag_ceiling_mean"] = sum(drag_vals) / len(drag_vals)
    return out


def _fmt_stats(label: str, path: Path | None, stats: dict | None) -> list[str]:
    lines = [f"## {label}"]
    if path is not None:
        lines.append(f"Path: `{path}`")
    if not stats or "seeds" not in stats:
        lines.append("(missing survival @ drift 0.45 data)")
        return lines
    seeds = stats["seeds"]
    lines.append(f"Per-seed pool_target: {', '.join(f'{s:.3f}' for s in seeds)}")
    if "mean" in stats:
        lines.append(
            f"Mean +/- std (n={stats['n']}): **{stats['mean']:.3f} +/- {stats.get('std', 0):.3f}**"
        )
        lines.append(f"vs SESOI {SESOI}: {stats['vs_sesoi']:+.3f}  ->  {'PASS' if stats['meets_sesoi'] else 'FAIL'}")
    if "drag_ceiling_mean" in stats:
        lines.append(f"Drag ceiling (mean): {stats['drag_ceiling_mean']:.3f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side B-v2 survival @ drift 0.45 comparison.")
    ap.add_argument("--run", type=Path, default=None,
                    help="Run dir or expB2_results.json (default: results/LATEST_RUN.txt)")
    ap.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    ap.add_argument("--lab", type=Path, default=DEFAULT_LAB)
    args = ap.parse_args()

    run_path = _resolve_run_json(args.run)
    sections: list[str] = ["# Experiment B-v2 comparison (survival @ drift 0.45)", ""]

    if args.lab.is_file():
        sections.extend(_fmt_stats("Initial lab (confirmatory n=3)", args.lab,
                                   _survival_at_drift(_load(args.lab))))
    else:
        sections.extend(_fmt_stats("Initial lab (confirmatory n=3)", args.lab, None))
    sections.append("")

    if args.canonical.is_file():
        sections.extend(_fmt_stats("Canonical (Colab replication)", args.canonical,
                                   _survival_at_drift(_load(args.canonical))))
    else:
        sections.extend(_fmt_stats("Canonical (Colab replication)", args.canonical, None))
    sections.append("")

    if run_path and run_path.is_file():
        sections.extend(_fmt_stats("This run", run_path, _survival_at_drift(_load(run_path))))
    elif args.run is not None:
        sections.extend(_fmt_stats("This run", args.run, None))
    else:
        sections.append("## This run")
        sections.append("(no --run and no LATEST_RUN.txt)")

    text = "\n".join(sections)
    print(text)
    if run_path and run_path.is_file():
        run_stats = _survival_at_drift(_load(run_path))
        if run_stats.get("meets_sesoi"):
            sys.exit(0)
        if "mean" in run_stats:
            print("\nPre-registered verdict: H_B2 NOT met (below 0.65 SESOI).")


if __name__ == "__main__":
    main()
