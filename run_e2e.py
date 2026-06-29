"""
ITASORL end-to-end runner: pytest + all reproduction scripts in FINDINGS order.

Records every step under results/runs/<timestamp>/ (logs, metrics JSON, figures,
SUMMARY.md, bundle.zip).

Usage (from repo root):
    python run_e2e.py              # full suite + recorded results
    python run_e2e.py --quick      # all experiments; B-v2 at reduced scale
    python run_e2e.py --results-dir results/runs/my_run
    python run_e2e.py --no-zip     # skip zip bundle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from results_io import RunRecorder  # noqa: E402


def experiment_steps(*, quick: bool, b2_out: Path) -> list[tuple[str, list[str], list[str] | None]]:
    py = sys.executable
    b2 = [py, "run_expB2.py", "--out-dir", str(b2_out)]
    if quick:
        b2.append("--quick")
    return [
        ("expA_l1", [py, "run_expA.py"], None),
        ("expA_l2", [py, "run_expA_l2.py"], None),
        ("expB_smoke", [py, "experiment_b.py"], None),
        ("expB_full", [py, "run_expB_full.py"], None),
        ("expB_surprise", [py, "run_expB_surprise.py"], None),
        ("expB_kstep", [py, "run_expB_kstep.py"], None),
        ("expB_gap", [py, "run_expB_gap.py"], None),
        ("expB_nonlinear", [py, "run_expB_nonlinear.py"], None),
        ("expB2", b2, None),
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ITASORL pytest + all reproduction scripts.")
    ap.add_argument("--quick", action="store_true",
                    help="Run all experiments; B-v2 uses --quick (tiny scale).")
    ap.add_argument("--skip", action="append", default=[],
                    help="Skip steps: pytest, expA_l1, expA_l2, expB_smoke, expB_full, "
                         "expB_surprise, expB_kstep, expB_gap, expB_nonlinear, expB2, "
                         "or aliases: pytest, experiments, expA, expB.")
    ap.add_argument("--only", choices=("pytest", "experiments"),
                    help="Run only pytest or only the experiment scripts.")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Directory for this run's recorded output (default: results/runs/<timestamp>).")
    ap.add_argument("--no-zip", action="store_true", help="Do not create bundle.zip.")
    return ap.parse_args()


def expand_skip(raw: list[str]) -> set[str]:
    aliases = {
        "pytest": {"pytest"},
        "experiments": {s for s, _, _ in experiment_steps(quick=False, b2_out=Path("."))},
        "expA": {"expA_l1", "expA_l2"},
        "expB": {"expB_smoke", "expB_full", "expB_surprise", "expB_kstep",
                 "expB_gap", "expB_nonlinear"},
    }
    out: set[str] = set()
    for item in raw:
        key = item.strip().lower()
        if key in aliases:
            out |= aliases[key]
        else:
            out.add(key)
    return out


def main() -> None:
    args = parse_args()
    skip = expand_skip(args.skip)
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)

    recorder = RunRecorder.create(quick=args.quick, out_dir=args.results_dir)
    b2_out = recorder.run_dir / "artifacts"

    print(f"ITASORL end-to-end  (root={ROOT}, quick={args.quick})", flush=True)
    print(f"Recording results -> {recorder.run_dir}", flush=True)
    if skip:
        print(f"Skipping: {', '.join(sorted(skip))}", flush=True)

    t0 = time.perf_counter()

    if args.only != "experiments" and "pytest" not in skip:
        recorder.run_step("pytest", [sys.executable, "-m", "pytest", "-q"])

    if args.only != "pytest":
        for name, cmd, extra in experiment_steps(quick=args.quick, b2_out=b2_out):
            if name in skip:
                print(f"\n--- skip {name} ---", flush=True)
                recorder.manifest["steps"][name] = {"status": "skipped"}
                continue
            recorder.run_step(name, cmd, extra_artifacts=extra)

    run_dir = recorder.finalize(total_sec=time.perf_counter() - t0, make_zip=not args.no_zip)
    print(f"Read outcomes: {run_dir / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
