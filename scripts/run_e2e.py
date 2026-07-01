"""
ITASORL end-to-end runner: pytest + all reproduction scripts in FINDINGS order.

Records every step under fullruns/<MMDDYYYY>/ by default (logs, metrics JSON,
figures, SUMMARY.md, bundle.zip). Override with --results-dir.

Live output while running: combined.log and status.json are updated incrementally.
In a second terminal: python scripts/watch_run.py --follow

Usage (from repo root):
    python scripts/run_e2e.py              # full suite + recorded results
    python scripts/run_e2e.py --quick      # all experiments; B-v2 at reduced scale
    python scripts/run_e2e.py --results-dir results/runs/my_run
    python scripts/run_e2e.py --resume     # continue latest interrupted run
    python scripts/run_e2e.py --resume fullruns/06292026_143022
    python scripts/run_e2e.py --no-zip     # skip zip bundle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from itasorl.results_io import LATEST_RUN_PTR, RunRecorder, read_latest_run_dir  # noqa: E402


def _script(name: str) -> str:
    return str(SCRIPTS / name)


def experiment_steps(*, quick: bool, b2_out: Path,
                     b2_extra: list[str] | None = None) -> list[tuple[str, list[str], list[str] | None]]:
    py = sys.executable
    b2 = [py, _script("run_expB2.py"), "--out-dir", str(b2_out)]
    if quick:
        b2.append("--quick")
    if b2_extra:
        b2.extend(b2_extra)
    return [
        ("expA_l1", [py, _script("run_expA.py")], None),
        ("expA_l2", [py, _script("run_expA_l2.py")], None),
        ("expB_smoke", [py, "-m", "itasorl.experiment_b"], None),
        ("expB_full", [py, _script("run_expB_full.py")], None),
        ("expB_surprise", [py, _script("run_expB_surprise.py")], None),
        ("expB_kstep", [py, _script("run_expB_kstep.py")], None),
        ("expB_gap", [py, _script("run_expB_gap.py")], None),
        ("expB_nonlinear", [py, _script("run_expB_nonlinear.py")], None),
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
                    help="Directory for this run's recorded output (default: fullruns/<MMDDYYYY>/).")
    ap.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        help="Continue an interrupted run. Optional PATH, or omit value to use results/LATEST_RUN.txt.",
    )
    ap.add_argument("--no-zip", action="store_true", help="Do not create bundle.zip.")
    ap.add_argument("--b2-seeds", type=int, nargs="+", default=None,
                    help="Override Experiment B-v2 seeds (e.g. 0 1 2 ... 15 to power the null).")
    ap.add_argument("--b2-updates", type=int, default=None, help="Override B-v2 training updates.")
    ap.add_argument("--b2-hidden", type=int, default=None, help="Override B-v2 recurrent hidden size.")
    ap.add_argument("--b2-dump-states", type=str, default=None,
                    help="Persist B-v2 recurrent states to this dir (forwarded to run_expB2.py "
                         "--dump-states) for offline variance/selectivity re-probing with "
                         "scripts/reanalyze_expB2_states.py.")
    ap.add_argument("--b2-sysid-aux", action="store_true",
                    help="Run B-v2 with the system-ID CEILING control (forwards --sysid-aux). "
                         "Breaks readout-not-reward; report separately from the headline.")
    return ap.parse_args()


def build_b2_extra(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.b2_seeds is not None:
        extra += ["--seeds", *[str(s) for s in args.b2_seeds]]
    if args.b2_updates is not None:
        extra += ["--updates", str(args.b2_updates)]
    if args.b2_hidden is not None:
        extra += ["--hidden", str(args.b2_hidden)]
    if args.b2_dump_states is not None:
        extra += ["--dump-states", args.b2_dump_states]
    if args.b2_sysid_aux:
        extra += ["--sysid-aux"]
    return extra


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


def resolve_resume_dir(resume: str | None, results_dir: Path | None) -> Path | None:
    if resume is None:
        return None
    if resume == "latest":
        run_dir = read_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError(
                f"--resume requested but {LATEST_RUN_PTR} does not point at a run directory."
            )
        return run_dir
    path = Path(resume)
    if not path.is_dir():
        raise FileNotFoundError(f"--resume path not found: {path}")
    return path.resolve()


def main() -> None:
    args = parse_args()
    skip = expand_skip(args.skip)
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)

    resume_dir = resolve_resume_dir(args.resume, args.results_dir)
    if resume_dir is not None:
        if args.results_dir is not None and resume_dir.resolve() != args.results_dir.resolve():
            print(
                f"Warning: --results-dir ({args.results_dir}) differs from resume dir ({resume_dir}); "
                "using resume dir.",
                flush=True,
            )
        recorder = RunRecorder.resume(resume_dir)
        quick = recorder.quick
        print(f"ITASORL end-to-end RESUME  (root={ROOT}, quick={quick})", flush=True)
    else:
        recorder = RunRecorder.create(quick=args.quick, out_dir=args.results_dir)
        quick = args.quick
        print(f"ITASORL end-to-end  (root={ROOT}, quick={quick})", flush=True)

    b2_out = recorder.run_dir / "artifacts"
    print(f"Recording results -> {recorder.run_dir}", flush=True)
    if skip:
        print(f"Skipping: {', '.join(sorted(skip))}", flush=True)

    t0 = time.perf_counter()

    if args.only != "experiments" and "pytest" not in skip:
        if recorder.step_is_done("pytest"):
            print("\n--- resume skip pytest (already ok) ---", flush=True)
        else:
            recorder.run_step("pytest", [sys.executable, "-m", "pytest", "-q"], cwd=ROOT)

    if args.only != "pytest":
        b2_extra = build_b2_extra(args)
        for name, cmd, extra in experiment_steps(quick=quick, b2_out=b2_out, b2_extra=b2_extra):
            if name in skip:
                print(f"\n--- skip {name} ---", flush=True)
                recorder.note_step(name, status="skipped")
                continue
            if recorder.step_is_done(name):
                print(f"\n--- resume skip {name} (already ok) ---", flush=True)
                continue
            recorder.run_step(name, cmd, cwd=ROOT, extra_artifacts=extra)

    run_dir = recorder.finalize(total_sec=time.perf_counter() - t0, make_zip=not args.no_zip)
    print(f"Read outcomes: {run_dir / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
