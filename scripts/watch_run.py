"""
Tail the latest ITASORL e2e run in real time.

Usage (from repo root, in a second terminal while run_e2e.py is running):
    python scripts/watch_run.py --follow
    python scripts/watch_run.py --follow --run-dir fullruns/06292026
    python scripts/watch_run.py --status   # print status.json once and exit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from itasorl.results_io import LATEST_RUN_PTR, read_latest_run_dir  # noqa: E402


def resolve_run_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_dir():
            raise FileNotFoundError(f"Run directory not found: {explicit}")
        return explicit.resolve()
    run_dir = read_latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError(
            f"No latest run found. Expected {LATEST_RUN_PTR} to point at a run directory."
        )
    return run_dir


def print_status(run_dir: Path) -> None:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        print(f"(no status.json yet in {run_dir})")
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    print(f"Run:     {status.get('run_dir', run_dir)}")
    print(f"Step:    {status.get('current_step')} ({status.get('step_status')})")
    print(f"Elapsed: {status.get('elapsed_sec')} s")
    print(f"Running: {status.get('running')}")
    last = status.get("last_line") or ""
    if last:
        print(f"Last:    {last}")


def follow_log(run_dir: Path) -> None:
    combined = run_dir / "combined.log"
    status_path = run_dir / "status.json"
    print(f"Watching {combined}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print("=" * 72, flush=True)

    while not combined.is_file():
        if status_path.is_file():
            st = json.loads(status_path.read_text(encoding="utf-8"))
            if st.get("step_status") == "finished":
                return
        time.sleep(0.3)

    with combined.open(encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                print(line, end="", flush=True)
                continue
            finished = False
            if status_path.is_file():
                st = json.loads(status_path.read_text(encoding="utf-8"))
                finished = not st.get("running", True) and st.get("step_status") == "finished"
            if finished:
                # Drain any trailing bytes written after the last read.
                rest = fh.read()
                if rest:
                    print(rest, end="", flush=True)
                break
            time.sleep(0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch an ITASORL e2e run in real time.")
    ap.add_argument("--run-dir", type=Path, default=None, help="Run folder (default: LATEST_RUN.txt)")
    ap.add_argument("--follow", "-f", action="store_true", help="Tail combined.log until the run finishes")
    ap.add_argument("--status", action="store_true", help="Print status.json once")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)

    if args.status and not args.follow:
        print_status(run_dir)
        return

    if args.follow:
        follow_log(run_dir)
        print("\n" + "=" * 72, flush=True)
        print_status(run_dir)
        return

    print_status(run_dir)
    print(f"\nTail live log: python scripts/watch_run.py --follow --run-dir {run_dir}")


if __name__ == "__main__":
    main()
