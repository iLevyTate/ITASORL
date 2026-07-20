"""
ITASORL - run any RUN_PROFILE preset locally, with resume.

Launches scripts/run_e2e.py --profile <name> (the profile table lives in
run_e2e.py; the Colab notebook uses the same presets) plus local preflight
checks (CUDA visible, enough free RAM). expB2 cells checkpoint after every
(drift, seed) pair, so an interrupted run continues with --resume losing at
most one cell.

Usage (from repo root, Git Bash or any shell):
    python scripts/run_local.py --list
    python scripts/run_local.py bv3_regime_n10            # fresh start
    python scripts/run_local.py bv3_regime_n10 --resume   # continue latest run
    python scripts/run_local.py quick --allow-cpu         # smoke test w/o GPU
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import contextlib
import ctypes
import json
import subprocess
import sys
from pathlib import Path

from itasorl.results_io import default_run_dir, read_latest_run_dir

SCRIPTS = Path(__file__).resolve().parent

from run_e2e import PROFILES  # noqa: E402  single source of truth

# Windows SetThreadExecutionState flags (winbase.h).
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _set_sleep_prevention(active: bool) -> bool:
    """Ask Windows to keep the SYSTEM awake (the display may still sleep, which
    is fine for a headless GPU run). The request is scoped to this thread, so a
    crash of run_local can never leave sleep permanently disabled. No-op on
    non-Windows. Returns True iff a request was issued."""
    if sys.platform != "win32":
        return False
    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if active else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
    return True


@contextlib.contextmanager
def keep_system_awake():
    """Suppress OS idle-sleep for the duration of the wrapped run, then restore
    normal power behavior. Lets a long unattended local run (e.g. the ~6 h
    bv3_ceiling_n10 sweep) finish without the machine sleeping mid-run."""
    active = _set_sleep_prevention(True)
    if active:
        print("keep-awake: Windows sleep suppressed for this run (the display "
              "may still turn off).", flush=True)
    try:
        yield
    finally:
        if active:
            _set_sleep_prevention(False)


def build_cmd(profile_name: str, run_dir: Path, *, resume: bool) -> list[str]:
    """Launch run_e2e.py with the named profile; run_e2e maps it to flags."""
    if profile_name not in PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}")
    cmd = [sys.executable, str(SCRIPTS / "run_e2e.py"), "--profile", profile_name]
    if resume:
        cmd += ["--resume", str(run_dir)]
    else:
        cmd += ["--results-dir", str(run_dir)]
    return cmd


PROFILE_FILE = "profile.txt"  # written by run_e2e.py; legacy name below
LEGACY_PROFILE_FILE = "local_profile.txt"


def unfinished_with_cells(run_dir: Path) -> bool:
    """True if run_dir holds expB2 cell checkpoints but never finalized."""
    if not any((run_dir / "artifacts" / "cells").glob("cell_d*_s*.json")):
        return False
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(
            encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return "finished_at_utc" not in manifest


def check_cuda(allow_cpu: bool) -> None:
    import torch

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        return
    if allow_cpu:
        print("WARNING: no CUDA GPU visible to torch; continuing on CPU "
              "(--allow-cpu).", flush=True)
        return
    raise SystemExit("No CUDA GPU visible to torch. Fix the driver/torch "
                     "install, or pass --allow-cpu for a deliberate CPU run.")


def free_ram_gb() -> float | None:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "[int]((Get-CimInstance Win32_OperatingSystem)"
                 ".FreePhysicalMemory/1024)"],
                capture_output=True, text=True, check=True).stdout
            return int("".join(ch for ch in out if ch.isdigit())) / 1024.0
        except Exception:
            return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return None


def check_ram(min_free_gb: float) -> None:
    free = free_ram_gb()
    if free is None:
        print("WARNING: could not determine free RAM; continuing.", flush=True)
        return
    if free < min_free_gb:
        raise SystemExit(
            f"Only {free:.1f} GB system RAM free (< {min_free_gb:.0f} GB "
            "needed; episode buffers live in RAM). Free memory first, e.g. "
            "close other heavy processes.")
    print(f"Free RAM: {free:.1f} GB -- OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a Colab RUN_PROFILE locally with resume support.")
    ap.add_argument("profile", nargs="?", choices=sorted(PROFILES),
                    help="run profile (see --list)")
    ap.add_argument("--resume", nargs="?", const="latest", default=None,
                    metavar="RUN_DIR",
                    help="continue an interrupted run (latest by default, or "
                         "an explicit fullruns/<dir>)")
    ap.add_argument("--list", action="store_true",
                    help="list profiles and exit")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run even without a CUDA GPU")
    ap.add_argument("--min-free-gb", type=float, default=4.0,
                    help="free system RAM required to start (default 4)")
    a = ap.parse_args()

    if a.list:
        for name, p in PROFILES.items():
            bits = [p["run_mode"]]
            if p["only"]:
                bits.append(f"only={p['only']}")
            if p["b2_seeds"] is not None:
                bits.append(f"seeds={len(p['b2_seeds'])}")
            if p["drift_mode"]:
                bits.append(f"drift_mode={p['drift_mode']}")
            if p["sysid_aux"]:
                bits.append("sysid_aux")
            if p["skip_steps"]:
                bits.append(f"skip={','.join(p['skip_steps'])}")
            print(f"{name:18s} {' '.join(bits)}")
        return
    if not a.profile:
        ap.error("profile is required (or use --list)")

    check_cuda(a.allow_cpu)
    check_ram(a.min_free_gb)

    if a.resume is not None:
        if a.resume == "latest":
            run_dir = read_latest_run_dir()
            if run_dir is None:
                raise SystemExit("--resume: no latest-run pointer found; "
                                 "start a fresh run first or pass an explicit "
                                 "run directory: --resume fullruns/<dir>")
        else:
            run_dir = Path(a.resume)
            if not run_dir.is_dir():
                raise SystemExit(f"--resume: run directory not found: {run_dir}")
        run_dir = Path(run_dir)
        recorded = None
        for fname in (PROFILE_FILE, LEGACY_PROFILE_FILE):
            pf = run_dir / fname
            if pf.is_file():
                recorded = pf.read_text(encoding="utf-8").strip()
                break
        if recorded is None:
            print(f"WARNING: no recorded profile in {run_dir}; proceeding.",
                  flush=True)
        elif recorded != a.profile:
            print(f"WARNING: resuming run recorded as profile "
                  f"'{recorded}' with profile '{a.profile}'. Cell mixing "
                  "is gated by the expB2 config fingerprint, but step "
                  "selection (--only/--skip) is not; proceeding.",
                  flush=True)
    else:
        prev = read_latest_run_dir()
        if prev is not None and unfinished_with_cells(Path(prev)):
            raise SystemExit(
                f"An unfinished run with checkpointed cells exists at {prev}.\n"
                f"Continue it:  python scripts/run_local.py {a.profile} --resume\n"
                "Or start fresh anyway by deleting that run directory.")
        run_dir = Path(default_run_dir())
        run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(a.profile, run_dir, resume=a.resume is not None)
    print("Launching: " + " ".join(cmd), flush=True)
    with keep_system_awake():
        rc = subprocess.run(cmd).returncode
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
