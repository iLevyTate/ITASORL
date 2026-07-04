"""
ITASORL - run any Colab notebook profile locally, with resume.

Mirrors the RUN_PROFILE presets in notebooks/colab_gpu.ipynb and launches
scripts/run_e2e.py with the mapped flags plus local preflight checks
(CUDA visible, enough free RAM). expB2 cells checkpoint after every
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
import json
import subprocess
import sys
from pathlib import Path

from itasorl.results_io import default_run_dir, read_latest_run_dir

SCRIPTS = Path(__file__).resolve().parent

# Keep in sync with _PROFILES in notebooks/colab_gpu.ipynb. The notebook cannot
# import repo code before it clones the repo, so the table is duplicated there.
PROFILES = {
    "quick":             dict(run_mode="quick", only=None,    skip_steps=[],        b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=True),
    "full":              dict(run_mode="full",  only=None,    skip_steps=[],        b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=True),
    "bv3_regime":        dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode="regime", sysid_aux=False, dump_states=True),
    "bv3_regime_n10":    dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=list(range(10)), b2_updates=300, drift_mode="regime", sysid_aux=False, dump_states=True),
    "bv2_ceiling":       dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode=None,     sysid_aux=True,  dump_states=True),
    "bv3_ceiling":       dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode="regime", sysid_aux=True,  dump_states=True),
    "b2_only":           dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode=None,     sysid_aux=False, dump_states=True),
    "b2_seed0":          dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=[0],  b2_updates=300,  drift_mode=None,     sysid_aux=False, dump_states=True),
    "experiments_no_b2": dict(run_mode="full",  only=None,    skip_steps=["expB2"], b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=False),
}


def build_cmd(profile: dict, run_dir: Path, *, resume: bool) -> list[str]:
    """Map one PROFILES entry onto a run_e2e.py argv (pure, unit-tested)."""
    cmd = [sys.executable, str(SCRIPTS / "run_e2e.py")]
    if resume:
        cmd += ["--resume", str(run_dir)]
    else:
        cmd += ["--results-dir", str(run_dir)]
    if profile["run_mode"] == "quick":
        cmd += ["--quick"]
    elif profile["run_mode"] != "full":
        raise ValueError(f"unknown run_mode {profile['run_mode']!r}")
    if profile["only"]:
        cmd += ["--only", profile["only"]]
    for step in profile["skip_steps"]:
        cmd += ["--skip", step]
    if profile["b2_seeds"] is not None:
        cmd += ["--b2-seeds", *[str(s) for s in profile["b2_seeds"]]]
    if profile["b2_updates"] is not None:
        cmd += ["--b2-updates", str(profile["b2_updates"])]
    if profile["drift_mode"]:
        cmd += ["--b2-drift-mode", profile["drift_mode"]]
    if profile["sysid_aux"]:
        cmd += ["--b2-sysid-aux"]
    if profile["dump_states"]:
        cmd += ["--b2-dump-states", str(Path(run_dir) / "states")]
    return cmd


PROFILE_FILE = "local_profile.txt"


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
            "stop the ralph loop.")
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
        profile_file = run_dir / PROFILE_FILE
        if profile_file.is_file():
            recorded = profile_file.read_text(encoding="utf-8").strip()
            if recorded != a.profile:
                print(f"WARNING: resuming run recorded as profile "
                      f"'{recorded}' with profile '{a.profile}'. Cell mixing "
                      "is gated by the expB2 config fingerprint, but step "
                      "selection (--only/--skip) is not; proceeding.",
                      flush=True)
        else:
            print(f"WARNING: no recorded profile in {run_dir}; proceeding.",
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
        (run_dir / PROFILE_FILE).write_text(a.profile + "\n", encoding="utf-8")

    cmd = build_cmd(PROFILES[a.profile], run_dir, resume=a.resume is not None)
    print("Launching: " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
