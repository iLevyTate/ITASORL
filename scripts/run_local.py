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

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# Keep in sync with _PROFILES in notebooks/colab_gpu.ipynb. The notebook cannot
# import repo code before it clones the repo, so the table is duplicated there.
PROFILES = {
    "quick":             dict(run_mode="quick", only=None,    skip_steps=[],        b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=True),   # noqa: E501
    "full":              dict(run_mode="full",  only=None,    skip_steps=[],        b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=True),   # noqa: E501
    "bv3_regime":        dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode="regime", sysid_aux=False, dump_states=True),   # noqa: E501
    "bv3_regime_n10":    dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=list(range(10)), b2_updates=300, drift_mode="regime", sysid_aux=False, dump_states=True),  # noqa: E501
    "bv2_ceiling":       dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode=None,     sysid_aux=True,  dump_states=True),   # noqa: E501
    "bv3_ceiling":       dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode="regime", sysid_aux=True,  dump_states=True),   # noqa: E501
    "b2_only":           dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode=None,     sysid_aux=False, dump_states=True),   # noqa: E501
    "b2_seed0":          dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=[0],  b2_updates=300,  drift_mode=None,     sysid_aux=False, dump_states=True),   # noqa: E501
    "experiments_no_b2": dict(run_mode="full",  only=None,    skip_steps=["expB2"], b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=False),  # noqa: E501
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
