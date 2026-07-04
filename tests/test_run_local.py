"""Profile-to-argv mapping for the local launcher (pure functions, no GPU)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_local  # noqa: E402

RUN_DIR = Path("fullruns") / "01011999"

NOTEBOOK_PROFILES = {"quick", "full", "bv3_regime", "bv3_regime_n10",
                     "bv2_ceiling", "bv3_ceiling", "b2_only", "b2_seed0",
                     "experiments_no_b2"}


def test_profiles_match_notebook_table():
    assert set(run_local.PROFILES) == NOTEBOOK_PROFILES


@pytest.mark.parametrize("name", sorted(NOTEBOOK_PROFILES))
def test_every_profile_builds_a_command(name):
    cmd = run_local.build_cmd(run_local.PROFILES[name], RUN_DIR, resume=False)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("run_e2e.py")
    assert "--results-dir" in cmd and "--resume" not in cmd


def test_bv3_regime_n10_flags():
    cmd = run_local.build_cmd(run_local.PROFILES["bv3_regime_n10"], RUN_DIR,
                              resume=False)
    i = cmd.index("--b2-seeds")
    assert cmd[i + 1:i + 11] == [str(s) for s in range(10)]
    assert cmd[cmd.index("--b2-drift-mode") + 1] == "regime"
    assert cmd[cmd.index("--b2-updates") + 1] == "300"
    assert cmd[cmd.index("--b2-dump-states") + 1] == str(RUN_DIR / "states")
    assert cmd[cmd.index("--only") + 1] == "expb2"


def test_resume_swaps_results_dir_for_resume():
    cmd = run_local.build_cmd(run_local.PROFILES["b2_only"], RUN_DIR, resume=True)
    assert cmd[cmd.index("--resume") + 1] == str(RUN_DIR)
    assert "--results-dir" not in cmd


def test_quick_profile_uses_quick_flag():
    cmd = run_local.build_cmd(run_local.PROFILES["quick"], RUN_DIR, resume=False)
    assert "--quick" in cmd and "--only" not in cmd


def test_experiments_no_b2_skips_and_dumps_nothing():
    cmd = run_local.build_cmd(run_local.PROFILES["experiments_no_b2"], RUN_DIR,
                              resume=False)
    assert cmd[cmd.index("--skip") + 1] == "expB2"
    assert "--b2-dump-states" not in cmd


def test_ceiling_profiles_set_sysid_aux():
    for name in ("bv2_ceiling", "bv3_ceiling"):
        cmd = run_local.build_cmd(run_local.PROFILES[name], RUN_DIR, resume=False)
        assert "--b2-sysid-aux" in cmd, name
