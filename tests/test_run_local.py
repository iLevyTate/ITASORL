"""Local launcher: profile names delegate to run_e2e.py --profile (no GPU)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_e2e  # noqa: E402
import run_local  # noqa: E402

RUN_DIR = Path("fullruns") / "01011999"

NB_PATH = ROOT / "notebooks" / "colab_gpu.ipynb"


def test_profiles_come_from_run_e2e():
    assert run_local.PROFILES is run_e2e.PROFILES


def test_notebook_dropdown_matches_profiles():
    """The notebook only carries profile NAMES (a form dropdown); they must
    match run_e2e.PROFILES so every dropdown choice is a valid --profile."""
    import json
    import re
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    src = next(("".join(c["source"]) for c in nb["cells"]
                if "RUN_PROFILE" in "".join(c["source"])
                and c["cell_type"] == "code"), None)
    assert src, "config cell with RUN_PROFILE not found in colab_gpu.ipynb"
    m = re.search(r'RUN_PROFILE\s*=\s*"[^"]*"\s*#\s*@param\s*\[([^\]]*)\]', src)
    assert m, "RUN_PROFILE form dropdown (# @param [...]) missing from config cell"
    dropdown = re.findall(r'"([^"]+)"', m.group(1))
    assert dropdown == list(run_e2e.PROFILES)


def test_notebook_run_cell_uses_profile_flag():
    """The run cell must pass --profile (all flag logic lives in run_e2e.py)."""
    import json
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    joined = "".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")
    assert '"--profile", RUN_PROFILE' in joined
    assert "_PROFILES" not in joined  # no duplicated profile table


@pytest.mark.parametrize("name", sorted(run_e2e.PROFILES))
def test_every_profile_builds_a_command(name):
    cmd = run_local.build_cmd(name, RUN_DIR, resume=False)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("run_e2e.py")
    assert cmd[cmd.index("--profile") + 1] == name
    assert "--results-dir" in cmd and "--resume" not in cmd


def test_resume_swaps_results_dir_for_resume():
    cmd = run_local.build_cmd("b2_only", RUN_DIR, resume=True)
    assert cmd[cmd.index("--resume") + 1] == str(RUN_DIR)
    assert "--results-dir" not in cmd


def test_build_cmd_rejects_unknown_profile():
    with pytest.raises(ValueError):
        run_local.build_cmd("not_a_profile", RUN_DIR, resume=False)


def test_list_prints_every_profile(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_local.py", "--list"])
    run_local.main()
    out = capsys.readouterr().out
    for name in run_e2e.PROFILES:
        assert name in out


def test_profile_required_without_list(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_local.py"])
    with pytest.raises(SystemExit):
        run_local.main()


def test_unknown_profile_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_local.py", "not_a_profile"])
    with pytest.raises(SystemExit):
        run_local.main()


def _stub_launch(monkeypatch, calls):
    class Ret:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return Ret()

    monkeypatch.setattr(run_local, "check_cuda", lambda allow_cpu: None)
    monkeypatch.setattr(run_local, "check_ram", lambda min_free_gb: None)
    monkeypatch.setattr(run_local.subprocess, "run", fake_run)


def test_main_launches_mapped_command(monkeypatch, tmp_path):
    calls = {}
    _stub_launch(monkeypatch, calls)
    monkeypatch.setattr(run_local, "default_run_dir", lambda: tmp_path / "run")
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "b2_seed0"])
    with pytest.raises(SystemExit) as exc:
        run_local.main()
    assert exc.value.code == 0
    cmd = calls["cmd"]
    assert cmd[cmd.index("--profile") + 1] == "b2_seed0"
    assert cmd[cmd.index("--results-dir") + 1] == str(tmp_path / "run")


def test_main_resume_requires_latest_pointer(monkeypatch):
    monkeypatch.setattr(run_local, "check_cuda", lambda allow_cpu: None)
    monkeypatch.setattr(run_local, "check_ram", lambda min_free_gb: None)
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "b2_only", "--resume"])
    with pytest.raises(SystemExit):
        run_local.main()


@pytest.mark.parametrize("profile_file", ["profile.txt", "local_profile.txt"])
def test_main_resume_warns_on_profile_mismatch(monkeypatch, tmp_path, capsys,
                                               profile_file):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / profile_file).write_text("bv3_regime", encoding="utf-8")
    calls = {}
    _stub_launch(monkeypatch, calls)
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: run_dir)
    monkeypatch.setattr(sys, "argv",
                        ["run_local.py", "bv3_regime_n10", "--resume"])
    with pytest.raises(SystemExit) as exc:
        run_local.main()
    assert exc.value.code == 0
    assert "recorded as profile 'bv3_regime'" in capsys.readouterr().out


def _make_unfinished_run(tmp_path):
    run_dir = tmp_path / "run"
    cells = run_dir / "artifacts" / "cells"
    cells.mkdir(parents=True)
    (cells / "cell_d0.00_s0.json").write_text("{}", encoding="utf-8")
    (run_dir / "manifest.json").write_text('{"run_id": "x"}', encoding="utf-8")
    return run_dir


def test_fresh_start_blocked_by_unfinished_run(monkeypatch, tmp_path):
    calls = {}
    _stub_launch(monkeypatch, calls)
    run_dir = _make_unfinished_run(tmp_path)
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: run_dir)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "b2_seed0"])
    with pytest.raises(SystemExit) as exc:
        run_local.main()
    assert "unfinished run" in str(exc.value)
    assert "cmd" not in calls


def test_fresh_start_proceeds_when_last_run_finished(monkeypatch, tmp_path):
    calls = {}
    _stub_launch(monkeypatch, calls)
    run_dir = _make_unfinished_run(tmp_path)
    (run_dir / "manifest.json").write_text(
        '{"run_id": "x", "finished_at_utc": "2026-07-03T00:00:00+00:00"}',
        encoding="utf-8")
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: run_dir)
    monkeypatch.setattr(run_local, "default_run_dir",
                        lambda: tmp_path / "fresh")
    monkeypatch.setattr(sys, "argv", ["run_local.py", "b2_seed0"])
    with pytest.raises(SystemExit) as exc:
        run_local.main()
    assert exc.value.code == 0
    assert "--results-dir" in calls["cmd"]


def test_resume_accepts_explicit_run_dir(monkeypatch, tmp_path):
    calls = {}
    _stub_launch(monkeypatch, calls)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(sys, "argv",
                        ["run_local.py", "b2_seed0", "--resume", str(run_dir)])
    with pytest.raises(SystemExit) as exc:
        run_local.main()
    assert exc.value.code == 0
    assert calls["cmd"][calls["cmd"].index("--resume") + 1] == str(run_dir)


def test_resume_explicit_dir_missing_errors(monkeypatch, tmp_path):
    calls = {}
    _stub_launch(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv",
                        ["run_local.py", "b2_seed0", "--resume",
                         str(tmp_path / "nope")])
    with pytest.raises(SystemExit):
        run_local.main()
    assert "cmd" not in calls


def test_keep_system_awake_noop_off_windows(monkeypatch):
    """Off Windows the context manager must be a clean no-op (never touch the
    Win32 API) - this is the path CI exercises on Linux."""
    monkeypatch.setattr(run_local.sys, "platform", "linux")
    with run_local.keep_system_awake():
        pass  # must not raise


def test_keep_system_awake_sets_and_clears_on_windows(monkeypatch):
    """On Windows: request ES_CONTINUOUS|ES_SYSTEM_REQUIRED on enter and reset to
    ES_CONTINUOUS on exit, so sleep is suppressed only for the run's duration."""
    class _Kernel32:
        def __init__(self):
            self.states = []

        def SetThreadExecutionState(self, state):
            self.states.append(state.value)   # ctypes.c_uint -> int
            return 0

    class _WinDLL:
        kernel32 = _Kernel32()

    win = _WinDLL()
    monkeypatch.setattr(run_local.sys, "platform", "win32")
    monkeypatch.setattr(run_local.ctypes, "windll", win, raising=False)
    with run_local.keep_system_awake():
        assert win.kernel32.states == [run_local._ES_CONTINUOUS
                                       | run_local._ES_SYSTEM_REQUIRED]
    assert win.kernel32.states == [run_local._ES_CONTINUOUS
                                   | run_local._ES_SYSTEM_REQUIRED,
                                   run_local._ES_CONTINUOUS]
