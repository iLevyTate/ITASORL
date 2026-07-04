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
    assert cmd[cmd.index("--b2-updates") + 1] == "300"  # profile flags survive resume


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


def test_list_prints_every_profile(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_local.py", "--list"])
    run_local.main()
    out = capsys.readouterr().out
    for name in NOTEBOOK_PROFILES:
        assert name in out


def test_profile_required_without_list(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_local.py"])
    with pytest.raises(SystemExit):
        run_local.main()


def test_unknown_profile_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_local.py", "not_a_profile"])
    with pytest.raises(SystemExit):
        run_local.main()


def test_main_launches_mapped_command(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(run_local, "check_cuda", lambda allow_cpu: None)
    monkeypatch.setattr(run_local, "check_ram", lambda min_free_gb: None)
    monkeypatch.setattr(run_local, "default_run_dir", lambda: tmp_path / "run")
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: None)

    class Ret:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return Ret()

    monkeypatch.setattr(run_local.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "b2_seed0"])
    with pytest.raises(SystemExit) as exc:
        run_local.main()
    assert exc.value.code == 0
    cmd = calls["cmd"]
    assert cmd[cmd.index("--b2-seeds") + 1] == "0"
    assert (tmp_path / "run" / "local_profile.txt").read_text(
        encoding="utf-8").strip() == "b2_seed0"


def test_main_resume_requires_latest_pointer(monkeypatch):
    monkeypatch.setattr(run_local, "check_cuda", lambda allow_cpu: None)
    monkeypatch.setattr(run_local, "check_ram", lambda min_free_gb: None)
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_local.py", "b2_only", "--resume"])
    with pytest.raises(SystemExit):
        run_local.main()


def test_main_resume_warns_on_profile_mismatch(monkeypatch, tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "local_profile.txt").write_text("bv3_regime", encoding="utf-8")
    monkeypatch.setattr(run_local, "check_cuda", lambda allow_cpu: None)
    monkeypatch.setattr(run_local, "check_ram", lambda min_free_gb: None)
    monkeypatch.setattr(run_local, "read_latest_run_dir", lambda: run_dir)

    class Ret:
        returncode = 0

    monkeypatch.setattr(run_local.subprocess, "run", lambda cmd, **kw: Ret())
    monkeypatch.setattr(sys, "argv",
                        ["run_local.py", "bv3_regime_n10", "--resume"])
    with pytest.raises(SystemExit) as exc:
        run_local.main()
    assert exc.value.code == 0
    assert "recorded as profile 'bv3_regime'" in capsys.readouterr().out


def test_build_cmd_rejects_unknown_run_mode():
    with pytest.raises(ValueError):
        run_local.build_cmd(dict(run_local.PROFILES["full"], run_mode="ful"),
                            RUN_DIR, resume=False)


def _stub_launch(monkeypatch, calls):
    class Ret:
        returncode = 0

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return Ret()

    monkeypatch.setattr(run_local, "check_cuda", lambda allow_cpu: None)
    monkeypatch.setattr(run_local, "check_ram", lambda min_free_gb: None)
    monkeypatch.setattr(run_local.subprocess, "run", fake_run)


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


def test_profiles_values_match_notebook_table():
    import json
    nb = json.loads((ROOT / "notebooks" / "colab_gpu.ipynb").read_text(
        encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"]
               if "_PROFILES" in "".join(c["source"]))
    block = src[src.index("_PROFILES = {"):]
    block = block[:block.index("\n}") + 2]
    ns = {}
    exec(block, {"dict": dict, "list": list, "range": range}, ns)
    assert ns["_PROFILES"] == run_local.PROFILES
