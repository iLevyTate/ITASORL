"""Mirror fault tolerance and incremental checkpoint sync (no GPU, no Colab)."""

import shutil
from pathlib import Path

import pytest

from itasorl import results_io
from itasorl.results_io import RunRecorder


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "LATEST_RUN_PTR", tmp_path / "LATEST_RUN.txt")
    monkeypatch.setenv("ITASORL_DRIVE_SYNC", str(tmp_path / "mirror"))
    return RunRecorder.create(quick=True, out_dir=tmp_path / "run")


def _break_mirror(tmp_path: Path) -> Path:
    """Replace the mirror directory with a plain file so any write raises OSError."""
    mirror = tmp_path / "mirror"
    shutil.rmtree(mirror)
    mirror.write_text("not a directory", encoding="utf-8")
    return mirror


def test_mirror_failure_never_raises(recorder, tmp_path):
    _break_mirror(tmp_path)
    recorder._write_status(current_step="s", step_status="running", force=True)


def test_mirror_degraded_warns_once_then_recovers(recorder, tmp_path, capsys):
    mirror = _break_mirror(tmp_path)
    recorder._write_status(current_step="s", step_status="running", force=True)
    recorder._write_status(current_step="s", step_status="running", force=True)
    out = capsys.readouterr().out
    assert out.count("Drive mirror unreachable") == 1
    mirror.unlink()
    recorder._write_status(current_step="s", step_status="running", force=True)
    assert "Drive mirror recovered" in capsys.readouterr().out
