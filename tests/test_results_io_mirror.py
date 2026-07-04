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


def _add_cell_file(recorder, name: str = "cell_d0.00_s0.json") -> Path:
    cells = recorder.run_dir / "artifacts" / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    path = cells / name
    path.write_text("{}", encoding="utf-8")
    return path


def test_ckpt_sync_copies_new_artifact_files(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "0")
    _add_cell_file(recorder)
    recorder._sync_ckpt_mirror()
    mirrored = (tmp_path / "mirror" / recorder.run_dir.name
                / "artifacts" / "cells" / "cell_d0.00_s0.json")
    assert mirrored.is_file()


def test_ckpt_sync_skips_unchanged_files(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "0")
    _add_cell_file(recorder)
    recorder._sync_ckpt_mirror()
    mirrored = (tmp_path / "mirror" / recorder.run_dir.name
                / "artifacts" / "cells" / "cell_d0.00_s0.json")
    mirrored.write_text("sentinel", encoding="utf-8")  # newer mtime than source
    recorder._sync_ckpt_mirror()
    assert mirrored.read_text(encoding="utf-8") == "sentinel"


def test_ckpt_sync_honors_interval(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "3600")
    _add_cell_file(recorder)
    recorder._sync_ckpt_mirror()  # _ckpt_last_sync was set at create(); 0 s elapsed
    assert not (tmp_path / "mirror" / recorder.run_dir.name / "artifacts").exists()


def test_ckpt_sync_runs_via_write_status(recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "0")
    _add_cell_file(recorder, "cell_d0.45_s3.json")
    recorder._write_status(current_step="expB2", step_status="running", force=True)
    mirrored = (tmp_path / "mirror" / recorder.run_dir.name
                / "artifacts" / "cells" / "cell_d0.45_s3.json")
    assert mirrored.is_file()


def test_local_status_written_even_when_mirror_broken(recorder, tmp_path):
    _break_mirror(tmp_path)
    recorder._write_status(current_step="s", step_status="running", force=True)
    assert (recorder.run_dir / "status.json").is_file()


def test_finalize_mirrors_summary_and_bundle(recorder, tmp_path):
    recorder.finalize(total_sec=1.0, make_zip=True)
    dest = tmp_path / "mirror" / recorder.run_dir.name
    assert (dest / "SUMMARY.md").is_file()
    assert (dest / "bundle.zip").is_file()


def test_ckpt_sync_recopies_updated_source(recorder, tmp_path, monkeypatch):
    import os as _os
    monkeypatch.setenv("ITASORL_CKPT_SYNC_SEC", "0")
    src = _add_cell_file(recorder)
    recorder._sync_ckpt_mirror()
    src.write_text('{"v": 2}', encoding="utf-8")
    t = src.stat().st_mtime + 10
    _os.utime(src, (t, t))
    recorder._sync_ckpt_mirror()
    mirrored = (tmp_path / "mirror" / recorder.run_dir.name
                / "artifacts" / "cells" / "cell_d0.00_s0.json")
    assert mirrored.read_text(encoding="utf-8") == '{"v": 2}'
