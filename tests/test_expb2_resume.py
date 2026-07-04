"""Cell-level checkpoint/resume for run_expB2.py (no training, no GPU)."""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_expB2  # noqa: E402
import run_e2e  # noqa: E402


BASE = dict(updates=300, n_eps=16, max_steps=80, hidden=96, ray_steps=5,
            shaping_coef=1.0, pool_n=110, pool_steps=24, mp_pairs=60,
            mp_prefix=20, mp_branch=24, basal_e=None, n_pellets=None,
            reach=None, dump_states=None, sysid_aux=False, sysid_coef=1.0,
            drift_mode="regime", drifts=[0.0, 0.45], device="cuda")


def make_cell(drift, seed):
    target = 0.55 + 0.03 * seed + 0.01 * drift
    pool = {"target": target, "target_lo": target - 0.07,
            "target_hi": target + 0.07, "target_var": 0.5,
            "target_full": float("nan"), "selectivity": 0.1,
            "selectivity_var": 0.0, "selectivity_full": 0.0, "speed": 0.9,
            "shuffled": 0.5, "anchor_energy": 0.8, "anchor_food": 0.8,
            "ceiling_drag": float("nan")}
    mp = {"target": 0.5, "leakage_clean": True, "leakage_max_dev": 0.0}
    return {"drift": drift, "seed": seed,
            "eng": {"engaged": True, "trained_return": 0.1},
            "xeval": {"0.00": 0.1, "0.45": 0.0},
            "agents": {g: {"pool": dict(pool), "mp": dict(mp)}
                       for g in run_expB2.AG}}


def test_fingerprint_stable_and_ignores_dump_states():
    fp1 = run_expB2.config_fingerprint(dict(BASE))
    fp2 = run_expB2.config_fingerprint(dict(BASE, dump_states="/somewhere/else"))
    assert fp1 == fp2


def test_fingerprint_changes_on_science_knob():
    fp1 = run_expB2.config_fingerprint(dict(BASE))
    for knob, value in [("hidden", 64), ("drift_mode", "ar1"),
                        ("updates", 60), ("device", "cpu"),
                        ("drifts", [0.0])]:
        fp2 = run_expB2.config_fingerprint(dict(BASE, **{knob: value}))
        assert fp1 != fp2, knob


def test_cell_roundtrip_preserves_nan(tmp_path):
    fp = run_expB2.config_fingerprint(dict(BASE))
    cell = make_cell(0.45, 3)
    run_expB2.write_cell_file(tmp_path, fp, "abc1234", cell)
    done = run_expB2.load_cell_files(tmp_path, fp)
    got = done[(0.45, 3)]
    assert got["seed"] == 3
    assert got["agents"]["survival"]["pool"]["target"] == pytest.approx(
        cell["agents"]["survival"]["pool"]["target"])
    assert math.isnan(got["agents"]["survival"]["pool"]["ceiling_drag"])


def test_load_rejects_fingerprint_mismatch(tmp_path):
    fp = run_expB2.config_fingerprint(dict(BASE))
    run_expB2.write_cell_file(tmp_path, fp, "abc1234", make_cell(0.0, 0))
    other = run_expB2.config_fingerprint(dict(BASE, hidden=64))
    with pytest.raises(SystemExit):
        run_expB2.load_cell_files(tmp_path, other)


def test_load_rejects_corrupt_file(tmp_path):
    fp = run_expB2.config_fingerprint(dict(BASE))
    (tmp_path / "cell_d0.00_s0.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_expB2.load_cell_files(tmp_path, fp)


def test_load_missing_dir_returns_empty(tmp_path):
    fp = run_expB2.config_fingerprint(dict(BASE))
    assert run_expB2.load_cell_files(tmp_path / "nope", fp) == {}


def _run_main(tmp_path, monkeypatch, extra=(), cell_fn=None):
    """Run run_expB2.main() with a stubbed run_cell (no training, seconds not hours).
    --quick gives drifts [0.0, 0.45] x seeds [0, 1] = 4 cells.

    NOTE: these tests run the serial (workers=1) path only; with --workers > 1,
    spawned children re-import the module and would get the REAL run_cell
    (hours of training), so never add --workers to these tests."""
    monkeypatch.setattr(run_expB2, "run_cell",
                        cell_fn or (lambda t: make_cell(t["drift"], t["seed"])))
    argv = ["run_expB2.py", "--quick", "--out-dir", str(tmp_path), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    run_expB2.main()


def test_fresh_run_writes_cell_files_and_results(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    cells = sorted(p.name for p in (tmp_path / "cells").glob("*.json"))
    assert cells == ["cell_d0.00_s0.json", "cell_d0.00_s1.json",
                     "cell_d0.45_s0.json", "cell_d0.45_s1.json"]
    assert (tmp_path / "expB2_results.json").is_file()


def test_fresh_run_refuses_stale_cells(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch)  # no --resume


def test_resume_skips_completed_cells(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)

    def boom(task):
        raise AssertionError("run_cell must not be called on full resume")

    _run_main(tmp_path, monkeypatch, extra=["--resume"], cell_fn=boom)


def test_resume_runs_only_missing_cells(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    (tmp_path / "cells" / "cell_d0.45_s1.json").unlink()
    ran = []

    def spy(task):
        ran.append((task["drift"], task["seed"]))
        return make_cell(task["drift"], task["seed"])

    _run_main(tmp_path, monkeypatch, extra=["--resume"], cell_fn=spy)
    assert ran == [(0.45, 1)]


def test_results_are_seed_ordered_after_resume(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    # remove seed 0 so on resume seed 0 completes AFTER resumed seed 1
    (tmp_path / "cells" / "cell_d0.45_s0.json").unlink()
    _run_main(tmp_path, monkeypatch, extra=["--resume"])
    res = json.loads((tmp_path / "expB2_results.json").read_text())
    got = res["0.45"]["survival"]["pool_target"]
    want = [make_cell(0.45, s)["agents"]["survival"]["pool"]["target"]
            for s in (0, 1)]
    assert got == pytest.approx(want)


def test_resume_rejects_different_config(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch,
                  extra=["--resume", "--shaping_coef", "2.0"])


def test_resume_with_more_seeds_runs_only_new_seeds(tmp_path, monkeypatch):
    """The n=10 extension path: resume an n=2 run with --seeds 0 1 2."""
    ran = []

    def spy(task):
        ran.append((task["drift"], task["seed"]))
        return make_cell(task["drift"], task["seed"])

    monkeypatch.setattr(run_expB2, "run_cell", spy)
    base_argv = ["run_expB2.py", "--out-dir", str(tmp_path),
                 "--updates", "1", "--n_eps", "1", "--max_steps", "1"]
    monkeypatch.setattr(sys, "argv", base_argv + ["--seeds", "0", "1"])
    run_expB2.main()
    ran.clear()
    monkeypatch.setattr(sys, "argv",
                        base_argv + ["--seeds", "0", "1", "2", "--resume"])
    run_expB2.main()
    assert ran == [(0.0, 2), (0.45, 2)]
    res = json.loads((tmp_path / "expB2_results.json").read_text())
    assert len(res["0.45"]["survival"]["pool_target"]) == 3


# ---------------------------------------------------------------------------
# Task 3: run_e2e.build_b2_extra plumbing tests
# ---------------------------------------------------------------------------


def _e2e_args(**over):
    defaults = dict(b2_seeds=None, b2_updates=None, b2_hidden=None,
                    b2_dump_states=None, b2_sysid_aux=False, b2_drift_mode=None)
    defaults.update(over)
    import argparse
    return argparse.Namespace(**defaults)


def test_build_b2_extra_emits_resume_only_in_resume_mode():
    assert "--resume" not in run_e2e.build_b2_extra(_e2e_args())
    assert "--resume" in run_e2e.build_b2_extra(_e2e_args(), resume=True)


def test_build_b2_extra_keeps_other_flags_with_resume():
    extra = run_e2e.build_b2_extra(
        _e2e_args(b2_drift_mode="regime", b2_seeds=[0, 1]), resume=True)
    assert extra[-1] == "--resume"
    assert ["--drift-mode", "regime"] == extra[extra.index("--drift-mode"):
                                              extra.index("--drift-mode") + 2]
