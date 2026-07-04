"""Cell-level checkpoint/resume for run_expB2.py (no training, no GPU)."""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_expB2  # noqa: E402


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
