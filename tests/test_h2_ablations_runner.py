"""Unit tests for the pure helpers of run_l3_h2_ablations (no GPU, no pools)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_l3_h2_ablations as rh  # noqa: E402


def test_parse_agent_filename():
    d, s, arm = rh.parse_agent_filename("agent_d0.45_s7_survival.pt")
    assert (d, s, arm) == (0.45, 7, "survival")
    with pytest.raises(ValueError):
        rh.parse_agent_filename("checkpoint_final.pt")


def test_rename_transfer_keys():
    out = rh.rename_transfer_keys({"transfer_target": 0.7, "transfer_lo": 0.6}, "gn")
    assert out == {"transfer_gn_target": 0.7, "transfer_gn_lo": 0.6}
    out2 = rh.rename_transfer_keys({"transfer_target": 0.5}, "h16")
    assert out2 == {"transfer_h16_target": 0.5}


def test_selected_gn_sigma(tmp_path):
    p = tmp_path / "gate0_gn.json"
    p.write_text(json.dumps({"rows": [], "selected": {"family": "gn", "sigma_v": 0.01}}))
    assert rh.selected_gn_sigma(str(p)) == 0.01
    p2 = tmp_path / "gate0_gn_drop.json"
    p2.write_text(json.dumps({"rows": [], "selected": None}))
    assert rh.selected_gn_sigma(str(p2)) is None


def test_ladder_capacities_ignores_band_requires_floor_and_leak(tmp_path):
    p = tmp_path / "gate0_ladder.json"
    p.write_text(json.dumps({
        "rows": [
            {"hidden": 8, "oracle_auroc": 0.928, "mech_leak_pass": True, "floor_ok": True},
            {"hidden": 16, "oracle_auroc": 0.72, "mech_leak_pass": True, "floor_ok": True},
            {"hidden": 32, "oracle_auroc": 0.70, "mech_leak_pass": True, "floor_ok": False},
            {"hidden": 64, "oracle_auroc": 0.63, "mech_leak_pass": False, "floor_ok": True},
        ]
    }))
    assert rh.ladder_capacities(str(p)) == [(16, 0.72)]


def test_gn_verdict_table():
    assert rh.gn_verdict(0.55, 0.54) == "H2_SUPPORTED"          # fails both
    assert rh.gn_verdict(0.60, 0.50) == "PARTIAL"                # sub-bar, clears floor
    assert rh.gn_verdict(0.70, 0.50) == "H2_NEGATIVE"            # passes both
    assert rh.gn_verdict(0.66, 0.63) == "UNINFORMATIVE"          # bar but floor>=0.60 close


def test_ladder_promotion_pass():
    assert rh.ladder_promotion_pass(0.72, 0.55) is True
    assert rh.ladder_promotion_pass(0.60, 0.58) is False   # h16 not > h64+0.05
    assert rh.ladder_promotion_pass(0.80, 0.70) is False   # h64 not below bar


def test_integrity_compare():
    a = {"Ha": np.zeros((3, 4, 5)), "Hs": np.ones((3, 4, 5))}
    assert rh.pools_match(a["Ha"], a["Hs"], a["Ha"], a["Hs"])
    assert not rh.pools_match(a["Ha"], a["Hs"], a["Ha"] + 1e-12, a["Hs"])
