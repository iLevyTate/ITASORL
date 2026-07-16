"""Unit tests for the pure helpers of run_l3_crossrecipe (no GPU, no pools)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_l3_crossrecipe as rc  # noqa: E402


def test_parse_agent_filename():
    d, s, arm = rc.parse_agent_filename("agent_d0.45_s7_survival.pt")
    assert (d, s, arm) == (0.45, 7, "survival")
    assert rc.parse_agent_filename("agent_d0.00_s0_untrained.pt") == (0.0, 0, "untrained")
    with pytest.raises(ValueError):
        rc.parse_agent_filename("checkpoint_final.pt")


def test_rename_transfer_keys():
    out = rc.rename_transfer_keys({"transfer_target": 0.7, "transfer_lo": 0.6}, "rff")
    assert out == {"transfer_rff_target": 0.7, "transfer_rff_lo": 0.6}


def test_selected_knob_from_gate0_json(tmp_path):
    p = tmp_path / "gate0_rff.json"
    p.write_text(json.dumps({"rows": [], "selected": {"family": "rff", "D": 32}}))
    assert rc.selected_knob(str(p), "rff") == 32
    p2 = tmp_path / "gate0_cd.json"
    p2.write_text(json.dumps({"rows": [], "selected": None}))
    assert rc.selected_knob(str(p2), "cd") is None  # dropped family


def test_integrity_compare():
    a = {"Ha": np.zeros((3, 4, 5)), "Hs": np.ones((3, 4, 5))}
    assert rc.pools_match(a["Ha"], a["Hs"], a["Ha"], a["Hs"])
    assert not rc.pools_match(a["Ha"], a["Hs"], a["Ha"] + 1e-12, a["Hs"])
