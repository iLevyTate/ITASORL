"""The frozen decision rule must be machine-checked, not hand-applied."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_l3_crossrecipe as rc  # noqa: E402


def test_decision_rule_pass():
    assert rc.decision_rule_pass(0.773, 0.569)          # published heldout case
    assert not rc.decision_rule_pass(0.64, 0.5)         # below absolute bar
    assert not rc.decision_rule_pass(0.66, 0.62)        # below floor + 0.05
    assert not rc.decision_rule_pass(0.66, 0.61)        # boundary: 0.66 == 0.61+0.05 not >
    assert rc.decision_rule_pass(0.6601, 0.61)
