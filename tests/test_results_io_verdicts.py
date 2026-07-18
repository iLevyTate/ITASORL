"""Verdict/headline integrity in results_io (no GPU):

- NaN pool_target seeds (too_few_survivors) must not poison the across-seed mean
  or fall through _encodes to a spurious "weak";
- the expB2 log parser must recognize a nan headline row instead of silently
  attributing an earlier (weaker-drift / control) row to the headline keys;
- the expB_full headline sentence must come from the MEASURED drift sweep, not
  from the step merely exiting 0.
"""

import json
import math
from pathlib import Path

import pytest

from itasorl.results_io import (_encodes, _headline, _load_expb2_json,
                                parse_step_metrics)


def test_encodes_nan_and_none_give_no_verdict():
    assert _encodes(None) is None
    assert _encodes(float("nan")) is None
    assert _encodes(0.5) == "no"
    assert _encodes(0.72, threshold=0.65) == "strong"


def test_load_expb2_json_filters_nonfinite_seeds(tmp_path):
    p = tmp_path / "expB2_results.json"
    payload = {"0.45": {"survival": {"pool_target": [0.72, 0.70, float("nan")]}}}
    p.write_text(json.dumps(payload), encoding="utf-8")
    cell = _load_expb2_json(p)["0.45"]["survival"]
    assert cell["pool_target_mean"] == pytest.approx(0.71)
    assert cell["pool_target_n_finite"] == 2
    assert cell["pool_target_n_seeds"] == 3
    assert cell["organism_encodes_world"] == "strong"


def test_expb2_log_parser_matches_nan_headline_row():
    log = (
        "drift_sigma = 0.00\n"
        "  survival   PRIMARY pool target = 0.503+/-0.012\n"
        "drift_sigma = 0.45\n"
        "  survival   PRIMARY pool target = nan+/-nan\n"
    )
    m = parse_step_metrics("expB2", log)
    # the strongest-drift row IS the nan row; falling back to the 0.503 control
    # row would understate/mislabel the run
    assert math.isnan(m["survival_pool_target_mean"])
    assert m["organism_encodes_world"] is None


def _expb_full_headline(tmp_path: Path, sweep: list[dict]) -> str:
    (tmp_path / "steps").mkdir(exist_ok=True)
    (tmp_path / "steps" / "expB_full.json").write_text(
        json.dumps({"drift_sweep": sweep}), encoding="utf-8")
    steps = {"expB_full": {"status": "ok", "metrics": "steps/expB_full.json"}}
    return _headline(steps, tmp_path)


def test_headline_expb_full_null_only_when_measured(tmp_path):
    null_sweep = [
        {"drift": 0.2, "target_mean": 0.51, "organism_encodes_world": "no"},
        {"drift": 0.45, "target_mean": 0.52, "organism_encodes_world": "no"},
    ]
    assert "near chance" in _expb_full_headline(tmp_path, null_sweep)

    positive_sweep = [
        {"drift": 0.2, "target_mean": 0.55, "organism_encodes_world": "no"},
        {"drift": 0.45, "target_mean": 0.93, "organism_encodes_world": "strong"},
    ]
    h = _expb_full_headline(tmp_path, positive_sweep)
    assert "near chance" not in h
    assert "0.93" in h
