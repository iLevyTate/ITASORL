"""Tests for viz/collect.py pure helpers (no agent bundle, no surrogate training)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("viz_collect", ROOT / "viz" / "collect.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def _beats(surv="73%", untr="50%", orac="99%"):
    return {
        "numbers": {
            "oracle_l2": {"display": orac},
            "probe_chance": {"display": untr},
            "probe_survival": {"display": surv},
        },
        "beats": [
            {"id": "observer", "gauge": {"display": orac}},
            {"id": "nocare", "gauge": {"display": untr}},
            {"id": "survival", "gauge": {"display": surv}},
        ],
    }


def _cells(surv=0.726, untr=0.488):
    out = []
    for s in range(3):
        out.append({"agent": "survival", "drift": "0.45", "seed": s,
                    "resid_trace": surv, "target": 0.75})
        out.append({"agent": "untrained", "drift": "0.45", "seed": s,
                    "resid_trace": 0.5, "target": untr})
        out.append({"agent": "survival", "drift": "0.00", "seed": s,
                    "resid_trace": 0.5, "target": 0.5})
    return out


def test_verify_numbers_passes_on_published_values():
    got = vc.verify_numbers(_beats(), _cells(), "oracle AUROC 0.993 etc")
    assert got["survival_resid_trace_pooled"] == 0.726
    assert got["untrained_target_pooled"] == 0.488


def test_verify_numbers_fails_on_drifted_survival():
    with pytest.raises(SystemExit):
        vc.verify_numbers(_beats(), _cells(surv=0.61), "0.993")


def test_verify_numbers_fails_when_untrained_leaves_chance():
    with pytest.raises(SystemExit):
        vc.verify_numbers(_beats(), _cells(untr=0.60), "0.993")


def test_verify_numbers_fails_without_findings_oracle():
    with pytest.raises(SystemExit):
        vc.verify_numbers(_beats(), _cells(), "no oracle number here")


def test_sample_fields_normalized_grid():
    from itasorl.experiment_b2 import _seeds, make_world
    w = make_world(vc.P, 0.0, vc.RAY_STEPS)
    w.reset(_seeds(1234))
    height, wet = vc.sample_fields(w, 8)
    assert height.shape == (64,) and wet.shape == (64,)
    assert height.min() == 0.0 and height.max() == 1.0
    assert set(np.unique(wet)) <= {0.0, 1.0}


def test_in_split_window():
    inside = [[0.5, 0.5, 0.0, 1.0]] * 300
    assert vc.in_split_window(inside)
    outside = [list(p) for p in inside]
    outside[150][0] = 0.9  # wanders out of the split-panel strip mid-beat
    assert not vc.in_split_window(outside)


def test_build_scene_schema_and_coverage():
    r = {"pts": [[0.5, 0.5, 0.0, 0.8]] * 900, "pellets_t": [[[0.1, 0.2]]] * 900}
    scene = vc.build_scene({"source": "collect.py"}, 8, np.zeros(64), np.zeros(64), r, r)
    for key in ("meta", "grid_n", "height", "wet", "pellets", "trajs", "pellets_t", "step_ms"):
        assert key in scene
    json.dumps(scene)  # fully JSON-serializable
    assert len(scene["trajs"]["auth"]) * scene["step_ms"] >= vc.LAST_WORLD_MS


def test_build_scene_rejects_short_traj():
    r = {"pts": [[0.5, 0.5, 0.0, 0.8]] * 100, "pellets_t": [[[0.1, 0.2]]] * 100}
    with pytest.raises(AssertionError):
        vc.build_scene({"source": "collect.py"}, 8, np.zeros(64), np.zeros(64), r, r)
