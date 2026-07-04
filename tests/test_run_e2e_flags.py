"""The b2 dump-states 'auto' sentinel: recorded raw, resolved per run dir."""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_e2e  # noqa: E402


def _ns(**overrides):
    base = dict(b2_seeds=None, b2_updates=None, b2_hidden=None,
                b2_dump_states=None, b2_sysid_aux=False, b2_drift_mode=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_auto_resolves_under_run_dir(tmp_path):
    extra = ["--seeds", "0", "--dump-states", "auto", "--sysid-aux"]
    out = run_e2e.resolve_dump_states(extra, tmp_path)
    assert out[out.index("--dump-states") + 1] == str(tmp_path / "artifacts" / "states")
    assert extra[extra.index("--dump-states") + 1] == "auto"  # input not mutated


def test_explicit_path_untouched(tmp_path):
    extra = ["--dump-states", str(tmp_path / "elsewhere")]
    assert run_e2e.resolve_dump_states(extra, tmp_path) == extra


def test_no_dump_states_flag_untouched(tmp_path):
    extra = ["--seeds", "0", "1"]
    assert run_e2e.resolve_dump_states(extra, tmp_path) == extra


def test_auto_recorded_raw_and_reresolved_on_resume(tmp_path):
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    extra = run_e2e.resolve_b2_extra(_ns(b2_dump_states="auto"),
                                     resume=False, run_dir=fresh)
    recorded = json.loads((fresh / "b2_flags.json").read_text(encoding="utf-8"))
    assert recorded == ["--dump-states", "auto"]

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    shutil.copy2(fresh / "b2_flags.json", resumed / "b2_flags.json")
    replayed = run_e2e.resolve_b2_extra(_ns(), resume=True, run_dir=resumed)
    assert replayed == ["--dump-states", "auto", "--resume"]
    resolved = run_e2e.resolve_dump_states(replayed, resumed)
    assert resolved[1] == str(resumed / "artifacts" / "states")
    assert extra == ["--dump-states", "auto"]
