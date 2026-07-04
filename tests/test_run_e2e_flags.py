"""The b2 dump-states 'auto' sentinel: recorded raw, resolved per run dir.

Also covers the Colab notebook auto-resume matching contract: what
``build_b2_extra`` produces from a synthetic Namespace built from a profile
dict must equal what ``resolve_b2_extra`` records in ``b2_flags.json`` for a
fresh run of that profile. The notebook uses the first form; the resumed run
compares against the second. Divergence would silently break auto-resume.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_e2e  # noqa: E402
import run_local  # noqa: E402


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


def _notebook_ns_from_profile(profile: dict) -> argparse.Namespace:
    """Mirror _expected_b2_flags() in notebooks/colab_gpu.ipynb (cell 23887898).
    Keep in sync with that function; a divergence here breaks Colab auto-resume."""
    return argparse.Namespace(
        b2_seeds=profile["b2_seeds"],
        b2_updates=profile["b2_updates"],
        b2_hidden=None,
        b2_dump_states=("auto" if profile["dump_states"] else None),
        b2_sysid_aux=profile["sysid_aux"],
        b2_drift_mode=profile["drift_mode"],
    )


@pytest.mark.parametrize("name", sorted(run_local.PROFILES))
def test_notebook_flags_match_fresh_run_record(name, tmp_path):
    """The auto-resume guard in notebooks/colab_gpu.ipynb compares
    build_b2_extra(nb_ns) against the b2_flags.json a fresh run of the same
    profile wrote. They must be equal, or every real Colab resume hits the
    mismatch SystemExit."""
    profile = run_local.PROFILES[name]
    nb_ns = _notebook_ns_from_profile(profile)
    expected = run_e2e.build_b2_extra(nb_ns)

    fresh = tmp_path / name
    fresh.mkdir()
    run_e2e.resolve_b2_extra(nb_ns, resume=False, run_dir=fresh)
    recorded = json.loads((fresh / "b2_flags.json").read_text(encoding="utf-8"))
    assert recorded == expected
