"""run_e2e profile presets, the b2 dump-states 'auto' sentinel, and auto-resume.

The profile table lives in run_e2e.PROFILES (single source of truth); the
Colab notebook only passes --profile <name>, so there is no notebook/scripts
flag contract to keep in lockstep anymore. What matters now:

- apply_profile maps a profile onto args without clobbering explicit flags,
- a fresh run records profile.txt + b2_flags.json,
- find_auto_resume continues an unfinished run of the SAME profile (local
  first, then the Drive mirror) and leaves other profiles alone.
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


def _ns(**overrides):
    base = dict(profile=None, drive_sync=None, force_fresh=False, quick=False,
                skip=[], only=None, results_dir=None, resume=None, no_zip=False,
                b2_seeds=None, b2_updates=None, b2_hidden=None,
                b2_dump_states=None, b2_sysid_aux=False, b2_drift_mode=None)
    base.update(overrides)
    return argparse.Namespace(**base)


# --- dump-states sentinel ---------------------------------------------------

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


# --- profiles ----------------------------------------------------------------

def test_profile_table_has_expected_entries():
    assert set(run_e2e.PROFILES) == {
        "quick", "full", "bv3_regime", "bv3_regime_n10", "bv2_ceiling",
        "bv3_ceiling", "bv3_ceiling_n10", "b2_only", "b2_seed0", "experiments_no_b2"}


@pytest.mark.parametrize("name", sorted(run_e2e.PROFILES))
def test_apply_profile_fills_args(name):
    args = _ns(profile=name)
    run_e2e.apply_profile(args)
    p = run_e2e.PROFILES[name]
    assert args.quick == (p["run_mode"] == "quick")
    assert args.only == p["only"]
    assert args.skip == p["skip_steps"]
    assert args.b2_seeds == p["b2_seeds"]
    assert args.b2_updates == p["b2_updates"]
    assert args.b2_drift_mode == p["drift_mode"]
    assert args.b2_sysid_aux == p["sysid_aux"]
    assert args.b2_dump_states == ("auto" if p["dump_states"] else None)


def test_apply_profile_keeps_explicit_flags():
    args = _ns(profile="bv3_regime", b2_updates=50, b2_seeds=[7])
    run_e2e.apply_profile(args)
    assert args.b2_updates == 50
    assert args.b2_seeds == [7]
    assert args.b2_drift_mode == "regime"  # unset fields still filled


def test_bv3_regime_n10_flags():
    args = _ns(profile="bv3_regime_n10")
    run_e2e.apply_profile(args)
    extra = run_e2e.build_b2_extra(args)
    i = extra.index("--seeds")
    assert extra[i + 1:i + 11] == [str(s) for s in range(10)]
    assert extra[extra.index("--updates") + 1] == "300"
    assert extra[extra.index("--drift-mode") + 1] == "regime"
    assert args.only == "expb2"


# --- auto-resume -------------------------------------------------------------

def _make_run(root: Path, name: str, *, profile: str | None = None,
              flags: list[str] | None = None, quick: bool = False,
              finished: bool = False, started: str = "2026-07-01T00:00:00+00:00") -> Path:
    run = root / name
    run.mkdir(parents=True)
    manifest = {"run_id": name, "started_at_utc": started, "quick": quick}
    if finished:
        manifest["finished_at_utc"] = "2026-07-02T00:00:00+00:00"
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if profile is not None:
        (run / "profile.txt").write_text(profile + "\n", encoding="utf-8")
    if flags is not None:
        (run / "b2_flags.json").write_text(json.dumps(flags), encoding="utf-8")
    return run


def test_run_is_unfinished(tmp_path):
    unfinished = _make_run(tmp_path, "a")
    finished = _make_run(tmp_path, "b", finished=True)
    empty = tmp_path / "c"
    empty.mkdir()
    assert run_e2e.run_is_unfinished(unfinished)
    assert not run_e2e.run_is_unfinished(finished)
    assert not run_e2e.run_is_unfinished(empty)


def test_run_matches_profile_by_name(tmp_path):
    run = _make_run(tmp_path, "a", profile="bv3_regime")
    assert run_e2e.run_matches_profile(run, _ns(profile="bv3_regime"))
    assert not run_e2e.run_matches_profile(run, _ns(profile="b2_only"))


def test_run_matches_profile_legacy_flags(tmp_path):
    """Runs from before profile.txt existed match on recorded b2 flags + quick."""
    args = _ns(profile="bv3_regime")
    run_e2e.apply_profile(args)
    run = _make_run(tmp_path, "a", flags=run_e2e.build_b2_extra(args))
    assert run_e2e.run_matches_profile(run, args)

    other = _ns(profile="b2_only")
    run_e2e.apply_profile(other)
    assert not run_e2e.run_matches_profile(run, other)


def test_find_auto_resume_prefers_local(tmp_path, monkeypatch):
    local = _make_run(tmp_path / "fullruns", "local", profile="b2_only")
    monkeypatch.setattr(run_e2e, "read_latest_run_dir", lambda: local)
    args = _ns(profile="b2_only")
    run_e2e.apply_profile(args)
    assert run_e2e.find_auto_resume(args, None) == local


def test_find_auto_resume_copies_drive_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_e2e, "read_latest_run_dir", lambda: None)
    monkeypatch.setattr(run_e2e, "FULLRUNS_ROOT", tmp_path / "fullruns")
    sync = tmp_path / "drive"
    _make_run(sync, "old_other", profile="bv3_regime",
              started="2026-06-01T00:00:00+00:00")
    _make_run(sync, "mine", profile="b2_only",
              started="2026-06-05T00:00:00+00:00")
    args = _ns(profile="b2_only")
    run_e2e.apply_profile(args)
    got = run_e2e.find_auto_resume(args, sync)
    assert got == tmp_path / "fullruns" / "_resume_local" / "mine"
    assert (got / "manifest.json").is_file()  # copied local
    assert (got / "profile.txt").read_text(encoding="utf-8").strip() == "b2_only"


def test_find_auto_resume_ignores_finished_and_mismatched(tmp_path, monkeypatch):
    monkeypatch.setattr(run_e2e, "read_latest_run_dir", lambda: None)
    monkeypatch.setattr(run_e2e, "FULLRUNS_ROOT", tmp_path / "fullruns")
    sync = tmp_path / "drive"
    _make_run(sync, "done", profile="b2_only", finished=True)
    _make_run(sync, "other_profile", profile="bv3_regime")
    args = _ns(profile="b2_only")
    run_e2e.apply_profile(args)
    assert run_e2e.find_auto_resume(args, sync) is None


def test_find_auto_resume_no_sync_root(monkeypatch, tmp_path):
    monkeypatch.setattr(run_e2e, "read_latest_run_dir", lambda: None)
    args = _ns(profile="b2_only")
    run_e2e.apply_profile(args)
    assert run_e2e.find_auto_resume(args, None) is None
    assert run_e2e.find_auto_resume(args, tmp_path / "nope") is None


def test_expand_skip_returns_canonical_step_names():
    """Regression: expand_skip lowercased names but the run loop compares against
    the mixed-case step names from experiment_steps(), so --skip expB2 (and the
    experiments_no_b2 profile) silently skipped nothing."""
    step_names = {s for s, _, _ in run_e2e.experiment_steps(quick=False, b2_out=Path("."))}
    assert "expB2" in run_e2e.expand_skip(["expB2"])
    assert "expB2" in run_e2e.expand_skip(["EXPB2"])  # case-insensitive in
    # aliases expand to canonical names the loop can actually match
    assert run_e2e.expand_skip(["expA"]) == {"expA_l1", "expA_l2"}
    assert run_e2e.expand_skip(["expb"]) <= step_names
    assert run_e2e.expand_skip(["experiments"]) == step_names
    # the experiments_no_b2 profile's whole purpose
    prof = run_e2e.PROFILES["experiments_no_b2"]
    assert "expB2" in run_e2e.expand_skip(prof["skip_steps"])
