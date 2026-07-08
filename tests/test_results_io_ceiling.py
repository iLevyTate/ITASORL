"""Sysid-aux CEILING runs must be labeled as a capacity reference, not H_B2
evidence, in both the persisted verdict and the rendered SUMMARY (no GPU)."""

import json
from pathlib import Path

from itasorl.results_io import _b2_used_sysid_aux, build_summary


def _write_b2_run(tmp_path: Path, *, sysid: bool) -> tuple[dict, Path]:
    run_dir = tmp_path / "run"
    (run_dir / "steps").mkdir(parents=True)
    metrics = {
        "survival_pool_target_mean": 0.622,
        "survival_pool_target_std": 0.06,
        "organism_encodes_world": "weak",  # what the old pipeline would write
        "agents_by_drift": {
            "0.45": {
                "untrained": {"pool_target_mean": 0.527, "organism_encodes_world": "no"},
                "predictor": {"pool_target_mean": 0.536, "organism_encodes_world": "no"},
                "survival": {"pool_target_mean": 0.622, "organism_encodes_world": "weak"},
            }
        },
    }
    (run_dir / "steps" / "expB2.json").write_text(json.dumps(metrics), encoding="utf-8")
    flags = ["--updates", "300", "--drift-mode", "regime"]
    if sysid:
        flags.append("--sysid-aux")
    (run_dir / "b2_flags.json").write_text(json.dumps(flags), encoding="utf-8")
    manifest = {
        "run_id": "test",
        "quick": False,
        "git_commit": "abc123",
        "total_elapsed_sec": 60.0,
        "steps": {"expB2": {"status": "ok", "metrics": "steps/expB2.json", "elapsed_sec": 60.0}},
    }
    return manifest, run_dir


def test_ceiling_run_headline_and_verdict(tmp_path):
    manifest, run_dir = _write_b2_run(tmp_path, sysid=True)
    summary = build_summary(manifest, run_dir)
    # Headline reframed as a capacity ceiling, not a weak-trace verdict.
    assert "Sysid-aux **CEILING** run" in summary
    assert "NOT H_B2 evidence" in summary
    assert "below the pre-registered encoding bar" not in summary
    # Survival row relabeled; number unchanged.
    assert "`survival`: pool target **0.622** → **ceiling**" in summary
    # Controls untouched.
    assert "`untrained`: pool target **0.527** → **no**" in summary


def test_non_sysid_run_keeps_weak_verdict(tmp_path):
    manifest, run_dir = _write_b2_run(tmp_path, sysid=False)
    summary = build_summary(manifest, run_dir)
    assert "below the pre-registered encoding bar (0.65)." in summary
    assert "CEILING" not in summary
    assert "`survival`: pool target **0.622** → **weak**" in summary


def test_used_sysid_aux_detection(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert _b2_used_sysid_aux(run_dir) is False  # no flags file
    (run_dir / "b2_flags.json").write_text(json.dumps(["--updates", "300"]), encoding="utf-8")
    assert _b2_used_sysid_aux(run_dir) is False
    (run_dir / "b2_flags.json").write_text(
        json.dumps(["--updates", "300", "--sysid-aux"]), encoding="utf-8"
    )
    assert _b2_used_sysid_aux(run_dir) is True
