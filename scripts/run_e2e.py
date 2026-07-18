"""
ITASORL end-to-end runner: pytest + all reproduction scripts in FINDINGS order.

Records every step under fullruns/<MMDDYYYY>/ by default (logs, metrics JSON,
figures, SUMMARY.md, bundle.zip). Override with --results-dir.

Live output while running: combined.log and status.json are updated incrementally.
In a second terminal: python scripts/watch_run.py --follow

Usage (from repo root):
    python scripts/run_e2e.py              # full suite + recorded results
    python scripts/run_e2e.py --profile bv3_regime          # named preset
    python scripts/run_e2e.py --profile bv3_regime_n10 --drive-sync /content/drive/MyDrive/ITASORL_results
    python scripts/run_e2e.py --quick      # all experiments; B-v2 at reduced scale
    python scripts/run_e2e.py --results-dir results/runs/my_run
    python scripts/run_e2e.py --resume     # continue latest interrupted run (recorded --b2-* flags replay automatically)
    python scripts/run_e2e.py --resume fullruns/06292026_143022
    python scripts/run_e2e.py --only expb2 --b2-drift-mode regime --b2-dump-states runs/bv3
    python scripts/run_e2e.py --no-zip     # skip zip bundle

With --profile, an interrupted run of the same profile is found and continued
automatically (checking the local latest-run pointer first, then --drive-sync).
Pass --force-fresh to ignore unfinished runs and start over.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from itasorl.results_io import (  # noqa: E402
    FULLRUNS_ROOT,
    LATEST_RUN_PTR,
    RunRecorder,
    read_latest_run_dir,
)

# Named presets for the Colab notebook and scripts/run_local.py. This table is
# the single source of truth; the notebook only carries the profile *names* in
# its dropdown and passes --profile, so it never has to stay in lockstep with
# the flag plumbing here.
PROFILES = {
    "quick":             dict(run_mode="quick", only=None,    skip_steps=[],        b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=True),
    "full":              dict(run_mode="full",  only=None,    skip_steps=[],        b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=True),
    "bv3_regime":        dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode="regime", sysid_aux=False, dump_states=True),
    "bv3_regime_n10":    dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=list(range(10)), b2_updates=300, drift_mode="regime", sysid_aux=False, dump_states=True),
    "bv2_ceiling":       dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode=None,     sysid_aux=True,  dump_states=True),
    "bv3_ceiling":       dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode="regime", sysid_aux=True,  dump_states=True),
    "bv3_ceiling_n10":   dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=list(range(10)), b2_updates=300, drift_mode="regime", sysid_aux=True,  dump_states=True),
    "b2_only":           dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=None, b2_updates=300,  drift_mode=None,     sysid_aux=False, dump_states=True),
    "b2_seed0":          dict(run_mode="full",  only="expb2", skip_steps=[],        b2_seeds=[0],  b2_updates=300,  drift_mode=None,     sysid_aux=False, dump_states=True),
    "experiments_no_b2": dict(run_mode="full",  only=None,    skip_steps=["expB2"], b2_seeds=None, b2_updates=None, drift_mode=None,     sysid_aux=False, dump_states=False),
}


def _script(name: str) -> str:
    return str(SCRIPTS / name)


def experiment_steps(*, quick: bool, b2_out: Path,
                     b2_extra: list[str] | None = None) -> list[tuple[str, list[str], list[str] | None]]:
    py = sys.executable
    b2 = [py, _script("run_expB2.py"), "--out-dir", str(b2_out)]
    if quick:
        b2.append("--quick")
    if b2_extra:
        b2.extend(b2_extra)
    return [
        ("expA_l1", [py, _script("run_expA.py")], None),
        ("expA_l2", [py, _script("run_expA_l2.py")], None),
        ("expB_smoke", [py, "-m", "itasorl.experiment_b"], None),
        ("expB_full", [py, _script("run_expB_full.py")], None),
        ("expB_surprise", [py, _script("run_expB_surprise.py")], None),
        ("expB_kstep", [py, _script("run_expB_kstep.py")], None),
        ("expB_gap", [py, _script("run_expB_gap.py")], None),
        ("expB_nonlinear", [py, _script("run_expB_nonlinear.py")], None),
        ("expB2", b2, None),
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ITASORL pytest + all reproduction scripts.")
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None,
                    help="Named preset (sets --quick/--only/--skip/--b2-* for you). "
                         "Explicit flags still win. Enables auto-resume of an "
                         "unfinished run of the same profile.")
    ap.add_argument("--drive-sync", type=Path, default=None,
                    help="Mirror live logs + checkpoints to this directory (e.g. "
                         "Google Drive on Colab) and scan it for unfinished runs "
                         "to auto-resume. Sets ITASORL_DRIVE_SYNC.")
    ap.add_argument("--force-fresh", action="store_true",
                    help="With --profile: ignore any unfinished run and start over.")
    ap.add_argument("--quick", action="store_true",
                    help="Run all experiments; B-v2 uses --quick (tiny scale).")
    ap.add_argument("--skip", action="append", default=[],
                    help="Skip steps: pytest, expA_l1, expA_l2, expB_smoke, expB_full, "
                         "expB_surprise, expB_kstep, expB_gap, expB_nonlinear, expB2, "
                         "or aliases: pytest, experiments, expA, expB.")
    ap.add_argument("--only", choices=("pytest", "experiments", "expb2"),
                    help="Run only pytest, all experiment scripts, or only expB2.")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Directory for this run's recorded output (default: fullruns/<MMDDYYYY>/).")
    ap.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        help="Continue an interrupted run. Optional PATH, or omit value to use results/LATEST_RUN.txt.",
    )
    ap.add_argument("--no-zip", action="store_true", help="Do not create bundle.zip.")
    ap.add_argument("--b2-seeds", type=int, nargs="+", default=None,
                    help="Override Experiment B-v2 seeds (e.g. 0 1 2 ... 15 to power the null).")
    ap.add_argument("--b2-updates", type=int, default=None, help="Override B-v2 training updates.")
    ap.add_argument("--b2-hidden", type=int, default=None, help="Override B-v2 recurrent hidden size.")
    ap.add_argument("--b2-dump-states", type=str, default=None,
                    help="Persist B-v2 recurrent states to this dir (forwarded to run_expB2.py "
                         "--dump-states) for offline variance/selectivity re-probing with "
                         "scripts/reanalyze_expB2_states.py. Pass 'auto' to place them under "
                         "<run_dir>/artifacts/states (mirrored, bundled, resume-safe).")
    ap.add_argument("--b2-sysid-aux", action="store_true",
                    help="Run B-v2 with the system-ID CEILING control (forwards --sysid-aux). "
                         "Breaks readout-not-reward; report separately from the headline.")
    ap.add_argument("--b2-drift-mode", choices=("ar1", "regime"), default=None,
                    help="B-v2/B-v3 surrogate coupling (forwards --drift-mode): ar1 volatility "
                         "vs regime per-episode constant offset.")
    return ap.parse_args()


def build_b2_extra(args: argparse.Namespace, *, resume: bool = False) -> list[str]:
    extra: list[str] = []
    if args.b2_seeds is not None:
        extra += ["--seeds", *[str(s) for s in args.b2_seeds]]
    if args.b2_updates is not None:
        extra += ["--updates", str(args.b2_updates)]
    if args.b2_hidden is not None:
        extra += ["--hidden", str(args.b2_hidden)]
    if args.b2_dump_states is not None:
        extra += ["--dump-states", args.b2_dump_states]
    if args.b2_sysid_aux:
        extra += ["--sysid-aux"]
    if args.b2_drift_mode is not None:
        extra += ["--drift-mode", args.b2_drift_mode]
    if resume:
        extra += ["--resume"]
    return extra


B2_FLAGS_FILE = "b2_flags.json"


def resolve_b2_extra(args: argparse.Namespace, *, resume: bool, run_dir: Path) -> list[str]:
    """Persist b2 flags on fresh runs; on resume, replay them unless the user
    passed explicit --b2-* flags. Seeds and dump-states are not covered by the
    expB2 config fingerprint, so a bare --resume without replay would silently
    under-run an interrupted n=10 extension."""
    extra = build_b2_extra(args)
    flags_path = run_dir / B2_FLAGS_FILE
    if not resume:
        flags_path.write_text(json.dumps(extra), encoding="utf-8")
    elif not extra and flags_path.is_file():
        extra = json.loads(flags_path.read_text(encoding="utf-8"))
        if extra:
            print(f"Resume: replaying recorded B-v2 flags: {' '.join(extra)}",
                  flush=True)
    if resume:
        extra = [*extra, "--resume"]
    return extra


DUMP_STATES_AUTO = "auto"


def resolve_dump_states(extra: list[str], run_dir: Path) -> list[str]:
    """Resolve the 'auto' dump-states sentinel against the active run dir.
    b2_flags.json keeps the raw sentinel so a resume on a different path
    (new VM, fullruns/_resume_local copy) re-resolves correctly."""
    out = list(extra)
    for i, tok in enumerate(out[:-1]):
        if tok == "--dump-states" and out[i + 1] == DUMP_STATES_AUTO:
            out[i + 1] = str(run_dir / "artifacts" / "states")
    return out


def expand_skip(raw: list[str]) -> set[str]:
    """Expand --skip names/aliases to the CANONICAL step names the run loop compares
    against. Matching is case-insensitive, but the returned names keep the exact
    casing of experiment_steps() (e.g. "expb2" -> "expB2") - the loop's `name in skip`
    check is case-sensitive."""
    step_names = {s for s, _, _ in experiment_steps(quick=False, b2_out=Path("."))}
    aliases = {
        "pytest": {"pytest"},
        "experiments": set(step_names),
        "expa": {"expA_l1", "expA_l2"},
        "expb": {"expB_smoke", "expB_full", "expB_surprise", "expB_kstep",
                 "expB_gap", "expB_nonlinear"},
    }
    canonical = {s.lower(): s for s in step_names}
    out: set[str] = set()
    for item in raw:
        key = item.strip().lower()
        if key in aliases:
            out |= aliases[key]
        else:
            out.add(canonical.get(key, key))
    return out


def resolve_resume_dir(resume: str | None, results_dir: Path | None) -> Path | None:
    if resume is None:
        return None
    if resume == "latest":
        run_dir = read_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError(
                f"--resume requested but {LATEST_RUN_PTR} does not point at a run directory."
            )
        return run_dir
    path = Path(resume)
    if not path.is_dir():
        raise FileNotFoundError(f"--resume path not found: {path}")
    return path.resolve()


def apply_profile(args: argparse.Namespace) -> None:
    """Fill unset args from the named profile; explicit flags win."""
    p = PROFILES[args.profile]
    if p["run_mode"] == "quick":
        args.quick = True
    if args.only is None:
        args.only = p["only"]
    args.skip = list(args.skip) + [s for s in p["skip_steps"] if s not in args.skip]
    if args.b2_seeds is None:
        args.b2_seeds = p["b2_seeds"]
    if args.b2_updates is None:
        args.b2_updates = p["b2_updates"]
    if args.b2_drift_mode is None:
        args.b2_drift_mode = p["drift_mode"]
    if p["sysid_aux"]:
        args.b2_sysid_aux = True
    if p["dump_states"] and args.b2_dump_states is None:
        args.b2_dump_states = DUMP_STATES_AUTO


PROFILE_FILE = "profile.txt"
RESUME_LOCAL_DIRNAME = "_resume_local"


def run_is_unfinished(run_dir: Path) -> bool:
    """A run directory that started (has a manifest) but never finalized."""
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return "finished_at_utc" not in manifest


def run_matches_profile(run_dir: Path, args: argparse.Namespace) -> bool:
    """Does an unfinished run belong to args.profile? Prefer the recorded
    profile name; fall back to comparing recorded b2 flags + quick for runs
    started before profile.txt existed."""
    pf = run_dir / PROFILE_FILE
    if pf.is_file():
        try:
            return pf.read_text(encoding="utf-8").strip() == args.profile
        except OSError:
            return False
    recorded: list[str] = []
    ff = run_dir / B2_FLAGS_FILE
    if ff.is_file():
        try:
            recorded = json.loads(ff.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return recorded == build_b2_extra(args) and bool(manifest.get("quick", False)) == args.quick


def find_auto_resume(args: argparse.Namespace, sync_root: Path | None) -> Path | None:
    """Locate an unfinished run of this profile to continue.

    Local latest run first (same VM, notebook restarted), then the newest
    unfinished run under sync_root (fresh VM, Drive survived). A Drive
    candidate is copied under fullruns/_resume_local/ so the run itself works
    on fast local disk while mirroring continues to the same Drive folder.
    """
    local = read_latest_run_dir()
    if local is not None and run_is_unfinished(local):
        if run_matches_profile(local, args):
            print(f"Auto-resume: continuing local unfinished run {local}", flush=True)
            return local
        print(f"Note: local unfinished run {local.name!r} is a different profile; "
              "leaving it alone.", flush=True)

    if sync_root is None or not sync_root.is_dir():
        return None
    cands = []
    for mf in sync_root.glob("*/manifest.json"):
        folder = mf.parent
        if not run_is_unfinished(folder):
            continue
        try:
            started = json.loads(mf.read_text(encoding="utf-8")).get("started_at_utc", "")
        except (OSError, ValueError):
            continue
        cands.append((started, folder))
    for _, folder in sorted(cands, reverse=True):
        if not run_matches_profile(folder, args):
            print(f"Note: unfinished Drive run {folder.name!r} is a different "
                  "profile; leaving it alone.", flush=True)
            continue
        return localize_drive_run(folder)
    return None


def localize_drive_run(folder: Path) -> Path:
    """Copy a mirrored run local so the run itself works on fast disk; the
    mirror keeps writing back to the same Drive folder (same run name)."""
    local_copy = FULLRUNS_ROOT / RESUME_LOCAL_DIRNAME / folder.name
    print(f"Copying Drive run {folder.name!r} to local disk ({local_copy}) "
          "and continuing it there.", flush=True)
    shutil.copytree(folder, local_copy, dirs_exist_ok=True)
    return local_copy


def main() -> None:
    args = parse_args()
    if args.profile:
        apply_profile(args)
    if args.drive_sync is not None:
        os.environ["ITASORL_DRIVE_SYNC"] = str(args.drive_sync)
        print(f"Live mirror -> {args.drive_sync}", flush=True)
    skip = expand_skip(args.skip)
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)

    resume_dir = resolve_resume_dir(args.resume, args.results_dir)
    if (resume_dir is not None and args.drive_sync is not None
            and resume_dir.resolve().is_relative_to(Path(args.drive_sync).resolve())):
        resume_dir = localize_drive_run(resume_dir)  # never run on DriveFS directly
    if (resume_dir is None and args.profile and not args.force_fresh
            and args.results_dir is None):
        resume_dir = find_auto_resume(args, args.drive_sync)
    if resume_dir is not None:
        if args.results_dir is not None and resume_dir.resolve() != args.results_dir.resolve():
            print(
                f"Warning: --results-dir ({args.results_dir}) differs from resume dir ({resume_dir}); "
                "using resume dir.",
                flush=True,
            )
        recorder = RunRecorder.resume(resume_dir)
        quick = recorder.quick
        print(f"ITASORL end-to-end RESUME  (root={ROOT}, quick={quick})", flush=True)
    else:
        recorder = RunRecorder.create(quick=args.quick, out_dir=args.results_dir)
        quick = args.quick
        print(f"ITASORL end-to-end  (root={ROOT}, quick={quick})", flush=True)
    if args.profile and not (recorder.run_dir / PROFILE_FILE).is_file():
        (recorder.run_dir / PROFILE_FILE).write_text(args.profile + "\n", encoding="utf-8")

    b2_out = recorder.run_dir / "artifacts"
    print(f"Recording results -> {recorder.run_dir}", flush=True)
    if skip:
        print(f"Skipping: {', '.join(sorted(skip))}", flush=True)

    t0 = time.perf_counter()

    try:
        if args.only not in ("experiments", "expb2") and "pytest" not in skip:
            if recorder.step_is_done("pytest"):
                print("\n--- resume skip pytest (already ok) ---", flush=True)
            else:
                recorder.run_step("pytest", [sys.executable, "-m", "pytest", "-q"], cwd=ROOT)

        if args.only == "expb2":  # run B-v2 alone: mark every other experiment step skipped
            skip |= {s for s, _, _ in experiment_steps(quick=quick, b2_out=b2_out) if s != "expB2"}

        if args.only != "pytest":
            b2_extra = resolve_b2_extra(args, resume=resume_dir is not None,
                                        run_dir=recorder.run_dir)
            b2_extra = resolve_dump_states(b2_extra, recorder.run_dir)
            for name, cmd, extra in experiment_steps(quick=quick, b2_out=b2_out, b2_extra=b2_extra):
                if name in skip:
                    print(f"\n--- skip {name} ---", flush=True)
                    recorder.note_step(name, status="skipped")
                    continue
                if recorder.step_is_done(name):
                    print(f"\n--- resume skip {name} (already ok) ---", flush=True)
                    continue
                recorder.run_step(name, cmd, cwd=ROOT, extra_artifacts=extra)
    except subprocess.CalledProcessError as exc:
        print("\n" + "=" * 72, flush=True)
        print(f"RUN INTERRUPTED (step exited {exc.returncode}). Everything so far", flush=True)
        print(f"is checkpointed under {recorder.run_dir}", flush=True)
        if args.drive_sync is not None:
            print(f"and mirrored to {Path(args.drive_sync) / recorder.run_dir.name}", flush=True)
        print("Re-run the same command to continue from the last checkpoint.", flush=True)
        print("=" * 72, flush=True)
        raise SystemExit(exc.returncode or 1) from exc

    run_dir = recorder.finalize(total_sec=time.perf_counter() - t0, make_zip=not args.no_zip)
    print(f"Read outcomes: {run_dir / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
