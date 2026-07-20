# Local interruptible B-v3 n=10 runner

Date: 2026-07-03
Status: approved (pending spec review)
Branch: local-bv3-n10-runner (off origin/main 39df3bd, kept local)

## Problem

The pre-registered adjudication of the 07032026 B-v3 regime result (survival
pool target 0.615 +/- 0.083, intermediate zone) requires the n=10 power
extension. Colab free-tier sessions disconnect before a 20-cell run
(2 drifts x 10 seeds, roughly 20-40 min per cell) can finish. The run must
execute locally on a Windows laptop (local GPU, Git Bash) and must survive
interruption: sleep, reboot, OOM, Ctrl+C. `run_e2e.py --resume` resumes at
step granularity only, and expB2 is a single step, so today an interrupted
expB2 step restarts from cell 1.

## Non-goals

- No change to experiment science (training, probes, verdict logic).
- No replacement of `scripts/run_expB2_n10.sh` (plain B-v2, artifacts naming);
  it stays as is.
- No scheduling, thermal management, or keep-awake tooling.

## Design

### 1. Cell checkpoint files (scripts/run_expB2.py)

`run_cell()` already returns a JSON-serializable dict per (drift, seed) cell.
After each cell completes, write it to
`<out-dir>/cells/cell_d{drift:.2f}_s{seed}.json` atomically (write to a temp
file in the same directory, then `os.replace`). The existing aggregate
checkpoint write is unchanged so `scripts/watch_run.py` live-tailing keeps
working.

Each cell file contains:

- `fingerprint`: hash of science-relevant config: updates, n_eps, max_steps,
  hidden, ray_steps, shaping_coef, pool_n, pool_steps, mp_pairs, mp_prefix,
  mp_branch, basal_e, n_pellets, reach, sysid_aux, sysid_coef, drift_mode,
  drifts, resolved device.
- `git_commit`: short hash, provenance only (recorded, not enforced; warn on
  mismatch at resume).
- `cell`: the `run_cell()` output dict.

NaN values appear in cell metrics (for example `pool_ceiling_drag` at
drift 0). Python json serializes and reloads NaN by default; the round-trip
must be covered by a test.

### 2. Resume semantics (explicit, never silent)

New `--resume` flag on `run_expB2.py`:

- Without `--resume`: if `<out-dir>/cells/` contains cell files, abort with a
  loud error naming the directory and the two options (pass `--resume`, or use
  a fresh out-dir). A fresh run can never silently skip work.
- With `--resume`: load each cell file, verify its fingerprint equals the
  current config fingerprint (mismatch is a hard error naming the file), feed
  matched cells through the existing `record_cell()` + `print_cell()` path
  (marked as resumed in output), and run only the missing (drift, seed) tasks.
- A corrupt or unreadable cell file is a hard error naming the file; the user
  deletes that one file and resumes.

Final `expB2_results.json` is rebuilt from cell files in canonical order
(sorted by drift, then seed) so per-seed list positions are deterministic even
with `--workers > 1` or interleaved resume. This also fixes the existing
ordering ambiguity where `imap_unordered` fills lists in completion order.

### 3. Harness plumbing (scripts/run_e2e.py)

When `run_e2e.py` runs with `--resume`, forward `--resume` to the expB2
command in `build_b2_extra()` (the function gains a resume argument or reads
it from args). Fresh `run_e2e.py` runs forward nothing, so the fresh-run guard
in run_expB2.py stays active. Result: `python scripts/run_e2e.py --resume`
works end-to-end and produces the standard `fullruns/MMDDYYYY/` layout. The
same path benefits Colab runs whose run dir is mirrored to Drive.

### 4. Local launcher (scripts/run_local.py)

Python launcher `scripts/run_local.py`, profile-driven like the Colab
notebook's config cell. Usage:

    python scripts/run_local.py bv3_regime_n10          # fresh start
    python scripts/run_local.py bv3_regime_n10 --resume # continue
    python scripts/run_local.py --list                  # show profiles

Behavior:

1. Profile table: a `PROFILES` dict with the same keys and entries as the
   notebook's `_PROFILES` (quick, full, bv3_regime, bv3_regime_n10,
   bv2_ceiling, bv3_ceiling, b2_only, b2_seed0, experiments_no_b2). The
   notebook cannot import repo code before it clones the repo, so the two
   tables are intentionally duplicated; each carries a comment pointing at
   the other ("keep in sync").
2. Profile-to-flags mapping: run_mode quick -> `--quick`; only ->
   `--only expb2`; skip_steps -> `--skip ...`; b2_seeds -> `--b2-seeds ...`;
   b2_updates -> `--b2-updates ...`; drift_mode -> `--b2-drift-mode ...`;
   sysid_aux -> `--b2-sysid-aux`; dump_states -> `--b2-dump-states
   <RUN_DIR>/states`. Implemented as a pure function returning the argv list
   so it is unit-testable.
3. CUDA check: fail loudly if torch cannot see a GPU (skippable with
   `--allow-cpu` for deliberate CPU runs).
4. Free-RAM guard: refuse to start below MIN_FREE_GB (default 4) free system
   RAM, Windows-aware (same intent as run_expB2_n10.sh, implemented in
   Python).
5. Fresh start: `RUN_DIR = fullruns/<MMDDYYYY>` and invoke run_e2e.py with
   `--results-dir RUN_DIR` plus the mapped profile flags. States live inside
   the run dir, so the whole run is one folder.
6. Resume: resolve RUN_DIR from run_e2e's latest-run pointer file (exact
   constant read from run_e2e.py at implementation time) and invoke
   `run_e2e.py --resume RUN_DIR` with the same profile flags, keeping the
   states path stable across sessions. The profile argument is still
   required on resume; passing a different profile than the original run is
   caught by the run_expB2.py fingerprint gate, not by the launcher.

Worst-case loss on interruption is one cell. A second fresh start on the
same day targets the same RUN_DIR and is stopped by the fresh-run guard
(section 2); use `--resume` or remove the day's folder first.

### 5. Testing

Pure-function tests, no training and no GPU:

- fingerprint: identical config matches, any science knob change mismatches.
- cell file round-trip including NaN values.
- resume task filtering: given a cells dir with k of n cells, exactly n-k
  tasks remain.
- fresh-run guard: non-empty cells dir without `--resume` aborts.
- canonical ordering: rebuilt results lists are ordered by (drift, seed)
  regardless of cell completion order.
- run_e2e plumbing: `build_b2_extra` emits `--resume` only in resume mode.
- launcher mapping: each PROFILES entry produces the expected run_e2e argv
  (pure function, parametrized over all profiles); `--list` output includes
  every profile.

CI hygiene: `ruff check .` clean before any push (F541 trap).

### 6. Delivery

All work stays local on branch `local-bv3-n10-runner`. No push and no PR
until the user asks. The user launches the run themselves.

## Acceptance criteria

1. Interrupting the n=10 run at any point and rerunning with `--resume` skips
   all completed cells and finishes the remaining ones.
2. A completed run produces `fullruns/MMDDYYYY/` with expB2 results whose
   per-seed lists are in seed order, consumable by
   `scripts/compare_expB2_artifacts.py --run`.
3. A fresh run pointed at a dir with stale cells aborts instead of silently
   skipping.
4. Tests pass on Python 3.10-3.12 locally; ruff clean.
5. Every notebook RUN_PROFILE can be launched locally by name via
   `python scripts/run_local.py <profile>`, including `--resume`.
