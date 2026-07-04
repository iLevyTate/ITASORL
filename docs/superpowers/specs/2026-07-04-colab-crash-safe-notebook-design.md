# Crash-safe Colab notebook: forced copy, forms, auto-resume

Date: 2026-07-04
Status: approved (pending spec review)
Branch: local-bv3-n10-runner

## Problem

`notebooks/colab_gpu.ipynb` runs multi-hour experiments (bv3_regime_n10 is
roughly 11.5 h) on Colab sessions that disconnect without warning. Three gaps
make a disconnect expensive today:

1. Nothing forces the user off the read-only GitHub copy. Config edits and
   outputs vanish with the tab because the notebook was never copied to Drive.
2. Resume is manual. After a crash the user must hand-edit `FRESH_RUN` and
   `RESUME_RUN_DIR` in a Python cell.
3. The Drive mirror is step-granular. `_sync_mirror()` in
   `itasorl/results_io.py` copies `artifacts/` (which holds the per
   (drift, seed) checkpoint cells) only when a step finishes. For expb2-only
   profiles the single expB2 step is essentially the whole run, so a mid-step
   disconnect loses every checkpoint on the ephemeral VM disk. Dumped
   recurrent states (`runs/<RUN_ID>/states`) are never mirrored at all.
   Additionally, a dead Drive FUSE mount ("transport endpoint is not
   connected", a known Colab failure) makes `shutil.copy2` raise inside the
   recorder, which would kill the run itself.

Goal: open notebook, pick a profile from a dropdown, Run all. If anything
crashes, reopen the Drive copy and Run all again; the run continues where it
left off, losing at most a few minutes of work.

## Non-goals

- No change to experiment science (training, probes, verdict logic).
- No change to the local Windows runner flow (`scripts/run_local.py`); it
  benefits from the shared `results_io.py` fixes but its CLI is untouched.
- No Drive API bulk upload of dumped states (too large for the HTTP fallback).
- No paid-Colab background execution support.

## Design

### 1. Force-copy guard cell (notebook, new first code cell)

Colab notebooks opened from GitHub are in playground mode; the URL contains
`/github/`. Copies saved to Drive have `/drive/` URLs. The guard cell:

- Skips entirely when not on Colab (local mode).
- Reads the browser URL via
  `google.colab.output.eval_js('window.location.href')`.
- If the URL contains `/github/`, raises `SystemExit` with instructions:
  "File -> Save a copy in Drive, then Runtime -> Run all inside your copy."
  Colab cancels all queued cells on error, so Run all stops cold.

`eval_js` needs a connected frontend; during Run all one is always attached.

### 2. Config cell becomes a Colab form

Form fields (Colab `# @param` syntax, `{ display-mode: "form" }`):

- `RUN_PROFILE`: dropdown listing exactly the keys of `_PROFILES`.
- `BRANCH`: string field, default `main`. Kept because the Drive copy always
  clones code from GitHub, and the field is the only way to run an unmerged
  branch without editing code. When `BRANCH != "main"` the cell prints a loud
  warning banner. Provenance is already covered: the manifest records the
  exact git commit.
- `FORCE_FRESH`: boolean checkbox, default off. Deliberately start a new run
  even when an unfinished one exists on Drive.
- `RESUME_RUN_DIR`: string field, default empty. Explicit override; empty
  means automatic detection.

Removed: `FRESH_RUN`, `COPY_DRIVE_RESUME_TO_LOCAL` (behavior becomes
always-on), `RUN_ID`, and the `STATES_DIR` global (see section 4). The
`_PROFILES` dict and its keep-in-sync breadcrumb with
`scripts/run_local.py` stay.

Drive run folders keep the existing date naming: the run dir is
`fullruns/<MMDDYYYY>/` (time-suffixed only when that date folder is taken)
and the mirror folder inherits the name, so runs land at
`MyDrive/ITASORL_results/<MMDDYYYY>/`.

### 3. Auto-resume (notebook run cell)

Decision order when the run cell executes:

1. `RESUME_RUN_DIR` non-empty: resume that folder (Drive paths are copied
   local first, as today).
2. `FORCE_FRESH` checked: start a new run.
3. Auto-scan: list `DRIVE_RESULTS/*/manifest.json`; a candidate is unfinished
   when `bundle.zip` is absent (the bundle is written only at successful run
   end). Take the newest candidate by manifest `started_at_utc`.
   - Compare its recorded `b2_flags.json` with the flags the currently
     selected profile would produce (ignoring `--resume`). The notebook does
     not duplicate the flag-building logic: it imports `build_b2_extra` from
     `scripts/run_e2e.py` (the repo is already cloned by this point) and
     feeds it the profile fields. Because `quick` and `full` record identical
     b2 flags, the manifest `quick` field is also checked against the
     profile's `run_mode`. On match, copy the
     Drive folder to `fullruns/_resume_local/<name>` and pass
     `--resume <local copy>`. If a `resume_pack.zip` (section 6) exists in
     the Drive folder and is newer than the mirrored `manifest.json`, extract
     it over the local copy first, so the freshest checkpoints win.
     Unreadable manifests are skipped with a warning.
   - On mismatch, hard stop with a message naming the profile recorded in the
     unfinished run: switch the dropdown to it, or check `FORCE_FRESH`.

A resumed run mirrors back to the same Drive folder because the mirror
destination is keyed by run dir name and `_resume_local/<name>` preserves it.

The commented-out resume-only cell (current section 8) is deleted; auto-resume
makes it dead weight.

### 4. Dumped states move under the run dir (scripts/run_e2e.py)

`--b2-dump-states` accepts the sentinel value `auto`, resolved at command
build time to `<run_dir>/artifacts/states`. The raw value `auto` is what gets
recorded in `b2_flags.json`, so a resume re-resolves against the active run
dir; absolute paths recorded on a dead VM can never leak into a new session.
The notebook passes `auto` for every profile with `dump_states=True`. The
reanalysis cell reads `<run_dir>/artifacts/states` via `LATEST_RUN.txt`.
Placing states under `artifacts/` also puts them inside the mirror and the
bundle.

### 5. Live incremental checkpoint mirroring (itasorl/results_io.py)

`_write_status()` already runs on every output line, throttled by
`STATUS_SYNC_INTERVAL_SEC`. Add a second, slower timer: every
`CKPT_SYNC_INTERVAL_SEC` (default 300, env override `ITASORL_CKPT_SYNC_SEC`)
mirror files under `artifacts/` whose mtime is newer than the previous
checkpoint sync. Copy is incremental: unchanged files are not recopied. A
disconnect now loses at most one interval of work instead of the whole step.

Fault tolerance, same file:

- Every mirror write (both the fast small-file sync and the checkpoint sync)
  is wrapped so an `OSError` can never propagate; the run always continues on
  local disk.
- On failure the recorder marks the mirror degraded, prints one loud warning
  (not one per line), and keeps retrying every interval; FUSE mounts
  sometimes recover on their own.

### 6. Notebook watchdog thread

The run cell starts a daemon thread in the notebook runtime (the parent
process can talk to `google.colab`; the run_e2e child cannot). Every
`CKPT_SYNC_INTERVAL_SEC` seconds it:

1. Compares the mtime of `status.json` on Drive against the local one. Fresh
   means the FUSE mirror is alive; nothing to do.
2. If stale, attempts `drive.mount('/content/drive', force_remount=True)`.
3. If still stale, uploads `resume_pack.zip` to the run's Drive folder over
   the Drive v3 HTTP API, which does not depend on the FUSE mount. The pack
   contains `manifest.json`, `b2_flags.json`, `status.json`, and
   `artifacts/cells/*.json`, everything resume needs, a few MB at most.
   Auth is requested once up front in the mount cell
   (`google.colab.auth.authenticate_user()`), so no popup appears mid-run.
4. Prints a screaming banner on any degradation so the user sees it in the
   open tab (the keep-alive cell keeps the tab alive anyway).

The thread exits when the run subprocess does. Watchdog failures are caught
and printed, never raised.

Durability ladder: local disk (always) -> FUSE mirror (normal) -> auto
remount (self-heal) -> Drive API resume pack (worst case).

### 7. Docs and cleanup

- Notebook header markdown rewritten: 1. Copy to Drive (guard enforces),
  2. pick a profile in the form, 3. Run all; after any crash reopen the Drive
  copy and Run all again.
- Failure diagnostics cell message updated: everything is mirrored on Drive
  at the printed path; reopening the Drive copy and Run all resumes
  automatically.
- `results/README.md` mirror description updated (line about
  `ITASORL_DRIVE_SYNC`) to cover incremental checkpoint sync and fault
  tolerance.
- `scripts/README.md` resume notes updated for `--b2-dump-states auto`.

## Error handling summary

- Guard cell: non-Colab skips; `eval_js` failure falls back to a printed
  warning rather than blocking local runs.
- Auto-scan: corrupt or unreadable `manifest.json` skips that folder with a
  warning; zero candidates means a fresh run.
- Mirror and watchdog: never crash the run; degrade loudly and retry.
- Profile mismatch on auto-resume: hard stop, explicit instructions.

## Testing

pytest (runs in CI, no Colab dependency):

- Incremental checkpoint mirror: new file under `artifacts/cells/` appears in
  the mirror after the interval elapses; unchanged files are not recopied
  (mtime comparison); interval honored.
- Fault tolerance: removing the mirror directory mid-run degrades with a
  warning and does not raise; recovery resumes copying.
- `run_e2e.py`: `--b2-dump-states auto` resolves to
  `<run_dir>/artifacts/states` on fresh runs; `b2_flags.json` records the
  raw `auto`; replay on resume re-resolves against the resumed run dir.
- Notebook profile sync: parse `notebooks/colab_gpu.ipynb` JSON, assert the
  `RUN_PROFILE` dropdown options equal `scripts/run_local.py` `PROFILES`
  keys and the notebook `_PROFILES` keys.

Manual (Colab only, stays with the user): guard cell on the GitHub copy vs a
Drive copy, form rendering, a quick-profile crash-and-resume cycle, watchdog
banner on a forced unmount.

## Accepted trade-offs

- Dumped states are excluded from the HTTP resume pack. If FUSE dies, never
  recovers, and the VM dies, some states for offline re-probing are lost, but
  the run itself always remains resumable.
- Auto-resume keys on `bundle.zip` absence; a run that failed terminally (as
  opposed to disconnecting) is also offered for resume, which is the desired
  behavior after fixing the underlying error.
- `bundle.zip` zips the whole run dir, so states now ride along: roughly
  1 MB per agent per cell (110 eps x 24 steps x 96 hidden float32), about
  40-60 MB for bv3_regime_n10. Accepted; the bundle becomes self-contained
  for offline reanalysis. A side benefit of states living in the run dir:
  the Drive-to-local resume copy restores states from before the crash,
  which are lost entirely today.
- An unfinished run recorded before this change carries an old explicit
  states path in `b2_flags.json`; auto-resume flags the mismatch and hard
  stops. The `RESUME_RUN_DIR` override still resumes it. No compatibility
  shim.

## Files touched

- `notebooks/colab_gpu.ipynb` (guard cell, form config, auto-resume run cell,
  watchdog, reanalysis path, header and diagnostics text, delete resume-only
  cell)
- `itasorl/results_io.py` (incremental checkpoint sync, fault-tolerant mirror)
- `scripts/run_e2e.py` (`--b2-dump-states auto` sentinel)
- `tests/` (new tests per above)
- `results/README.md`, `scripts/README.md`
