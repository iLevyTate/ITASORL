# ITASORL recorded runs

End-to-end runs via `python scripts/run_e2e.py` write a dated folder under `fullruns/`:

```
fullruns/MMDDYYYY/
  SUMMARY.md       ← start here (plain-English outcome)
  manifest.json    ← step status, timings, artifact paths
  status.json      ← live step + last line (updated during run)
  combined.log     ← full stdout (updated live during run)
  bundle.zip       ← download this to keep everything
  steps/
    pytest.log
    expB_full.log
    expB_full.json   ← parsed metrics per step
    ...
  artifacts/
    docs/figures/...  ← copied PNGs
    expB2_results.json
    expB2_survival.png
```

**Latest run path** is also written to `results/LATEST_RUN.txt`.

**Canonical B-v2 metrics** (full scale, 300 updates, 3 seeds) live in
`artifacts/expB2/expB2_results.json`, promoted from `fullruns/06302026` (Colab,
2026-06-30). Narrative and comparison vs the initial lab run: `docs/FINDINGS.md` §9.
The prior lab confirmatory JSON is archived as `expB2_results_confirmatory_n3.json`.

**Watch a run in progress** (second terminal):

```bash
python scripts/watch_run.py --follow
```

On Google Colab (see `notebooks/colab_gpu.ipynb`):

- Pick **`RUN_PROFILE`** from the config-cell dropdown (full list in the notebook or `python scripts/run_local.py --list`).
- Runs use **local disk** (`fullruns/` under the repo). Do not set `--results-dir` to Drive (FUSE I/O is slow and often fails).
- `ITASORL_DRIVE_SYNC` mirrors `combined.log`, `status.json`, and `manifest.json` continuously, per-step outputs after each step, and `artifacts/` (expB2 checkpoint cells, dumped states) incrementally every `ITASORL_CKPT_SYNC_SEC` seconds (default 300). Mirror failures never stop the run; syncing warns once, retries, and reports recovery.
- If Colab disconnects, reopen your Drive copy of the notebook and Runtime -> Run all again. Auto-resume finds the newest unfinished run in `MyDrive/ITASORL_results`, checks it matches the selected profile, and continues it (paste a specific folder into `RESUME_RUN_DIR` to override).
- After the run: `python scripts/compare_expB2_artifacts.py --run fullruns/MMDDYYYY` (also in the notebook summary cell).
