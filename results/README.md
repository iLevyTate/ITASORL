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

**Watch a run in progress** (second terminal):

```bash
python scripts/watch_run.py --follow
```

On Google Colab (see `notebooks/colab_gpu.ipynb`):

- Results are written directly to Google Drive (`--results-dir`) so a VM crash does not lose completed steps.
- `ITASORL_DRIVE_SYNC` mirrors `combined.log`, `status.json`, `manifest.json`, and per-step outputs after each step.
- If Colab disconnects, set `RESUME_RUN_DIR` in the notebook and run `python scripts/run_e2e.py --resume <drive_run_folder>` to skip steps already marked `ok` in `manifest.json`.
