# ITASORL recorded runs

End-to-end runs via `python run_e2e.py` write a timestamped folder here:

```
results/runs/YYYYMMDD_HHMMSS/
  SUMMARY.md       ← start here (plain-English outcome)
  manifest.json    ← step status, timings, artifact paths
  combined.log     ← full stdout from every step
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

On Google Colab, the notebook copies `bundle.zip` to Drive and triggers a browser download.
