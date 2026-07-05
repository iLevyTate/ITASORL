# Colab run playbook

Use this with [`notebooks/colab_gpu.ipynb`](../notebooks/colab_gpu.ipynb). Ralph's
Claude Code loop does not run in Colab; this file maps **research next steps** to
notebook **RUN_PROFILE** presets you can execute on GPU.

Last updated: **2026-07-04**

---

## Before you start

1. **File -> Save a copy in Drive** (the guard cell blocks the read-only GitHub copy).
2. **Runtime -> Change runtime type -> GPU** (T4, L4, A100 all work).
3. Pick **`RUN_PROFILE`** from the config-cell dropdown. `BRANCH` defaults to `main`;
   change it only for testing unmerged code.
4. **Runtime -> Run all**. Leave the tab open (keep-alive cell keeps the session awake).
5. After the run, the reanalysis and compare cells run automatically.

**Branch:** use `main` unless you need an unmerged fix (check open PRs).

---

## RUN_PROFILE presets

| Profile | Wall time (T4, approx) | What it does | Ralph / research goal |
|---------|------------------------|--------------|------------------------|
| `quick` | ~25 min | Full e2e, B-v2 at 60 updates / 2 seeds | Smoke test after code changes |
| `full` | ~4 hr | Full e2e, B-v2 at 300 updates / 3 seeds | Replicate `fullruns/06302026` |
| `b2_seed0` | ~75 min | **Only** expB2, seed 0, 300 updates | Diagnose replication gap (Colab seed 0 was 0.586) |
| `b2_only` | ~3.7 hr | **Only** expB2, 3 seeds, 300 updates | Re-run B-v2 without repeating A/B |
| `experiments_no_b2` | ~15 min | All steps except expB2 | Confirm A/B unchanged; skip long GPU step |

Reference numbers (canonical): survival @ drift 0.45 = **0.523 ± 0.045**; SESOI = **0.65**.

---

## Colab timeout strategy

Free Colab often disconnects around **90 min** on GPU.

| Profile | Fits one session? | Strategy |
|---------|-------------------|----------|
| `quick` | Yes | Single session |
| `full` | Often no | Drive mirror; **resume** after disconnect |
| `b2_seed0` | Borderline | Keep-alive + resume if needed |
| `b2_only` | No | Resume; or run overnight on Pro |
| `experiments_no_b2` | Yes | Single session |

**Resume:** reopen your Drive copy of the notebook and **Runtime -> Run all** again.
Auto-resume finds the newest unfinished run in `MyDrive/ITASORL_results`, checks it
matches the selected `RUN_PROFILE`, and continues it. Checkpoints mirror to Drive
every ~5 min during the run, so at most a few minutes repeat.

To resume a specific folder instead (advanced): paste its path into the
`RESUME_RUN_DIR` form field and Run all.

---

## After the run

```bash
python scripts/compare_expB2_artifacts.py --run fullruns/MMDDYYYY
```

Or use the notebook **compare** cell (uses `LATEST_RUN.txt`).

If results supersede canonical metrics, promote manually:

```bash
cp fullruns/MMDDYYYY/artifacts/expB2_results.json artifacts/expB2/
cp fullruns/MMDDYYYY/artifacts/expB2_survival.png artifacts/expB2/
```

Then update `ralph/EXPERIMENT_STATUS.md` and `docs/FINDINGS.md` §9.

---

## Mapping to Ralph NEXT_STEPS

| Ralph item | Colab action |
|------------|--------------|
| Seed-0 diagnostic | `RUN_PROFILE = "b2_seed0"` |
| Full replication | `RUN_PROFILE = "full"` |
| Artifact comparison | Post-run compare cell (no GPU) |
| n=10 extension | Not in notebook yet; ~many hours; use `scripts/run_expB2_n10.sh` locally with approval |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Run died mid expB2 | Reopen Drive copy, Run all (auto-resume) |
| CUDA False | Change runtime type, re-run GPU check cell |
| Numbers differ from 06302026 | Expected (variance); compare per-seed with compare script |
