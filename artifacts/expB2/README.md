# Experiment B-v2 published artifacts

Canonical committed outputs from pre-registered confirmatory runs (not ephemeral
`fullruns/` bundles from `scripts/run_e2e.py`).

| File | Description |
|------|-------------|
| `expB2_results.json` | **Canonical** full-scale metrics from independent Colab replication (`fullruns/06302026`, commit `4c16be6`, 300 updates, 3 seeds, Tesla T4). Includes CIs, drag ceiling, manipulation-check fields. |
| `expB2_survival.png` | Figure matching the canonical JSON |
| `expB2_results_confirmatory_n3.json` | Archived initial lab confirmatory run (pre-rigor-hardening; survival @ drift 0.45 ≈ 0.595) |
| `expB2_survival_confirmatory_n3.png` | Figure for the archived initial run |
| `behavior_audit_l3_n10.json` | Behavior-mediation audit of the L3 n=10 dumps (`fullruns/l3_n10_audited/states`) via `scripts/audit_behavior_mediation.py`. Reproduces the published survival d=0.45 numbers exactly: target 0.752, behavior-only 0.689/0.705, in-fold controlled 0.676 linear / 0.659 quadratic. Old-format dumps, so per-episode control only; the per-timestep control needs the trace-extended dumps of the owed re-runs. |
| `behavior_audit_l3_h8_traces.json` | Behavior-mediation audit of the trace-extended hidden=8 re-run (`fullruns/l3_h8_traces/states`). Deterministic replication of every published figure, plus the decisive per-timestep control: survival resid_trace **0.726** [0.685, 0.765], 9/10 seeds >= 0.65 (quad 0.721); behavior trace alone 0.803; untrained resid_trace 0.498 (exact chance); predictor 0.574. Outcome per the pre-registered decision rule: claim STRENGTHENED. |

New runs write to `fullruns/<MMDDYYYY>/artifacts/` or here when promoted manually.
Promote a run by copying `expB2_results.json` and `expB2_survival.png` from the run
folder and updating this README if provenance changes.

## Promotion history

| Date | Source | Commit | Notes |
|------|--------|--------|-------|
| 2026-07-01 | `fullruns/06302026` | `4c16be6` | Independent Colab full run (T4, 237 min). Survival @ drift 0.45 = **0.523 ± 0.045** (seeds 0.586, 0.495, 0.488). Replaces prior `expB2_results.json`; initial lab run (≈ 0.595) kept as `expB2_results_confirmatory_n3.json`. Documented in `docs/FINDINGS.md` §9. |
| 2026-07-13 | `fullruns/l3_h8_traces` | `2888d37` | L3 hidden=8 n=10 re-run with per-timestep behavior traces (RTX 4050). Exact deterministic replication of the audited run (survival 0.752, L0 0.517); per-timestep-controlled behavior-independent signal **0.726** [0.685, 0.765]. Audit JSON promoted as `behavior_audit_l3_h8_traces.json`; documented in `docs/PREREGISTRATION_L3.md` (2026-07-13 entry). |
