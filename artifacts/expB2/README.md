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
| `behavior_audit_l3_h7_traces.json` | Behavior-mediation audit of the second-capacity hidden=7 n=10 run (`fullruns/l3_h7_traces/states`). Survival resid_trace **0.722** [0.678, 0.763] replicates hidden=8's 0.726; the survival-vs-predictor dissociation is NOT met (+0.023 < +0.05), so survival-specificity is artifact-conditional. FINDINGS 10.5. |
| `heldout_l3_h8_summary.json` | Per-seed summary of the held-out fingerprint probe (`fullruns/l3_h8_heldout`, frozen spec 2026-07-14), extracted by `scripts/promote_heldout_artifact.py` with the run's config fingerprint and git commit embedded. Transfer **0.773** (9/10, rule PASSES); common-garden tail **0.557** (rule FAILS -> reactive); late tail 0.492 (chance); standard pools reproduce the 0.752 headline exactly. FINDINGS 10.6. |
| `bv3_n10_summary.json` | Per-seed pooled targets for the B-v3 regime n=10 run (`fullruns/07062026`, `scripts/promote_ab_summaries.py`): survival **0.610** [0.585, 0.634], below the 0.65 bar. FINDINGS 7.1. |
| `sysid_ceiling_n10_summary.json` | Per-seed pooled targets for the sysid-aux capacity-ceiling n=10 run (`fullruns/07092026`, `scripts/promote_ab_summaries.py`): **0.596** [0.577, 0.616], below the bar. FINDINGS 7.1. |

Experiment A and Experiment B (L1/L2 arc) summaries live in `artifacts/expA/`
and `artifacts/expB/`, promoted from the recorded `fullruns/06302026` bundle by
`scripts/promote_ab_summaries.py`.

New runs write to `fullruns/<MMDDYYYY>/artifacts/` or here when promoted manually.
Promote a run by copying `expB2_results.json` and `expB2_survival.png` from the run
folder and updating this README if provenance changes.
`scripts/audit_stats_recheck.py` re-verifies every number quoted in
`README.md` / `docs/FINDINGS.md` / `docs/PAPER_OUTLINE.md` against these
committed artifacts; run it before publishing any doc change.

## Promotion history

| Date | Source | Commit | Notes |
|------|--------|--------|-------|
| 2026-07-01 | `fullruns/06302026` | `4c16be6` | Independent Colab full run (T4, 237 min). Survival @ drift 0.45 = **0.523 ± 0.045** (seeds 0.586, 0.495, 0.488). Replaces prior `expB2_results.json`; initial lab run (≈ 0.595) kept as `expB2_results_confirmatory_n3.json`. Documented in `docs/FINDINGS.md` §9. |
| 2026-07-13 | `fullruns/l3_h8_traces` | `2888d37` | L3 hidden=8 n=10 re-run with per-timestep behavior traces (RTX 4050). Exact deterministic replication of the audited run (survival 0.752, L0 0.517); per-timestep-controlled behavior-independent signal **0.726** [0.685, 0.765]. Audit JSON promoted as `behavior_audit_l3_h8_traces.json`; documented in `docs/PREREGISTRATION_L3.md` (2026-07-13 entry). |
| 2026-07-14 | `fullruns/l3_h7_traces` | see prereg log | Second-capacity hidden=7 n=10 run promoted as `behavior_audit_l3_h7_traces.json`; run provenance documented in `docs/PREREGISTRATION_L3.md` (2026-07-14 entries) and FINDINGS 10.5. |
| 2026-07-16 | `fullruns/l3_h8_heldout` | `ed88df0` (run) | Held-out probe per-seed summary promoted as `heldout_l3_h8_summary.json` (config fingerprint `7ae90a71eb8a4103` embedded). FINDINGS 10.6. |
| 2026-07-16 | `fullruns/07062026`, `fullruns/07092026`, `fullruns/06302026` | `820849f` / `f848738` / `4c16be6` (runs) | Research-integrity audit: B-v3 and capacity-ceiling per-seed summaries promoted here; Experiment A/B step summaries promoted to `artifacts/expA/` and `artifacts/expB/`; `scripts/audit_stats_recheck.py` added as the number-verification gate. |
