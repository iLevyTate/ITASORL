# Experiment status (living snapshot)

Ralph reads this **every run** before choosing work. Update it when new
`fullruns/` appear, canonical artifacts change, or conclusions shift.

Last updated: **2026-07-01** (post Colab full replication analysis)

---

## Core claim (unchanged)

**Detectable ≠ learned.** Outside oracles read L2 drift trivially (~0.99 AUROC).
Organisms trained only on prediction stay at chance for world identity (~0.51).
Survival pressure (B-v2) does **not** reach the pre-registered encoding bar (0.65).

---

## Latest end-to-end runs

| Run folder | Mode | Commit | GPU | Duration | B-v2 survival @ drift 0.45 |
|------------|------|--------|-----|----------|----------------------------|
| `fullruns/06292026` | quick (60 upd, 2 seeds) | `fbae1e5` | L4 | 22 min | **0.473** |
| `fullruns/06302026` | full (300 upd, 3 seeds) | `4c16be6` | T4 | 238 min | **0.523 ± 0.045** |

Exp A and Exp B metrics are **byte-identical** between quick and full runs.
Long runtime changed B-v2 numerics modestly (+0.05) but **not the verdict**.

Per-seed survival @ drift 0.45 (06302026): **0.586, 0.495, 0.488** (high variance).

Pre-registered verdict (06302026): **H_B2 NOT met** — strengthened negative.
Manipulation check passed. Drag ceiling ≈ 0.75 vs identity target ≈ 0.52.

---

## Canonical published artifacts

| Artifact | Provenance |
|----------|------------|
| `artifacts/expB2/expB2_results.json` | Promoted from `fullruns/06302026` (2026-07-01) |
| `artifacts/expB2/expB2_results_confirmatory_n3.json` | Initial lab run (survival @ 0.45 ≈ **0.595**) |

**Discrepancy:** independent Colab replication (0.523) does **not** reproduce the
initial lab mean (0.595). Same config (300 updates, 3 seeds); likely mix of code
version (rigor-hardening PR #8), GPU variance, and seed noise. Scientific
conclusion is the same (below 0.65); effect size in FINDINGS §9 is now split
into initial vs replication tables.

Narrative: `docs/FINDINGS.md` §7 (next steps), §9 (B-v2).

---

## Open scientific questions

1. **Replication gap:** Why do seeds 1–2 land ~0.49 while seed 0 ~0.586? Code,
   GPU, or true variance? Local seed-0 diagnostic deferred (needs human to start).
2. **Power:** n = 3 is underpowered for TOST/ROPE at L0; `scripts/run_expB2_n10.sh`
   exists for n = 10 extension (long GPU run — human approval required).
3. **Reactive vs representational:** Agent may exploit felt drag without encoding
   persistent world identity; held-out / common-garden probe not yet implemented.
4. **L3 artifact:** Generative fingerprint (§7.3) not built; highest-interest
   direction if L2 null holds under scale.

---

## Infrastructure notes

- Colab full runs: use `notebooks/colab_gpu.ipynb` with **`RUN_PROFILE`**
  (see `ralph/COLAB.md`). Local disk + Drive mirror; `--resume` for timeouts.
- Ralph must **not** start multi-hour GPU sweeps without explicit human approval.
  Prepare scripts, tests, and docs instead; log the run plan under BACKLOG Questions.

---

## How to refresh this file

After a new `fullruns/MMDDYYYY/` completes:

1. Read `SUMMARY.md`, `steps/expB2.json`, `steps/expB2.log` (verdict line).
2. Compare primary endpoint to rows above.
3. Update the table and "Last updated" date.
4. Note whether canonical artifacts should be re-promoted.
