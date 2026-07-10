# Experiment status (living snapshot)

Ralph reads this **every run** before choosing work. Update it when new
`fullruns/` appear, canonical artifacts change, or conclusions shift.

Last updated: **2026-07-10** (post n=10 sysid-aux CEILING; L2/pooled ceiling now decisively below 0.65)

---

## Core claim (unchanged)

**Detectable ≠ learned.** Outside oracles read L2 drift trivially (~0.99 AUROC).
Organisms trained only on prediction stay at chance for world identity (~0.51).
Survival pressure (B-v2) does **not** reach the pre-registered encoding bar (0.65).
The identifiable, policy-relevant B-v3 regime (per-episode constant drag) lifts the
survival probe to 0.610 at n=10, still below 0.65. The pre-registered sysid-aux ceiling
confirms this is near-architectural: even with the survival trunk supervised directly on
drag, the n=10 pooled probe reaches only 0.596 (90% CI [0.577, 0.616], upper bound below
0.65), while the matched-pair detectability channel reaches ~0.70. Identity is decodable
when forced in; the pooled persistent-direction readout simply saturates below the bar.
The n=10 ceiling CI now excludes 0.65 decisively (the n=3 CI [0.576, 0.667] had straddled
it), so the "L2 null holds under scale" gate is met and L3 is green-lit.

---

## Latest end-to-end runs

| Run folder | Mode | Commit | GPU | Duration | B-v2 survival @ drift 0.45 |
|------------|------|--------|-----|----------|----------------------------|
| `fullruns/06292026` | quick (60 upd, 2 seeds) | `fbae1e5` | L4 | 22 min | **0.473** |
| `fullruns/06302026` | full (300 upd, 3 seeds) | `4c16be6` | T4 | 238 min | **0.523 ± 0.045** |
| `fullruns/07062026` | full (300 upd, 10 seeds, regime B-v3) | `820849f` | RTX 4050 | 337 min | **0.610 +/- 0.047** |
| `fullruns/07072026` | full (300 upd, 3 seeds, regime, sysid-aux CEILING) | `0e69d3d` | T4 | 206 min | **0.622** pooled / **~0.80** matched-pair (capacity ref) |
| `fullruns/07092026` | full (300 upd, 10 seeds, regime, sysid-aux CEILING) | `f848738` | RTX 4050 | 355 min | **0.596** [0.577, 0.616] pooled / **~0.70** matched-pair (capacity ref, n=10) |

Exp A and Exp B metrics are **byte-identical** between quick and full runs.
Long runtime changed B-v2 numerics modestly (+0.05) but **not the verdict**.

Per-seed survival @ drift 0.45 (06302026): **0.586, 0.495, 0.488** (high variance).

Pre-registered verdict (06302026): **H_B2 NOT met** — strengthened negative.
Manipulation check passed. Drag ceiling ≈ 0.75 vs identity target ≈ 0.52.

Per-seed survival @ drift 0.45 (07062026, B-v3 regime, n=10): **0.669, 0.514, 0.594,
0.642, 0.611, 0.586, 0.615, 0.638, 0.558, 0.677** (2/10 cross 0.65).

Pre-registered verdict (07062026, B-v3 regime): **H_B3 NOT met** (intermediate zone).
Survival 0.610 (90% CI [0.585, 0.634]) beats untrained 0.500 and predictor 0.513 by
>0.05 but misses the 0.65 SESOI; the n=10 CI excludes 0.65, so the adjudication is
decisive. Volatility readout (target_var 0.535, target_full 0.611) also below 0.65. L0
control equivalent to chance; manipulation check passed. Dissociation: identity target
0.610 > momentary drag-tracking ceiling 0.487 (not merely reactive drag-tracking).

Sysid-aux CEILING (07072026, n=3, T4, 206 min, commit 0e69d3d): supervising the survival
trunk directly on drag (breaks readout-not-reward; a capacity reference, NOT H_B2 evidence)
lifts the pooled target to only 0.622 @ drift 0.45 (90% CI [0.576, 0.667]), barely above
the unsupervised n=10 headline (0.610). The pre-registered detectability-style matched-pair
channel reaches ~0.80 (0.785, 0.771, 0.849); ceiling(drag) 0.556. Reading: world identity
IS linearly decodable from this trunk when forced in (matched-pair ~0.80), but the pooled
persistent-direction readout saturates near ~0.62, so the 0.610 headline sits at the pooled
probe's architectural ceiling. This CONFIRMS the null (not a probe-capacity floor a stronger
pooled probe would clear) and does not license chasing 0.65. L2/pooled line concluded; the
pre-registered "L3 if L2 null holds under scale" gate is now met. Next: L3
generative-fingerprint scope spec for human sign-off (BACKLOG Questions). NOTE: the run
finalize did not fully close (status.json still running:true, no bundle.zip); the science is
complete (manifest expB2 ok, exit 0, results.json + png present).

Sysid-aux CEILING n=10 (07092026, RTX 4050, 355 min, commit f848738; `bv3_ceiling_n10`):
the pre-registered power extension of the ceiling. Pooled target 0.596 @ drift 0.45 (90% CI
[0.577, 0.616]), matched-pair 0.702; drag-ceiling 0.503, energy-ceiling 0.774; predictor
0.513 and untrained 0.500 at chance; L0 controls at chance; leakage clean; engagement and
speed (0.958) gates passed. The tight n=10 CI [0.577, 0.616] EXCLUDES 0.65 decisively
(where the n=3 ceiling CI [0.576, 0.667] had straddled it); a more honest t-based interval
[0.573, 0.619] (mean 4.3 SE below 0.65) also excludes it, so the claim does not rest on the
lower-coverage percentile-of-mean bootstrap (2026-07-10 audit finding). This settles the open
question: even a trunk supervised directly on drag saturates the pooled persistent-direction
readout at ~0.60, while identity stays decodable-when-forced (matched-pair 0.70). Caveat: the
pooled probe is Exp-B-comparable, not confound-clean (per-world early-death survivorship
asymmetry; matched-pair is the clean channel), and the volatility readouts are exploratory. This is the
strongest form of the architectural-ceiling reading and confirms the null is not a
probe-capacity floor. Run finalized cleanly (bundle.zip present, status finished). NOTE: this
run rendered on `main`, whose summary pipeline still lacks the CEILING relabel, so its
SUMMARY.md mislabels the number as a "weak H_B2 trace"; the fix is in this same change. Gate
"L3 if L2 null holds under scale" is now MET; PREREGISTRATION_L3.md drafted for sign-off.

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

**B-v3 candidate:** `fullruns/07062026` (n=10 regime, 0.610) is the candidate canonical
B-v3 artifact; not yet promoted (flag for human decision before moving files).

Narrative: `docs/FINDINGS.md` §7 (next steps), §9 (B-v2).

---

## Open scientific questions

1. **Replication gap:** Why do seeds 1–2 land ~0.49 while seed 0 ~0.586? Code,
   GPU, or true variance? Local seed-0 diagnostic deferred (needs human to start).
2. **Power:** RESOLVED (07062026): the n=10 regime extension ran (RTX 4050, 337 min);
   survival 0.610 +/- 0.047, 90% CI [0.585, 0.634] excludes 0.65, and L0 TOST/ROPE pass
   at n=10 (equivalent to chance).
3. **Reactive vs representational:** Agent may exploit felt drag without encoding
   persistent world identity; held-out / common-garden probe not yet implemented.
   (07062026: identity target 0.610 > drag-tracking ceiling 0.487 is suggestive of a
   persistent signal, but the held-out probe remains the clean test. 07072026 sysid-aux
   ceiling: pooled saturates ~0.62 even when supervised while matched-pair reaches ~0.80,
   consistent with a decodable-but-not-pooled-persistent signal.)
4. **L3 artifact:** Generative fingerprint (§7.2) not built; highest-interest
   direction if L2 null holds under scale. Gate now MET (n=10 null + sysid-aux ceiling
   confirmation); scope spec pending human sign-off in BACKLOG Questions.

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
