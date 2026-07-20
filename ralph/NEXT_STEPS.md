# Next steps (Ralph action queue)

Prioritized work Ralph **may** pick when no P0/P1 bug is open. Each item must
be **evidence-backed** and **verifiable without inventing results**.

Mark status: `[ready]` Ralph can implement · `[blocked]` needs human · `[done]`

---

## Tier 1 — Clarify the replication gap (highest leverage)

| Status | Item | Rationale |
|--------|------|-----------|
| `[ready]` | **CUDA/CPU + cross-run determinism test** for B-v2 pooled readout (seed 0, `--quick` scale first). Compare two back-to-back runs on same device; document drift. | Explains 0.595 vs 0.523 without a 4 hr sweep. Already on BACKLOG as P2. |
| `[done]` | **Script** `scripts/compare_expB2_artifacts.py` + Colab compare cell. | Run after Colab/local B-v2. |
| `[blocked]` | Local seed-0 full-scale re-run (300 updates) on lab GPU vs Colab T4 seed 0 (0.586). | Colab: `RUN_PROFILE = "b2_seed0"` in `notebooks/colab_gpu.ipynb`. |

---

## Tier 2 — Hardening & rigor (code, not long runs)

| Status | Item | Rationale |
|--------|------|-----------|
| `[done]` | `collect_pool` / early-death guard tests (BACKLOG P2). | Done: test_experiment_b2.py (test_collect_pool_returns_fixed_length, test_collect_pool_excludes_early_deaths, test_pooled_readout_too_few_survivors_guard), verified green 2026-07-20. |
| `[done]` | Extreme latent → env action bounds test (BACKLOG P3). | Done: test_agent_ac.py test_env_action_bounded_for_extreme_latents (magnitudes 50..1e8), verified green 2026-07-20. |
| `[done]` | Update `docs/FINDINGS.md` §7 item 1 to reference §9 replication (0.523) and retire "attempted" ambiguity if wording still implies open positive. | Done: §7.1 item 1 already references §9 and states 0.523 (matches canonical §9 table); no open-positive ambiguity remains (verified 2026-07-20). |

---

## Tier 3 — Scale & new experiments (human gate for GPU time)

| Status | Item | Rationale |
|--------|------|-----------|
| `[done]` | n = 10 extension. | Done for B-v3 (0.610) and both L3 capacities (h8 0.752, h7 0.737) at n=10. FINDINGS §7.1, §10. |
| `[done]` | Held-out / common-garden probe. | Built and run for L3 (`heldout_l3_h8_summary.json`): transfer 0.773, common-garden reactive 0.557; reverse + cross-recipe follow-ups also done. FINDINGS §10.6–10.7. |
| `[done]` | L3 generative fingerprint scaffold. | `itasorl/surrogate_l3.py` (`G_motion`) + `l3` world hook + oracle gate; run at hidden=8/7. FINDINGS §10. |

---

## Tier 4 — Deprioritized / closed

| Status | Item | Notes |
|--------|------|-------|
| `[done]` | Promote 06302026 to canonical `artifacts/expB2/`. | 2026-07-01. |
| `[done]` | FINDINGS §9 dual-run table (lab vs Colab). | 2026-07-01. |
| — | Nonlinear probe, longer open-loop objective | Already checked; not the bottleneck (FINDINGS §3.4, §7). |

---

## Rules for Ralph

1. Pick **at most one** item per run from this file **or** one bug from BACKLOG.
2. Prefer `[ready]` items that add tests, tooling, or docs over speculative features.
3. Move completed items to `[done]` here and note in `JOURNAL.md`.
4. If research changes the headline conclusion, update `EXPERIMENT_STATUS.md` in
   the same commit.
5. Never mark Tier 3 `[blocked]` items `[ready]` without human sign-off in BACKLOG.
