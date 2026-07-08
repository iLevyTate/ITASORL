# Next steps (Ralph action queue)

Prioritized work Ralph **may** pick when no P0/P1 bug is open. Each item must
be **evidence-backed** and **verifiable without inventing results**.

Mark status: `[ready]` Ralph can implement · `[blocked]` needs human · `[done]`

---

## Tier 1 — Clarify the replication gap (highest leverage)

| Status | Item | Rationale |
|--------|------|-----------|
| `[done]` | **CUDA/CPU + cross-run determinism test** (2026-07-08): `tests/test_experiment_b2.py::test_matched_pair_L0_bit_identical_on_device` + `test_readout_states_deterministic_across_runs`, parametrized over CPU + CUDA (CUDA skipped in CI). Within-device same-seed states are bit-identical on both devices; cross-device equality deliberately not asserted (FP reduction order differs). | Localizes 0.595 vs 0.523 to device/seed/code, not our readout, without a 4 hr sweep. Was BACKLOG P2. |
| `[done]` | **Script** `scripts/compare_expB2_artifacts.py` + Colab compare cell. | Run after Colab/local B-v2. |
| `[blocked]` | Local seed-0 full-scale re-run (300 updates) on lab GPU vs Colab T4 seed 0 (0.586). | Colab: `RUN_PROFILE = "b2_seed0"` in `notebooks/colab_gpu.ipynb`. |

---

## Tier 2 — Hardening & rigor (code, not long runs)

| Status | Item | Rationale |
|--------|------|-----------|
| `[done]` | `collect_pool` / early-death guard tests (2026-07-08): `test_collect_pool_excludes_early_deaths` (harsh E0 forces a mix; only full-length survivors returned, aligned) + `test_pooled_readout_too_few_survivors_guard` (stubbed <5-survivor pool -> all-NaN `too_few_survivors` result, no crash). | Ensures B-v2 pools are not silently biased under harsh metabolism. |
| `[done]` | Extreme latent → env action bounds test (2026-07-08): `test_agent_ac.py::test_env_action_bounded_for_extreme_latents` feeds raw latents up to +/-1e8 through `to_env_action`; asserts thrust in [0,1], turn in [-1,1], binaries in {0,1}, all finite. | Safety on actor-critic squash. |
| `[ready]` | Update `docs/FINDINGS.md` §7 item 1 to reference §9 replication (0.523) and retire "attempted" ambiguity if wording still implies open positive. | Docs aligned with canonical artifact. |

---

## Tier 3 — Scale & new experiments (human gate for GPU time)

| Status | Item | Rationale |
|--------|------|-----------|
| `[blocked]` | n = 10 B-v2 extension via `scripts/run_expB2_n10.sh`. | Needs free RAM + GPU + human; tightens CI on 0.523 vs 0.65. |
| `[blocked]` | Held-out fixed-dynamics / common-garden probe for B-v2. | Product decision on design; Ralph can draft API + stub in `experiment_b2.py` only if spec is written to BACKLOG Questions first. |
| `[blocked]` | L3 generative fingerprint scaffold. | Large feature; needs human scope in Questions before coding. |

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
