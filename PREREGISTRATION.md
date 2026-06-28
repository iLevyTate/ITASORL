# Pre-Registration — ITASORL Experiment B-v2 (Survival-Coupled Incidental Detection)

**Status:** pre-registered *before* the confirmatory run. **Date:** 2026-06-28.
**Branch:** `expB2-survival-coupling`. Configuration below is frozen at commit time;
any later change is logged in §12.

## 1. Background and the contrast

Experiment B trained an agent *only* to predict its sensory stream and asked whether
world identity is incidentally decodable from its recurrent state. It is not
(pooled target AUROC ≈ 0.50), despite a privileged oracle detecting the same L2
artifact at ≈ 0.99 (Experiment A). FINDINGS conjectures: *what the objective does not
require, the agent does not represent.* B-v2 changes exactly one thing — the agent now
**acts to stay alive in a world whose dynamics drift** — and asks whether incidental
encoding now emerges. Survival depends on coping with the drifting drag, so modelling
the dynamics becomes instrumentally useful.

## 2. Hypothesis

- **H_B2 (primary):** a survival actor-critic encodes world identity in its recurrent
  state *above* the prediction-only and untrained baselines and *above chance*.
- **Null:** pooled target ≈ 0.50, no better than the baselines — readout-not-reward
  bites even under survival pressure.

Either outcome is a substantive, reportable result.

## 3. Agents — identical trunk and identical readout; only the objective differs

| agent | objective | role |
|-------|-----------|------|
| `untrained` | random init (normalizer warmup only) | mechanical floor — drift perturbs inputs, so any recurrent code separates the worlds somewhat |
| `predictor` | next-step prediction, scripted policy | Experiment B's objective on this trunk (prediction, no survival) |
| `survival` | recurrent A2C + GAE (+ world-model decoder aux) | acts to stay alive under drifting dynamics |

"Survival induces encoding" requires `survival > predictor > untrained`.

## 4. World and the survival coupling (readout-not-reward preserved)

- `PatchOfEarthV0` with **harsh metabolism** (`E0=1.0, basal_E=0.4`): a non-forager
  starves in ≈ 50 steps, so staying alive *requires* foraging — and foraging
  efficiency depends on the L2-drifting drag that governs movement.
- **Denser food** (`n_pellets=24, reach=0.08, pellet_r=0.03`) so the eat-reward is not
  impossibly sparse for a from-scratch policy.
- **World identity is never in the observation or the reward.** Reward is the native
  homeostatic survival signal only.
- **Potential-based food-approach shaping** (`Φ = −dist to nearest pellet`) aids
  learning only; it provably preserves the optimal policy, is identical in authentic
  and surrogate worlds, and is *not* world identity. It enters the training reward
  only; reported returns/lifetimes use the TRUE reward.

## 5. Readouts (episode = unit of inference; GroupKFold)

- **PRIMARY — pooled (Experiment B frame):** independent authentic (`drift=0`) vs
  surrogate (`drift=d`) episodes, fixed length, drop early deaths. Probe a *persistent*
  world-identity direction in `episode_features(h_t) = [mean h, final h]`, reusing
  `experiment_b.episode_features` / `probe_auroc` verbatim. ≈ 0.50 ⇒ no incidental
  encoding (directly comparable to Exp B).
- **SECONDARY — matched-pair recurrent:** shared authentic prefix, branch authentic vs
  drift from a bit-identical state. This measures *detectability* of the artifact in
  the agent state, not persistent encoding; reported, but not the headline.

## 6. Estimand and smallest effect of interest

- **Estimand:** per-episode AUROC of the linear world-identity probe (pooled readout),
  averaged over seeds.
- **SESOI:** claim incidental encoding only if the survival pooled target is
  **≥ 0.65 AND exceeds both `predictor` and `untrained` by ≥ 0.05**.

## 7. Gates — ALL must pass before the survival target is interpreted

1. **Engagement:** the trained agent out-survives random and scripted policies (longer
   mean lifetime AND higher true return). Otherwise the run is **uninformative**.
2. **L0 control:** at `drift=0` the pooled target is equivalent to 0.5 by TOST
   (margin ±0.05) — proves the readout manufactures no signal.
3. **Positive control:** the speed probe is high (≥ 0.75) — the state is probeable.
4. **Leakage audit:** world identity is NOT decodable from reward-sum / episode-length
   / lifetime (each within 0.1 of 0.5) — the target reads the artifact, not "I lived
   longer in world X."

## 8. Decision matrix

- **Encoding induced:** all gates pass AND survival target ≥ 0.65 AND > predictor+0.05
  AND > untrained+0.05.
- **Strengthened negative:** all gates pass (incl. engagement) AND survival ≈ 0.50
  ≈ predictor ≈ untrained.
- **Uninformative:** engagement gate fails.

## 9. Fixed configuration (frozen at commit)

drifts `[0.0, 0.45]`; seeds `[0,1,2]`; updates `300`; `n_eps=16`; `max_steps=80`;
`hidden=96`; `ray_steps=5`; `shaping_coef=1.0`; pooled `n=110, steps=24`; matched-pair
`pairs=60, prefix=20, branch=24`. Optimizer Adam `lr=3e-4`, `gamma=0.99`, `lam=0.95`,
`ent_coef=0.01`, `vf_coef=0.5`, `wm_coef=1.0`. Hyperparameters were tuned on a
`drift=0.45` de-risk and are frozen for the confirmatory run.

## 10. Analysis plan

Mean ± std over seeds. Primary contrast: survival vs predictor vs untrained pooled
target across drift. L0 equivalence by TOST. Report every gate. Figure: |target−0.5|
vs drift per agent (`expB2_survival.png`); raw metrics in `expB2_results.json`.

## 11. Key interpretive caveat

- The matched-pair readout measures **detectability** (the artifact perturbs inputs of
  any recurrent net); the pooled readout measures a **persistent direction** =
  incidental encoding. Only the latter answers H_B2.
- **Reactive vs. representational:** the agent may adapt to *felt* drag reactively
  without *classifying* the world. The pooled probe across independent episodes is the
  conservative test — a consistent direction implies more than momentary reaction. A
  common-garden / held-out fixed-dynamics probe is the cleaner separation and is future
  work.
- The result is conditional on this architecture, this world family, and L2 only.

## 12. Deviations from pre-registration

(none yet — log any post-commit change here)
