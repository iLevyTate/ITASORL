# Pre-Registration - ITASORL Experiment B-v3 (Regime-Coupled Incidental Detection)

**Status:** pre-registered *before* the confirmatory run. **Date:** 2026-07-01.
**Branch:** `expB2-regime-coupling`. Configuration below is frozen at commit time;
any later change is logged in section 12.

## 1. Background and the contrast

Experiment B-v2 asked whether a survival actor-critic incidentally encodes world
identity when the surrogate's drag follows a slow AR(1) wander (drift_mode=`ar1`). It
does not: pooled target AUROC ~0.52 vs a privileged oracle at ~0.99, and the null
survived survival pressure, a nonlinear probe, and a 4-hour replication
(`docs/FINDINGS.md` section 9). Two structural reasons, converging from the code and
from the domain-randomization / meta-RL literature (RMA Kumar 2021; RL^2 Duan 2016;
PEARL Rakelly 2019; domain-randomization theory ICLR 2022):

1. **Identifiability.** An AR(1) wander is not a persistent per-episode quantity, so a
   single trajectory does not identify a stable "world" to hold in the recurrent state.
   The authentic and surrogate worlds differ mainly in drag VOLATILITY at a shared
   central level, which a persistent linear direction cannot express.
2. **Policy relevance.** A slowly-wandering drag can be handled reactively (react to the
   current felt drag); staying alive never requires *remembering* which world you are in.

B-v3 changes exactly one thing: the surrogate's drag is now a **per-episode CONSTANT
offset** (drift_mode=`regime`) rather than a within-episode wander. A constant is
identifiable from one trajectory, and a persistently higher-drag world changes the
optimal foraging strategy for the whole episode, so representing it becomes
instrumentally useful. World identity is still never in the observation or the reward.

## 2. Hypothesis

- **H_B3 (primary):** with an identifiable, policy-relevant regime, the survival agent's
  recurrent state encodes world identity *above* the prediction-only and untrained
  baselines and *above* the SESOI.
- **Null:** pooled target ~0.50, no better than baselines - even an identifiable,
  policy-relevant dynamics regime is not incidentally encoded under readout-not-reward.

Either outcome is substantive. A positive result answers a *conditional* question (see
section 11): it does not resurrect the original L2-drift claim; it shows *what kind* of
dynamics difference survival will encode.

## 3. Agents - identical trunk and identical readout; only the objective differs

Unchanged from B-v2: `untrained` (mechanical floor), `predictor` (next-step prediction
on this trunk), `survival` (recurrent A2C + GAE + world-model decoder aux). "Regime
induces encoding" requires `survival > predictor > untrained` and survival >= SESOI.

## 4. World and the regime coupling (readout-not-reward preserved)

- `PatchOfEarthV0(drift_mode="regime")`. At each episode reset, if drift_sigma>0 the
  drag multiplier offset is drawn ONCE as `_drift_w = drift_sigma * U(regime_lo,
  regime_hi)`, clipped to (-0.8, 8.0), and held CONSTANT for the episode. Defaults
  `regime_lo=0.5, regime_hi=1.5`, so at drift_sigma=0.45 the surrogate runs the whole
  episode at drag * (1 + offset) with offset in [0.225, 0.675] (mean 0.45), bounded
  away from 0. Authentic (drift_sigma=0) is byte-identical to B-v2's authentic world.
- Same harsh metabolism, denser food, and potential-based food shaping as B-v2
  (`SURVIVAL_METAB` / `SURVIVAL_FOOD`); reported returns use the TRUE reward.
- World identity is never in the observation or the reward.

## 5. Readouts (episode = unit of inference; GroupKFold)

- **PRIMARY - pooled (Experiment B frame):** the `target` AUROC of a linear probe on
  `episode_features(h_t) = [mean h, final h]`, plus the additive `target_var` /
  `target_full` (dispersion) and per-probe `selectivity = target - shuffled` from the
  variance-probe work (PR #13). ~0.50 selectivity means no incidental encoding.
- **SECONDARY - matched-pair recurrent:** detectability of the artifact in the agent
  state; reported, not the headline.

## 6. Estimand and smallest effect of interest

- **Estimand:** per-episode selectivity-corrected AUROC of the linear world-identity
  probe (pooled readout), averaged over seeds.
- **SESOI:** claim incidental encoding only if the survival pooled target is
  **>= 0.65 AND exceeds both `predictor` and `untrained` by >= 0.05** (same bar as B-v2,
  so the two experiments are directly comparable).

## 7. Gates - ALL must pass before the survival target is interpreted

Identical to B-v2: (1) engagement (trained TRUE return beats random and scripted by
ENGAGE_MARGIN with lifetime not worse by LIFE_TOL), (2) L0 control (drift=0 pooled
target equivalent to 0.5 by TOST), (3) positive control (speed probe >= 0.75),
(4) leakage audit (reward-sum / length / lifetime each within 0.1 of 0.5).

## 8. Decision matrix

- **Encoding induced (conditional):** all gates pass AND survival target >= 0.65 AND
  > predictor+0.05 AND > untrained+0.05.
- **Strengthened negative:** all gates pass AND survival ~= 0.50 ~= predictor ~= untrained.
- **Uninformative:** engagement gate fails.

## 9. Fixed configuration (frozen at commit)

drift_mode `regime`; regime band `U(0.5, 1.5)`; drifts `[0.0, 0.45]`; seeds `[0,1,2]`
(power extension to `0..9` via `scripts/run_expB2_n10.sh`); updates `300`; `n_eps=16`;
`max_steps=80`; `hidden=96`; `ray_steps=5`; `shaping_coef=1.0`; pooled `n=110, steps=24`;
matched-pair `pairs=60, prefix=20, branch=24`. Optimizer Adam `lr=3e-4`, `gamma=0.99`,
`lam=0.95`, `ent_coef=0.01`, `vf_coef=0.5`, `wm_coef=1.0`.

## 10. Analysis plan

Mean +/- std over seeds. Primary contrast: survival vs predictor vs untrained pooled
target across drift, in regime mode. Report every gate. Same figure/JSON as B-v2
(`expB2_survival.png` / `expB2_results.json` in the run output dir). Also report the
sysid-aux CEILING (PR #14) as the capacity reference.

## 11. Key interpretive caveat

- **This is a different rung, not the L2 claim.** B-v2's L2 artifact is a *drift* (a
  temporal signature); B-v3's regime is a per-episode *level* offset. A positive B-v3
  result therefore answers "does survival encode an *identifiable, policy-relevant*
  dynamics regime", NOT "does survival encode L2 drift". Both are reported side by side.
- **Reactive vs representational** is partly resolved here by construction: a per-episode
  constant, probed across INDEPENDENT episodes, requires a persistent per-world direction,
  not a momentary reaction. A held-out common-garden probe remains cleaner future work.
- The result is conditional on this architecture, this world family, and this regime band.

## 12. Deviations from pre-registration

- **2026-07-18 (recorded retroactively by the methodology audit): the n = 3
  confirmatory run was not published at the time.** The pre-registered n = 3 run
  at the frozen seeds completed 2026-07-03 and read pooled survival target
  0.615 ± 0.083 (intermediate zone); it was recorded only in the 2026-07-03
  runner-design spec (`docs/specs/`), and FINDINGS reported the
  n = 10 power extension (0.610) directly. Conclusion-neutral - both runs land
  below the 0.65 bar and in the same zone - but the primary run should have
  been recorded here when it completed. Now logged in FINDINGS section 7.1.
- **2026-07-18 (same audit): estimand wording vs adjudication.** Section 5
  names "selectivity-corrected AUROC" as the estimand while section 6's
  decision sentence and the promoted artifacts adjudicate the raw pooled
  target; the raw target is what every published B-v3 verdict used. For a
  below-bar null this is the anti-conservative (honest) choice; recorded here
  so the discrepancy in the frozen text is on the record rather than silent.
- **2026-07-18 (recorded late; audit finding): post-freeze apparatus fix.**
  The regime-mode matched-pair channel silently ran with the drag offset
  disabled (a plain `set_state` restored the prefix's `drift_w = 0.0`,
  collapsing `mp_target` to exactly 0.5 in every cell); fixed the day after
  registration (2026-07-02, commit `a3fac1a`) by keeping the surrogate's
  reset-drawn `drift_w` across the restore. The n = 10 run (`820849f`)
  post-dates the fix, so its numbers are unaffected. Logged here because the
  freeze promise requires it, not because any published number changes.
- **2026-07-18 (same audit): unpromoted gates.** The n = 10 run's gate values
  (engagement, L0 TOST, speed positive control, leakage) were never promoted
  into `artifacts/expB2/bv3_n10_summary.json`; promoting them from the
  `fullruns/07062026` bundle is an open task before the B-v3 verdict is cited
  as gate-complete.

## 13. How to run

    # confirmatory (3 seeds), regime mode, dumping states for offline re-probing:
    python scripts/run_expB2.py --drift-mode regime --dump-states runs/bv3/states
    # via the full pipeline / Colab:
    python scripts/run_e2e.py --b2-drift-mode regime --b2-dump-states runs/bv3/states
    # capacity ceiling for reference:
    python scripts/run_expB2.py --drift-mode regime --sysid-aux
    # power extension (n=10) uses run_expB2.py --seeds 0..9 (see scripts/run_expB2_n10.sh)
