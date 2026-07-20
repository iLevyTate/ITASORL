# Pre-Registration - ITASORL Experiment L3 (Learned-World-Model Surrogate, Incidental Detection)

**Status:** pre-registered *before* the confirmatory run. **Date:** 2026-07-10.
**Gate:** the L2/pooled line is a confirmed negative under scale - B-v3 regime n=10
survival pooled 0.610 (90% CI [0.585, 0.634]) and the sysid-aux capacity ceiling
n=10 pooled 0.596 (90% CI [0.577, 0.616]) both sit entirely below the 0.65 bar, so
the pre-registered "proceed to L3 only if L2 holds under scale" condition is met.
Configuration below is frozen at commit time; any later change is logged in section 12.

## 1. Background and the contrast

Experiments B-v2 (L2 drift, AR(1)) and B-v3 (L2 regime, per-episode constant offset)
asked whether a survival actor-critic incidentally encodes world identity when the
surrogate differs from the authentic world by a *hand-specified dynamics parameter*
(the drag). It does not, and the negative is now near-architectural: even a trunk
supervised directly on drag (the sysid-aux ceiling) reads the pooled
persistent-direction probe at only ~0.60, while the matched-pair detectability channel
reaches ~0.70 - identity is decodable when forced in, but the pooled readout saturates
below the bar (`docs/FINDINGS.md` section 7.1, section 9).

L3 changes *what makes the surrogate a surrogate*. Instead of a hand-tuned drag
parameter, the surrogate world is produced by a **separate neural network trained to
imitate the authentic dynamics**. Its differences from the authentic world are the
**generative fingerprint** of that imitation: smoothing / blur, missing fine detail, a
characteristic learned noise texture, and mode collapse (rare events that should occur
never do) (`docs/ITASORL.md` L3, ladder section). This is a structurally different, more
realistic surrogate than a single dynamics knob, and it is the project's headline target
(`docs/ITASORL.md`).

## 2. Hypothesis

- **H_L3 (primary):** a from-scratch survival agent living across authentic and
  L3-surrogate worlds incidentally encodes world identity in its recurrent state *above*
  the prediction-only and untrained baselines and *above* the SESOI, read out but never
  rewarded.
- **Null:** pooled target ~0.50, no better than baselines - even a rich generative
  fingerprint is not incidentally encoded under readout-not-reward.

Either outcome is substantive. The result is *conditional on this surrogate's difficulty*
(section 7 oracle gate): it says what kind of substrate artifact survival will or will not
encode, not that no fingerprint could ever be encoded.

## 3. Agents - identical trunk and identical readout; only the objective differs

Unchanged from B-v2/B-v3: `untrained` (mechanical floor), `predictor` (next-step
prediction on this trunk), `survival` (recurrent A2C + GAE + world-model decoder aux).
"L3 fingerprint induces encoding" requires `survival > predictor > untrained` and
survival >= SESOI. The probe is identical across all three.

## 4. World and the L3 surrogate (readout-not-reward preserved)

**Signal targeted (decision, section 11):** the organism test targets **Signal A**
(latent geometry) - the agent lives in both authentic and L3-surrogate worlds and the
probe reads world identity from its recurrent state - keeping the "incidental,
readout-not-reward" framing identical to B-v2/B-v3. Signal B (prediction-residual of a
world model frozen on authentic then dropped into the surrogate) is used only inside the
section 7 oracle check, not as the organism headline.

**Surrogate construction:**

1. **Two-stage construction (C -> B), resolved after a code-level integration review
   (2026-07-10).** `G` is a small predictive world-model (`RecurrentWorldModel`) trained
   on authentic `PatchOfEarthV0` rollouts and then frozen. Because `G` predicts
   OBSERVATIONS while the survival world needs full internal STATE (energy, food, death),
   "wrap `G` as the World" is realized in two stages:
   - **Stage 1 - oracle gate (Option A, observation-channel).** Authentic physics is kept
     intact (survival stays well-defined); an L3-surrogate trajectory is an authentic
     rollout whose OBSERVATIONS are passed through frozen `G`, so the tell is `G`'s
     generative reconstruction fingerprint (blur, missing fine detail, mode collapse).
     This is what the section-7 Experiment-A oracle detects. Cheap; no change to the
     survival dynamics.
   - **Stage 2 - organism run (Option B, dynamics-level), CONTINGENT on the gate.** Only
     if Stage 1's oracle lands in [0.85, 0.95] do we build the dynamics-level surrogate:
     `G` (trained on logged STATE transitions) drives the world's dynamics, with
     energy/food/death computed from `G`'s predicted state, plugged into
     `itasorl/experiment_b2.py::make_world` as the new `l3` drift-mode where the L2 path
     lives. If `G`'s learned dynamics cannot sustain survival (engagement gate fails),
     that is logged in section 12 and the organism construction is re-scoped before any
     multi-seed run - a stated empirical risk, not assumed away.
   The authentic world (drift off) is unchanged and byte-identical to B-v2/B-v3 authentic
   in both stages.
2. **Difficulty is calibrated by outcome, not architecture (decision).** We choose the
   **smallest-capacity** `G` whose Experiment-A oracle detectability (section 7) lands in
   the pre-set band **AUROC in [0.85, 0.95]**: clearly detectable (the fingerprint exists)
   but not trivially separable like L2's ~0.99. Capacity is the *single* difficulty knob.
3. **Fixed data budget (decision).** `G` trains on a fixed budget of authentic rollouts
   (frozen in section 9); we do NOT trade data against capacity - capacity alone sets
   difficulty, so the surrogate's fidelity is reproducible.
4. Same harsh metabolism, denser food, and potential-based food shaping as B-v2/B-v3
   (`SURVIVAL_METAB` / `SURVIVAL_FOOD`); reported returns use the TRUE reward.
5. World identity (authentic vs G-surrogate) is never in the observation or the reward;
   only the probe ever sees it.

## 5. Readouts (episode = unit of inference; GroupKFold)

Identical harness to B-v3:
- **PRIMARY - pooled (Experiment B frame):** `target` AUROC of a linear probe on
  `episode_features(h_t) = [mean h, final h]`, plus additive `target_var` / `target_full`
  and per-probe `selectivity = target - shuffled`.
- **SECONDARY - matched-pair recurrent:** detectability of the fingerprint in the agent
  state from a bit-identical shared prefix; reported, not the headline.

## 6. Estimand and smallest effect of interest

- **Estimand:** per-episode selectivity-corrected AUROC of the pooled linear
  world-identity probe, averaged over seeds.
- **SESOI (decision - reuse 0.65):** claim incidental encoding only if the survival
  pooled target is **>= 0.65 AND exceeds both `predictor` and `untrained` by >= 0.05** -
  the same bar as B-v2/B-v3, so L3 is directly comparable to the earlier rungs.

## 7. Gates - ALL must pass before the survival target is interpreted

The four B-v3 gates, PLUS a new L3-specific manipulation gate that must pass FIRST:

0. **Oracle-detectability gate (L3-specific, decision - oracle-first).** BEFORE any
   organism run, an Experiment-A oracle (a privileged discriminator with direct access to
   observations / prediction residuals) must detect authentic vs G-surrogate at
   **AUROC in [0.85, 0.95]**. Below 0.85: the surrogate is too good, there is no
   fingerprint to encode - fix `G`, do not run the organism. Above 0.95: too easy
   (approaching L2's trivial regime) - increase `G`'s capacity. This gate both confirms the
   tell exists and freezes the difficulty.
1. **Engagement:** trained TRUE return beats random and scripted by ENGAGE_MARGIN with
   lifetime not worse by LIFE_TOL.
2. **L0 control:** authentic-vs-authentic pooled target equivalent to 0.5 by TOST.
3. **Positive control:** speed probe >= 0.75.
4. **Leakage audit:** reward-sum / length / lifetime each within 0.1 of 0.5.

## 8. Decision matrix

- **Encoding induced (conditional on this fingerprint):** gate 0 passes AND all gates pass
  AND survival target >= 0.65 AND > predictor+0.05 AND > untrained+0.05.
- **Strengthened negative:** gate 0 passes AND all gates pass AND survival ~= 0.50 ~=
  predictor ~= untrained. (The strongest form of the project's core claim: even a rich
  learned-surrogate fingerprint, verified detectable by an oracle, is not incidentally
  encoded under survival pressure.)
- **Uninformative:** gate 0 fails (surrogate mis-calibrated) or engagement fails.

## 9. Fixed configuration (frozen at commit)

- **Surrogate net `G`:** family = small recurrent predictor (GRU core). Fixed training
  config, frozen here: next-state MSE loss, Adam `lr=1e-3`, a fixed number of epochs to
  convergence (early-stop on held-out authentic MSE, patience 5), seed 0. The ONLY free
  knob is hidden size, swept ascending over `{8, 16, 24, 32, 48, 64}`; the FIRST size
  whose section-7 oracle AUROC lands in [0.85, 0.95] is FROZEN (selected value + oracle
  AUROC recorded in section 12 before any organism run).
- **Capacity-sweep fallback (no valid capacity in the grid).** Because AUROC decreases
  with capacity, the sweep can step OVER the band (e.g. hidden=16 gives 0.97, hidden=24
  gives 0.82). If no grid point lands in [0.85, 0.95]: (a) bisect hidden size between the
  two bracketing points; if still no fit, (b) hold the smallest capacity that is >= 0.85
  and REDUCE the authentic-data budget (a secondary difficulty knob) until oracle AUROC
  enters the band. Whichever path is used, the final `G` config is frozen in section 12
  and the organism run does not start until gate 0 passes. This guarantees gate 0 is
  reachable.
- **`G` training data budget:** fixed authentic-rollout budget frozen here (initial
  proposal: the same collection scale as B-v2 `collect_pool`, i.e. on the order of a few
  hundred authentic episodes); no data/capacity trade.
- **Organism config (mirrors B-v3):** drifts `[0.0, surrogate]`; seeds `[0,1,2]`
  confirmatory, power extension `0..9`; updates `300`; `n_eps=16`; `max_steps=80`;
  `hidden=96`; `ray_steps=5`; `shaping_coef=1.0`; pooled `n=110, steps=24`; matched-pair
  `pairs=60, prefix=20, branch=24`. Adam `lr=3e-4`, `gamma=0.99`, `lam=0.95`,
  `ent_coef=0.01`, `vf_coef=0.5`, `wm_coef=1.0`.

## 10. Analysis plan

Mean +/- std over seeds. Primary contrast: survival vs predictor vs untrained pooled
target across authentic vs G-surrogate. Report every gate including the oracle gate.
Same figure/JSON schema as B-v2/B-v3. Also report the sysid-aux CEILING for L3 as the
capacity reference, and the matched-pair detectability as the "decodable-when-forced"
comparison, exactly as for L2.

Rigor carried from the B-v3 audit (2026-07-10):
- **Honest seed-level CI.** The primary "clears / misses 0.65" adjudication uses a
  **t-based (or BCa) 90% CI on the per-seed AUROCs**, not the percentile-of-mean
  bootstrap alone (at n=10 the percentile-of-mean interval under-covers at the boundary).
  Report both; the decision follows the t/BCa interval.
- **Per-world survivorship.** Report authentic vs surrogate survivor counts / death rates
  per cell, so the pooled survivorship asymmetry (section 11) is quantified, not assumed.
- **Multiplicity.** The volatility readouts (`target_var`, `target_full`) are **secondary
  / exploratory**; only the level `target` enters the pre-registered SESOI decision. Any
  volatility-based claim is flagged exploratory and not multiplicity-corrected.

## 11. Key interpretive caveat

- **This is a different, higher rung, not the L2 claim.** A positive L3 result answers
  "does survival encode a *learned-world-model generative fingerprint*", not "does
  survival encode L2 drift". A negative is the strongest form of *detectable != learned*:
  the fingerprint is oracle-verified present (gate 0), yet the survival organism does not
  form a persistent world-identity direction from it.
- **Signal A vs Signal B.** The organism headline is Signal A (latent geometry from living
  in both worlds); Signal B (frozen-world-model prediction-residual) is the oracle-side
  check only. Conflating them would confuse detectability with incidental encoding.
- **Difficulty is a design choice, made explicit.** The oracle band [0.85, 0.95] fixes how
  hard the tell is; the result is conditional on that band, this architecture, this world
  family, and this surrogate net.
- **Fingerprint character is confounded with capacity, and we bound it rather than assume
  it away.** A low-capacity `G` and a high-capacity `G` can both land in the oracle band
  while producing *qualitatively different* artifacts (heavy blur vs mode collapse vs a
  learned noise texture). The band fixes *difficulty*, not artifact *type*. We therefore
  (a) record which fingerprint dominates at the frozen capacity (via the oracle's most
  informative feature channel) in section 12, and (b) run the organism test at TWO
  in-band capacities when the sweep yields two candidates, reporting whether the
  encoding verdict is stable across artifact type. A single-capacity result is reported
  as conditional on that artifact.
- **Pooled readout is Exp-B-comparable, not confound-clean.** As in B-v2/B-v3, the pooled
  probe drops early deaths per world, so a harder surrogate that kills more episodes
  yields a survivorship-selected surrogate pool. The matched-pair channel (bit-identical
  shared prefix, equal-length truncation) is the confound-clean detectability guard; the
  pooled target is read as directly comparable to Experiment B, not as the
  confound-controlled estimate. This run additionally reports **per-world survivor counts
  / death rates** (section 10) so the asymmetry is visibly bounded.

## 12. Deviations from pre-registration

- **2026-07-10, milestone-1 empirical finding (Stage-1 oracle calibration, GPU).** The
  Stage-1 observation-channel construction (Option A: authentic physics, observations passed
  through frozen `G`) was built and its oracle calibrated on a from-scratch `G` capacity sweep
  (hidden 8-128; one-step reconstruction AND open-loop imagination; n=200-300 authentic
  episodes; local GPU). Result: the artifact is **trivially detectable (oracle AUROC ~1.0) at
  every capacity and every imagination horizon**, and a matched sensor-noise / variance floor
  did NOT restore a calibratable gradient. Reason: a strong privileged oracle separates "real
  sensor data" from "any neural-net reconstruction of it" via `G`'s systematic per-dimension
  manifold, not via `G`'s fidelity - so capacity is not a working difficulty knob and the
  [0.85, 0.95] band is unreachable without deliberately weakening the oracle (which would game
  the gate). Per the pre-agreed escalation, the Stage-1 obs-channel construction is **RETIRED**
  and the dynamics-level construction becomes primary: the surrogate's observations are
  produced by the REAL sensor model applied to `G`'s predicted STATE, so observations live on
  the authentic manifold and the sole tell is `G`'s dynamics error, which capacity controls.
  Section 4 Stage 1 is superseded by this entry; the section-7 oracle gate now runs against the
  dynamics-level surrogate, with capacity re-validated as the difficulty knob before any
  organism run.
- **2026-07-10, GATE 0 MET (dynamics-level oracle calibration, GPU).** Built the dynamics-level
  construction: `G_motion` (learned velocity law, `(vel,a)->vel_next`, drag withheld) via
  `itasorl/surrogate_l3.py`, the `l3` world hook, and the L2-style residual oracle
  (`itasorl/experiment_a_l3.py`, matched pairs, exact per-step authentic law from the logged
  transitions). Capacity is a clean MONOTONE difficulty knob; the section-7 [0.85, 0.95] band is
  reachable. **Frozen gate-0 configuration:** sensor-noise floor **sigma=0.05**, capacity
  **hidden=8** -> oracle AUROC **0.890** (centered in-band; hidden=4 -> 0.940 also in-band;
  hidden>=16 -> <0.75). Leakage: MECHANICAL channels (length, metadata) clean at 0.5; `reward`
  legitimately leaks (~0.86 at hidden=4; different dynamics -> different movement cost ->
  different reward, the documented dynamics-rung consequence, docs/FINDINGS.md section 2.2). The
  residual oracle uses ONLY the velocity residual, so it does not exploit reward - BUT the reward
  coupling is a readout-not-reward CONSIDERATION for the organism run: the B-v2/B-v3 leakage
  audit + matched-pair equal-length truncation must be re-checked at the frozen difficulty
  (smaller dynamics error at hidden=8 should shrink the reward leak vs hidden=4). Gate 0 is met;
  the organism run may proceed with these controls reported.
- **2026-07-10, ORGANISM RUN n=3 (confirmatory) + an unresolved measurement discrepancy.** Ran
  `run_expB2 --drift-mode l3 --l3-hidden 8` (n=3 seeds, 300 updates, local GPU, `fullruns/l3_n3`).
  At drift 0.45 the pooled target was **survival 0.801 (90% CI [0.760, 0.842]), predictor 0.779,
  untrained 0.706**; L0 control clean (0.514); manipulation check: artifact survival-relevant.
  Primary H_B2 NOT met by the SESOI (survival only +0.022 over predictor, < 0.05). TWO problems
  make this verdict NOT trustworthy yet: (1) **high organism mechanical floor** - the untrained
  (mechanical) agent already reads 0.706, so there is little headroom to isolate incidental
  encoding; (2) a **re-calibration sweep** (untrained floor vs oracle AUROC across `G` capacity)
  shows the tension is structural: larger `G` drops the untrained floor toward chance (hidden 32
  -> 0.47, 128 -> 0.46) but the oracle signal COLLAPSES below the band (hidden 32 -> 0.70, 128 ->
  0.63), so a clean-floor-AND-detectable capacity does not robustly exist above hidden=8.
  UNRESOLVED: an isolated untrained-floor probe at hidden=8 gives **0.548 reproducibly** (identical
  on CPU and GPU training, so `G` is not the noise source), which CONTRADICTS the full run's 0.706
  and lies OUTSIDE the run's CI [0.66, 0.80]. The full pipeline and the isolated measurement
  disagree for an unidentified reason. **The L3 organism verdict is ON HOLD** pending
  reconciliation of that discrepancy (a focused debugging task, not a re-run); only then is a
  clean-floor difficulty and a real H_L3 adjudication meaningful.
- **2026-07-10, DISCREPANCY RESOLVED - a world-params bug; the confound was an artifact.** Root
  cause of the 0.548-vs-0.706 divergence: `run_expB2` runs in
  `P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)`, but `setup_l3_surrogate` trained
  `G_motion` on bare `WorldParams()` DEFAULTS - so `G` learned the WRONG world's dynamics and was
  deployed in another, inflating its systematic error and faking the high mechanical floor.
  Confirmed on GPU: `G`-trained-on-default read on `P` reproduces the run's 0.734 exactly;
  `G`-trained-on-`P` drops the untrained floor to 0.483 (chance). Fixed by passing `params=P` at
  both setup sites (regression test `test_g_motion_wrong_world_diverges_more`). **The n=3 organism
  result above (untrained 0.706, survival 0.801) is INVALID - a bug artifact - and is retracted.**
  Re-calibrated GATE 0 on the CORRECT world `P`: at hidden=8 the dumb/untrained floor is ~0.48
  (chance) at every capacity 8-32, and the oracle lands in-band at **sigma=0.02 -> AUROC 0.928**
  (leakage clean). So a clean L3 difficulty DOES exist - oracle-detectable fingerprint AND a
  chance-level organism mechanical floor. Frozen gate 0 (world `P`): **hidden=8, sigma=0.02,
  oracle 0.928, untrained floor 0.483.** Next: re-run the organism with the fix (`G` on `P`) for
  a real H_L3 adjudication - this time the mechanical floor is at chance, so survival-specific
  encoding can be isolated.
- **2026-07-10, CORRECTED ORGANISM RUN n=3 - a PRELIMINARY POSITIVE (not yet decisive).** Re-ran
  `run_expB2 --drift-mode l3 --l3-hidden 8` (n=3, 300 updates, local GPU, `fullruns/l3_n3_fixed`,
  commit `eeee512`) with `G` trained on the correct world. At drift 0.45 the pooled target is
  **untrained 0.482 +/- 0.055** (mechanical floor AT CHANCE - the fix holds; per-seed
  0.483/0.414/0.549), **predictor 0.573 +/- 0.057**, **survival 0.777 +/- 0.122** (per-seed
  0.853/0.636/0.841). This is the project's FIRST genuine survival-specific dissociation:
  survival beats predictor by +0.204 and untrained by +0.295 (both >> the 0.05 SESOI), the
  shuffled negative control is at chance for every arm (survival 0.556, so it is real signal not
  probe overfitting), matched-pair 0.906, speed positive control 0.966, engagement 100%,
  manipulation check survival-relevant. The pipeline verdict prints **H MET (encoding induced,
  conditional on gates)**. HONEST CAVEATS (do not oversell): the mean 0.777 clears 0.65, but the
  t-based 90% CI **[0.648, 0.906]** just STRADDLES the bar (the printed percentile bootstrap
  [0.705, 0.849] excludes it, but that interval under-covers at the boundary - same issue flagged
  for the ceiling); 1/3 seeds (0.636) is below 0.65; variance is high (sd 0.122); the L0 TOST is
  underpowered at n=3 (mean 0.514, not shown equivalent); and the dynamics-rung reward-coupling is
  not yet ruled out for the organism (needs the leakage audit + a held-out probe). VERDICT: a
  strong, real, encouraging POSITIVE SIGNAL - the first sign that L3 (a learned-dynamics
  fingerprint) reverses the L2 nulls - but NOT a settled result at n=3. The **n=10 power
  extension is running** (`fullruns/l3_n10`) to test whether the CI clears 0.65 decisively; the
  reward/held-out controls remain to be reported before any headline claim.
- **2026-07-12, AUDITED ORGANISM RUN n=10 + POST-HOC CONFOUND AUDITS - POSITIVE, PARTLY
  BEHAVIOR-MEDIATED.** Re-ran `run_expB2 --drift-mode l3 --l3-hidden 8` (n=10, seeds 0..9, 300
  updates, local GPU, `fullruns/l3_n10_audited`) with the pooled-path leakage audit added (PR #39). At
  drift 0.45 the survival pooled target is **0.752**, honest t-based 90% CI **[0.698, 0.807]** (EXCLUDES
  the 0.65 bar; per-seed 0.853/0.636/0.841/0.823/0.830/0.573/0.705/0.782/0.759/0.723; 8/10 seeds >=
  0.65), with **predictor 0.573** and **untrained 0.488** (mechanical floor at chance) - survival beats
  both baselines by well over the 0.05 SESOI, and the L0 authentic-vs-authentic control is **0.517**
  (chance). This is a determinism reproduction of the `fullruns/l3_n10` headline (identical 0.752).
  CONTROLS THAT PASS: (1) POOLED reward-leakage audit - world identity is NOT decodable from summed
  reward: **reward-leak AUROC mean 0.541, range [0.517, 0.559], clean 10/10 seeds** (peer-review item 1,
  RESOLVED). (2) Per-world survivorship - **0 early deaths in every pool (110/110 authentic and surrogate,
  all seeds)**, so dropping early deaths introduces no asymmetry (peer-review item 2, RESOLVED). (3)
  Nonlinear-probe check (post-hoc, on the dumped states): a random-forest probe on the UNTRAINED agent
  reads only **0.517** (its shuffled baseline 0.506; L0 untrained 0.483), so the survival-vs-untrained
  dissociation is NOT a linear-probe artifact - the world is genuinely near-chance-decodable in the
  untrained state even nonlinearly.
  CONTROL THAT PARTLY MEDIATES - BEHAVIORAL MEDIATION (post-hoc, on the dumped states): the agent moves and
  forages differently in the two worlds, so per-episode behavior alone (speed/energy/food/drag) decodes the
  world at **~0.69** (0.689 linear, 0.704 nonlinear). Controlling for behavior CLEANLY (behavior->state
  regression fit IN-FOLD, no leakage) leaves a behavior-independent world-signal of **~0.66** (0.676 linear
  basis, 0.659 quadratic; per-seed quad 0.71/0.58/0.71/0.69/0.73/0.54/0.62/0.70/0.67/0.64, 6/10 >= 0.65) -
  clearly above the 0.488 untrained floor but only AT the 0.65 bar. So behavior mediates ~0.09 of the 0.752
  headline; a real behavior-independent component survives, but it is weak-to-moderate, not the abstract
  world-identity direction the raw 0.752 suggests. (An in-SAMPLE residualization gave a deflated 0.56-0.63
  by over-removing via leakage; the in-fold ~0.66 is the defensible figure. Still a soft UPPER bound: only
  4 coarse per-episode MEANS were controlled, so per-timestep behavior would likely lower it further.) The
  reward audit passed because reward is one scalar; behavior is multivariate and a much stronger separator.
  VERDICT: a real, reward- and survivorship-controlled, nonlinear-robust POSITIVE - L3 (a learned-dynamics
  fingerprint) reverses the L2 nulls and the survival agent (uniquely) carries world-discriminative state,
  read out and never rewarded. But it is only PARTLY behavior-independent (~0.66 after clean behavior
  control, right at the 0.65 bar), so the strong "abstract world-identity direction at 0.752" reading is
  NOT supported; the honest headline is "survival-relevant state that differs by world, plus a modest
  (~0.66) behavior-independent world component." STILL OWED: (a) a
  richer per-timestep behavior control to tighten the behavior-independent estimate; (b) the pre-registered
  hidden=4 second-capacity replication (section 11); (c) the held-out/common-garden probe (a world-signal
  that transfers to an UNSEEN fingerprint is much harder to dismiss as behavior).
- **2026-07-12 - BEHAVIOR AUDIT MADE REPRODUCIBLE (code, no new runs).** The ad hoc mediation audit above
  is now canonical committed code (`itasorl/behavior_audit.py` + `scripts/audit_behavior_mediation.py`,
  in-fold controls, control properties unit-tested on synthetic ground truth). Re-run on the same dumps it
  reproduces the published survival d=0.45 figures exactly: target 0.752, behavior-only 0.689 linear /
  0.705 nonlinear (prose said 0.704; random-forest seed wiggle), controlled 0.676 linear / 0.659 quadratic
  (artifact: `artifacts/expB2/behavior_audit_l3_n10.json`). The dump now also persists PER-TIMESTEP
  behavior traces (`bta`/`bts`), and the script runs the strictly stronger per-timestep control when
  traces are present, with a decision rule fixed in advance
  (`docs/specs/2026-07-12-l3-behavior-audit-design.md`): survival per-timestep-controlled
  mean >= 0.65 strengthens the claim; [0.60, 0.65) weakens it to a below-bar trace; < 0.60 means largely
  behavior-mediated. Owed items (a) and (b) reduce to two human-launched runs
  (`scripts/README.md`, "L3 owed runs"): hidden=8 with traces (headline capacity) and hidden=4 n=10.
- **2026-07-13 - PER-TIMESTEP BEHAVIOR CONTROL: CLAIM STRENGTHENED (hidden=8 re-run with traces).** Re-ran
  the frozen hidden=8 n=10 protocol with trace-extended dumps (`fullruns/l3_h8_traces`, local GPU). The
  pipeline is deterministic end-to-end: every published figure reproduced exactly (survival target 0.752
  [0.704, 0.797]; behavior-only 0.689/0.705; per-episode controlled 0.676/0.659; L0 mean 0.517, ROPE
  equivalence accepted; reward-leak clean 10/10; 0 deaths, all pools 110/110). The pre-registered decision
  rule then evaluated the strictly stronger per-timestep control: survival resid_trace = **0.726** (90% CI
  [0.685, 0.765], 9/10 seeds >= 0.65; quadratic variant 0.721 [0.678, 0.760]) -> >= 0.65, STRENGTHENS, and
  unlike the per-episode estimate the CI EXCLUDES the 0.65 bar. Honesty checks on real data: the untrained
  agent's resid_trace is 0.498 (exact chance) even though untrained BEHAVIOR alone decodes the world at
  0.645, so the control neither manufactures nor spares signal; predictor resid_trace 0.574 (the
  survival-only dissociation is preserved under the control). Two notable structural facts: (1) the full
  behavior trace alone decodes the world at 0.803, BETTER than the state probe itself - behavior is highly
  world-discriminative; (2) the per-episode-mean control had been OVER-removing (0.676), exactly the
  attenuation the synthetic tests predicted, while the surgical per-timestep control leaves ~0.73. The
  honest headline improves to: reward- and survivorship-controlled, nonlinear-robust, with a
  behavior-independent world-signal of **~0.73** whose CI clears the pre-registered bar. Caveats: the
  residualization is linear/quadratic over phi = [b_t, b_(t-1), cummean(b)]; a full-history or nonlinear
  control could in principle remove more. Artifact: `artifacts/expB2/behavior_audit_l3_h8_traces.json`.
  STILL OWED: the hidden=4 second-capacity replication (section 11) and the held-out/common-garden probe.
- **2026-07-13 - HIDDEN=4 RUN UNINFORMATIVE (gate failure); GATE 0 RE-VALIDATED PER CAPACITY AND THE
  SECOND CAPACITY RE-FROZEN AT HIDDEN=7.** The owed hidden=4 n=10 run (`fullruns/l3_h4_traces`, local
  GPU) completed and FAILS its gates at drift 0.45: untrained mechanical floor **0.891** (not
  chance), pooled reward-leak 0.637 with the clean gate passing in **0/10 seeds**, engagement
  passing in only **30% of seeds** (the drift-trained policy barely recovers: train@0.45 eval@0.45
  return -1.069, vs -0.219 at hidden=8). Per the section-8 decision matrix the run is
  **UNINFORMATIVE** (gate-0 / engagement failure), NOT a negative; the L0 control (0.517, TOST and
  ROPE both accept equivalence) confirms the apparatus itself is fine. ROOT CAUSE: hidden=4's
  in-band status (oracle 0.940) was frozen from the 2026-07-10 calibration made on the WRONG world
  at sigma=0.05, BEFORE the world-params fix; the post-fix re-calibration froze gate 0 only at
  hidden=8 and swept the untrained floor over capacities 8-32, so hidden=4 was never re-validated
  on world `P` (its reward leak was itself anticipated in the gate-0 entry above). FIX: gate 0 is
  now a committed, runnable check (`scripts/run_expA_l3.py`) validating BOTH halves per capacity on
  world `P` at the frozen sigma=0.02 - the residual oracle (band [0.85, 0.95], mechanical leakage)
  AND the organism-side untrained floor (|target - 0.5| < 0.1 at drift 0.45, arm built exactly as
  `run_expB2` builds it) - with a hidden=8 regression check and a selection rule frozen in advance:
  the smallest in-band capacity below 8 with a clean floor (section-9 fallback (a), bisection
  between the bracketing capacities 4 and 8). RECALIBRATION RESULT
  (`fullruns/l3_gate0_recal/calibration.json`): hidden=8 regression EXACT (oracle 0.928, floor
  0.482); hidden=4 oracle 0.932 in-band but floor **0.896** (independently reproducing the organism
  run's 0.891 from 3 seeds); hidden=5 oracle 0.972 (OUT of band); hidden=6 oracle 0.946 in-band but
  floor 0.647; **hidden=7 oracle 0.922 in-band, mechanical leakage clean, floor 0.566** -> second
  capacity RE-FROZEN at **hidden=7**. Honest note: the hidden=7 floor (0.566) is elevated relative
  to hidden=8 (0.482) though within the frozen tolerance; the SESOI already requires survival >
  untrained + 0.05, so the elevated floor is priced into adjudication. The hidden=7 n=10 organism
  run with trace dumps is running (`fullruns/l3_h7_traces`); its result and the per-timestep
  behavior audit will be recorded here when complete.
- **2026-07-14 - SECOND-CAPACITY REPLICATION AT HIDDEN=7 (n=10): THE BEHAVIOR-INDEPENDENT SURVIVAL
  SIGNAL REPLICATES (~0.72, CI clears the bar) BUT THE SURVIVAL-VS-PREDICTOR DISSOCIATION DOES NOT -
  THE FULL "ENCODING INDUCED" VERDICT IS ARTIFACT-CONDITIONAL.** The re-frozen hidden=7 n=10 run
  (`fullruns/l3_h7_traces`, local GPU, frozen protocol, trace dumps) completed with ALL gates passing:
  engagement 10/10 seeds at drift 0.45 (pooled train@0.45 eval@0.45 return -0.734, a partial recovery
  sitting between hidden=8's -0.219 and hidden=4's failing -1.069; the per-seed engagement criterion
  passes in every seed), L0 control 0.517 (TOST p=0.010 and ROPE P=0.999 both accept equivalence),
  speed positive control 0.959, pooled reward-leak 0.567 clean in 10/10 seeds, 0 deaths in every pool
  (110/110 both worlds, all seeds), and the untrained mechanical floor pooled at **0.586** (dev 0.086,
  inside the frozen <0.1 tolerance and matching the recalibration's 0.566, though per-seed it is
  violated in s3 0.732 and s9 0.683 with s6 0.605 marginal). PRIMARY at drift 0.45: survival pooled
  target **0.737**, 90% CI [0.688, 0.780], 8/10 seeds >= 0.65 (per-seed
  0.540/0.748/0.820/0.799/0.696/0.623/0.749/0.768/0.770/0.852) - above the 0.65 bar and clearing
  untrained (0.586) by +0.151, BUT **predictor reads 0.714** [0.687, 0.740], so survival leads the
  predictor by only +0.023 and the section-8 requirement (> predictor + 0.05) is NOT met: the clean
  hidden=8 dissociation (survival 0.752 vs predictor 0.573) does not replicate at hidden=7. BEHAVIOR
  AUDIT (`artifacts/expB2/behavior_audit_l3_h7_traces.json`, frozen decision rule): survival
  resid_trace = **0.722** (90% CI [0.678, 0.763], 8/10 seeds >= 0.65; quadratic 0.704) -> >= 0.65
  with the CI excluding the bar, an almost exact replication of hidden=8's 0.726 - the
  behavior-independent survival world-signal is STABLE across the two in-band artifacts. The
  dissociation is not: predictor resid_trace 0.691 (vs 0.574 at hidden=8) and untrained resid_trace
  0.579 (vs 0.498 = exact chance at hidden=8). READING (per the section-11 two-capacity clause): the
  hidden=7 artifact is qualitatively coarser - mechanically leakier (untrained floor 0.586/resid
  0.579) and far more behaviorally salient (the behavior trace alone decodes the world at 0.762-0.796
  in ALL arms, including untrained, vs 0.645 untrained at hidden=8) - so at this capacity every
  trained agent picks the fingerprint up and the survival-SPECIFIC part of the claim is conditional
  on the subtler hidden=8 artifact. The cross-capacity finding that survives both runs is: a
  reward-clean, survivorship-clean, behavior-independent world-signal of ~0.72 in the survival
  agent's state at both frozen capacities. Secondary notes: matched-pair survival mp_target 0.814;
  volatility readout target_var 0.706 / target_full 0.763 (above the 0.65 bar, consistent with the
  volatility-encoding observation on earlier rungs). LAST OWED ITEM, the held-out/common-garden probe
  (a world-signal that transfers to an UNSEEN fingerprint), is now run and recorded in the entry below.
- **2026-07-14 - HELD-OUT FINGERPRINT + COMMON-GARDEN PROBE (n=10): THE WORLD-SIGNAL GENERALIZES TO AN
  UNSEEN SAME-RECIPE FINGERPRINT (TRANSFER POSITIVE) BUT DOES NOT SURVIVE A COMMON-GARDEN CONTROL
  (REACTIVE, NOT A PERSISTENT REPRESENTATION).** Two readout-only evaluation channels on one frozen
  hidden=8 training run (`fullruns/l3_h8_heldout`, local GPU, frozen protocol, spec
  `docs/specs/2026-07-14-l3-heldout-common-garden-probe-design.md`); no change to training,
  the surrogate family, or the pre-registered headline probe. GATES REPRODUCED (third independent
  determinism check): survival pooled target **0.752** (90% CI [0.704, 0.797]), byte-matching the
  published hidden=8 headline; L0 control 0.517 (TOST p=0.010, ROPE P=0.999 both accept equivalence);
  pooled reward-leak 0.541 clean in 10/10 seeds; 0 deaths in every pool (110/110 both worlds, all
  seeds). CHANNEL 1, UNSEEN-FINGERPRINT TRANSFER: the world-identity direction fit against the trained
  hidden=8 fingerprint, frozen, then scored on a FRESH authentic pool vs the held-out hidden=7
  fingerprint the agent never trained against. Survival `transfer_target` = **0.773** (90% CI
  [0.722, 0.824], 9/10 seeds >= 0.65), predictor 0.633 (3/10), untrained mechanical floor 0.569
  (0/10). Frozen decision rule (survival >= 0.65 AND > untrained + 0.05 = 0.619) PASSES on both
  clauses -> GENERALIZES beyond the single trained fingerprint instance. Scope caveat, stated in the
  spec and honored here: hidden=7 is the SAME surrogate recipe at a different capacity, not a
  different surrogate family; cross-recipe transfer is out of scope for this run. CHANNEL 2, COMMON
  GARDEN: 20-step prefix in either the authentic or the hidden=8 surrogate world, then both groups
  continue under IDENTICAL authentic dynamics for a 24-step tail; the probe reads tail-only state,
  labels = prefix world, GroupKFold. Survival `cg_tail_target` = **0.557** (90% CI [0.492, 0.622],
  1/10 seeds >= 0.65), predictor 0.409, untrained 0.377. Frozen decision rule (survival >= 0.65 AND >
  untrained + 0.05) FAILS on the first clause. DECAY CHECK on the last 8 tail steps: survival
  `cg_latetail_target` = **0.492** (90% CI [0.431, 0.553]), at chance -> the prefix-world signal
  washes out along the shared tail. READING (frozen rules applied): the L3 world-signal is NOT an
  overfit to the one artifact instance `G_0` (transfer generalizes), but once the felt dynamics are
  made identical it does not persist in tail-only state - so the signal reads as REACTIVE tracking of
  the currently-felt dynamics, not a persistent stored world-identity representation. This is an
  informative negative for the representational reading (spec: "not a gate failure"), and it resolves
  the long-standing reactive-vs-representational ambiguity (section-11 caveats) toward reactive. This
  was the last owed post-hoc item; the L3 arc is complete. Provenance addendum (2026-07-16): the
  behavior audit re-run on this bundle's fresh `--save-agents` dumps reproduces resid_trace 0.726
  (90% CI [0.685, 0.765], 9/10 seeds) identically; per-seed values committed as
  `artifacts/expB2/behavior_audit_l3_h8_heldout.json`. Tooling: `audit_behavior_mediation.py` skips
  the heldout sibling dumps (`*_h7transfer.npz`, `*_cg.npz`) it does not parse.

- **2026-07-14 - REVERSE-TRANSFER RUN ANNOUNCED (frozen BEFORE launch).** The spec's staged
  follow-up condition is met (forward transfer positive), so the reverse direction runs next:
  train at hidden=7, hold out the SUBTLER hidden=8 fingerprint
  (`fullruns/l3_h7_heldout`, n=10, frozen protocol, `--save-agents`, trace dumps). Decision
  rules are IDENTICAL in form to the forward spec and frozen here: transfer - survival
  transfer_target >= 0.65 AND > untrained transfer + 0.05 -> the world-signal generalizes to a
  subtler unseen fingerprint (the stricter version of the 0.773 claim); common garden - same
  form, second data point on persistence at the coarser training artifact. INTERPRETATION
  LIMIT, stated in advance: at hidden=7 the survival-vs-predictor dissociation did not hold
  (two-capacity entry above), so this run reads the survival arm against the untrained floor
  only and cannot support any survival-SPECIFICITY claim. Determinism expectation: the
  standard-probe half must reproduce the `fullruns/l3_h7_traces` table exactly (survival 0.737
  [0.688, 0.780]); any deviation invalidates the run. Result to be recorded here when complete.

- **2026-07-16 - REVERSE-TRANSFER RUN RECORDED (completed 2026-07-15): THE COARSE-TRAINED
  DIRECTION READS THE SUBTLER UNSEEN FINGERPRINT ONLY PARTIALLY - THE FROZEN RULE FAILS
  (INFORMATIVE NEGATIVE); HELD-OUT TRANSFER IS DIRECTION-DEPENDENT.** Executes the 2026-07-14
  freeze above (`fullruns/l3_h7_heldout`, n=10, `--save-agents`; per-seed summary committed as
  `artifacts/expB2/heldout_l3_h7_reverse_summary.json`). DETERMINISM GATE PASSES: the
  standard-probe half reproduces the `fullruns/l3_h7_traces` table exactly - survival pooled
  target 0.737 (boot 90% CI [0.688, 0.780]), L0 0.517 (TOST p=0.010 / ROPE accept equivalence),
  reward-leak 0.567 clean 10/10, engagement 10/10, 0 deaths per pool - so the run is valid per
  the freeze. TRANSFER (train hidden=7, hold out the subtler hidden=8): survival
  `transfer_target` = **0.638** (t 90% CI [0.600, 0.676], 4/10 seeds >= 0.65) vs untrained
  mechanical floor **0.525** and predictor 0.603. The frozen rule (>= 0.65 AND > untrained +
  0.05 = 0.575) FAILS on the absolute bar; the floor-margin clause alone passes (+0.063).
  Adjudicated NEGATIVE per the frozen rule. COMMON GARDEN: survival `cg_tail_target` = **0.598**
  (t 90% CI [0.547, 0.649], 4/10 seeds >= 0.65), decaying to 0.489 on the last-8 window;
  predictor 0.504, untrained 0.456. Rule FAILS -> REACTIVE, a second independent data point
  matching the forward run. READING (frozen rules applied; the freeze's interpretation limit
  honored - no survival-specificity claim at hidden=7, and predictor transfer 0.603 indeed sits
  near survival 0.638): held-out transfer is DIRECTION-DEPENDENT. Fit on the subtle hidden=8
  fingerprint the direction reads coarser unseen artifacts (same-recipe 0.773, cross-recipe
  0.684); fit on the coarse hidden=7 fingerprint it reads the subtler one only partially (0.638,
  above the floor but below the bar). The published "fingerprint-GENERAL" wording is qualified
  accordingly in FINDINGS 10.6 and README.

- **2026-07-16 - RESEARCH-INTEGRITY AUDIT: FREEZE-TIMING NOTE FOR THE HIDDEN=7 SELECTION.** A
  commit-level audit of this log found: the capacity-fallback RULE was frozen in the original
  pre-registration commit (`0217263`, 2026-07-10), strictly before the hidden=4 gate failure it was
  later applied to. The specific "RE-FROZEN at hidden=7" statement, however, landed in the same
  commit as the hidden=7 n=10 result artifact (`85229c2`, 2026-07-14), so the capacity selection and
  its result are commit-simultaneous rather than sequential. The honest reading: hidden=7 was selected
  by a rule frozen before the failure, but the selection itself has no committed timestamp preceding
  the run. The per-timestep behavior-control spec has clean sequential timing (spec `1b5985e`
  2026-07-12, results `bf8a74e` 2026-07-13). Every published number was re-verified against committed
  per-seed artifacts by `scripts/audit_stats_recheck.py` (117 checks, all passing); all 15 cited arXiv
  IDs were resolved against arxiv.org with matching titles and authors. See `docs/AUDIT_2026-07.md`.

- **2026-07-16 - CROSS-RECIPE TRANSFER PROBE (n=10, readout-only): THE WORLD-SIGNAL READS A
  DIFFERENT SURROGATE FAMILY (RULE PASSES); THE CONSTANT-DRAG FAMILY IS UNCALIBRATABLE AND DROPPED.**
  Executes the pre-registered spec `docs/specs/2026-07-15-l3-crossrecipe-transfer-probe-design.md`
  (frozen before any run; PR #51) against the SAVED `fullruns/l3_h8_heldout` agents; no training
  anywhere. Committed artifact: `artifacts/l3_crossrecipe/summary.json`. GATE-0 CALIBRATION: the
  `G_rff` family's round-1 D sweep {8..128} stepped over the band (0.975 at D=64, 0.624 at D=128);
  the pre-registered bisection fallback landed **D=80** in-band (oracle 0.887, leakage pass,
  untrained floor 0.538). The `G_cd` family has an EMPTY calibration window: by the eps where its
  oracle reaches the band (eps=3.2: 0.854; eps=6.4: 0.907) the untrained mechanical floor exceeds
  0.6 (0.859, 0.913) - a drag-coefficient bias large enough to detect is felt so grossly that any
  recurrent state separates the worlds. DROPPED per the pre-stated rule (recorded, no transfer
  claim, no penalty to the primary). Both cd calibration double-runs produced byte-identical JSONs.
  INTEGRITY GATE (fourth determinism check): all 60 reloaded agents regenerated their standard
  pools bit-identically against the saved dumps and the drift-0.45 pooled survival mean reproduced
  the published **0.752** exactly. CHANNEL (PRIMARY, `G_rff` D=80): survival `transfer_rff_target`
  = **0.684** (boot 90% CI [0.657, 0.710]; t 90% CI [0.654, 0.715], lower bound above the bar;
  7/10 seeds >= 0.65, per-seed min 0.584), predictor 0.574 [0.554, 0.593] (0/10), untrained
  mechanical floor 0.548 [0.538, 0.557] (0/10). Frozen rule (survival >= 0.65 AND > untrained +
  0.05 = 0.598) PASSES on both clauses, machine-checked in the runner aggregate (`rff_rule_pass`
  true, margin +0.034). READING: the world-identity direction fit against the trained MLP
  fingerprint reads a different-recipe fingerprint (cosine-basis ridge texture) the agent never
  lived with, survival-specifically; attenuated versus same-recipe transfer (0.773 -> 0.684), as
  expected for a farther family. Scope, stated honestly: the reactive interpretation of the
  common-garden entry above is UNCHANGED - this extends the generality of the reactive world-signal
  across surrogate recipes, it does not revisit persistence. The cd drop is itself informative:
  coefficient-bias artifacts cannot be made subtle-but-detectable in world P, independently
  motivating the learned-texture construction of the L3 rung.

- **2026-07-18 - CORRECTION (OUTCOME-BLIND): COMMON-GARDEN CHANNEL RE-SCORING UNDER THE
  FIXED ESTIMATOR.** The 2026-07-14/16 held-out and reverse-run adjudications' common-garden
  channel (cg_tail 0.557 / late-tail 0.492; reverse 0.598/0.489) was computed with the
  pre-fix pair-splitting `cg_probe` (per-episode CV groups; bias toward AUROC 0 whenever the
  surviving pair count is not a multiple of 5 — fixed in commit `1633bca`; see FINDINGS
  13.C). The committed artifacts carry the bias signature (drift-0 cg floors 0.001-0.27;
  pair counts 96-110). The cg channel is being re-scored from the saved `_cg.npz` tail dumps
  with the fixed estimator (`scripts/reanalyze_cg_states.py`); the frozen decision rules are
  UNCHANGED and will re-adjudicate the corrected numbers. The transfer channel (frozen-fit
  train/test, no CV) and the mp channel (pair counts 60/25, fold-safe; no pair drops) are
  unaffected and stand. This entry is recorded BEFORE the re-score's outcome is known; the
  "REACTIVE" adjudication is provisional until it lands.

- **2026-07-18 - OPERATIVE-ESTIMAND ERRATA.** Section 6 registers the "selectivity-corrected
  AUROC" as the estimand; every adjudication (0.752, 0.737, 0.726, 0.722 vs the 0.65 bar)
  used the raw pooled/resid target with the shuffled probe as a gate-style control. No
  verdict could have flipped (shuffled at chance throughout); the operative definition is
  the raw target adjudicated at the bar, selectivity reported as a control.

## 13. How to run (milestones, in order)

1. **Build + calibrate the surrogate.** Train `G` on authentic rollouts; wrap as a World;
   run the Experiment-A oracle; ascend capacity until oracle AUROC in [0.85, 0.95]; FREEZE
   and record in section 12. (CPU-friendly; no GPU sweep.)
2. **Gate check.** Only if gate 0 passes, proceed. Otherwise re-tune `G`.
3. **Organism run** (GPU, mirrors B-v3), dumping states for offline re-probing:

       python scripts/run_expB2.py --drift-mode l3 --dump-states runs/l3/states
       python scripts/run_e2e.py --b2-drift-mode l3 --b2-dump-states runs/l3/states
       # capacity ceiling reference:
       python scripts/run_expB2.py --drift-mode l3 --sysid-aux
       # power extension (n=10) via the seeds 0..9 profile

   (The `l3` drift-mode and the `G` World are new code, gated behind this preregistration;
   no multi-hour organism sweep before milestone 1's oracle gate passes.)
