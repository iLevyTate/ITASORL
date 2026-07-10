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
  episodes; RTX 4050). Result: the artifact is **trivially detectable (oracle AUROC ~1.0) at
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
  `run_expB2 --drift-mode l3 --l3-hidden 8` (n=3 seeds, 300 updates, RTX 4050, `fullruns/l3_n3`).
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
