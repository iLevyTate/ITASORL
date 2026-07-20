# Pre-Registration - ITASORL Experiment C (Emergence of Detection under Selection)

**Status:** FROZEN at commit `0887200` (2026-07-16), pre-dating the pilot. The pilot's
recorded result was invalidated by two measurement defects (FINDINGS 13.C); a re-run on
fixed code is pending and is additionally governed by section 14 (Amendments) below,
which must be ratified and committed BEFORE the re-run launches.

**Date drafted / decisions resolved:** 2026-07-16.

## 1. Background and the contrast

Experiments A, B, B2, the B-v2/B-v3 survival-coupling arc, and the full L3 arc all
asked one within-life question: does a from-scratch agent, trained by gradient
descent with a survival reward and never rewarded to detect, incidentally encode
world identity (authentic vs surrogate)? The L3 answer was a qualified positive that
then narrowed under scrutiny: the world-signal is reward-clean and
survivorship-clean, but it reads as REACTIVE tracking of the currently-felt dynamics
(the common-garden probe washed it out along a shared tail), and the
survival-vs-predictor dissociation is conditional on the subtler in-band artifact
(`docs/PREREGISTRATION_L3.md` section 12, 2026-07-14 entries).

Experiment C changes the mechanism that could produce detection. Instead of
within-life gradient learning, it applies **Darwinian selection across generations**:
individuals that behave world-appropriately survive and reproduce, their (mutated)
descendants inherit whatever made them fit, and the question is whether a **persistent,
heritable world-identity detection capacity EMERGES over generations** where
within-life learning alone produced only a reactive signal. The spec frames C as
"population dynamics, mortality, and generations measure detection capacity over time
against a control group" (`docs/ITASORL.md`), and reproduction is defined in
`docs/ITASORL_world_spec.md` section 7 ("above an energy threshold, spawn offspring
with a mutated genome at an energy cost. Disabled for A/B").

The contrast that makes the result interpretable is between two populations run under
identical world exposure:

- **Treatment (world-coupled selection):** world identity is made fitness-relevant, via
  a surrogate-specific payoff (section 4). An individual that acts as if it knows which
  world it is in survives and reproduces more.
- **Control (world-neutral selection):** identical dynamics, identical reproduction,
  identical mutation, but world identity is decoupled from fitness (the surrogate-specific
  payoff is removed or randomized). Any rise in detection capacity here is not driven by
  selection on world identity.

"Detection emerged under selection" requires the treatment population to gain
detection capacity across generations AND to gain it by more than the control.

## 2. Hypothesis

- **H_C (primary):** in the world-coupled treatment population, the common-garden
  detection AUROC of the population's internal state rises across generations, and its
  gain from generation 0 to the final generation exceeds the world-neutral control
  population's gain by at least the SESOI (section 6).
- **Null:** treatment gain ~= control gain ~= 0 - selection over these generations does
  not build a heritable detection capacity beyond whatever the apparatus starts with.

Either outcome is substantive. A positive says selection can manufacture a persistent
detection capacity that within-life learning (L3) did not; a null says even fitness
pressure on world identity, over this horizon and this population size, does not.

## 3. Unit of selection and agents (RESOLVED 2026-07-16: Option P, pure neuroevolution)

What mutates and is selected determines what "detection capacity" means. The chosen design
(section 12, fork 1) is **Option P** with within-life gradient learning **OFF**:

- **Option P (neuroevolution of policy weights) - CHOSEN.** The agent's recurrent policy
  weights are the heritable material; offspring inherit a mutated copy (Gaussian perturbation
  on the weight vector). Within-life gradient learning is OFF (pure neuroevolution), so a
  generational gain in detection capacity is inherited, not learned within a lifetime, and the
  treatment-minus-control contrast attributes emergence cleanly to selection. This also makes C
  a genuinely different mechanism from A/B/L3, all of which were gradient-trained: a positive
  cannot be explained by the same learning that produced L3's reactive signal. Detection is read
  out from the recurrent state on the common-garden panel, exactly as in B/L3. This is the most
  direct test of "does a persistent detector emerge".
- **Option G (genome physiology only) - NOT CHOSEN.** The section-6 genome (body size, thrust,
  basal metabolism, sensory acuity, diet, thermal preference) mutates; the policy is fixed or
  learns identically in both arms. Closer to the biological "digital organism" framing and
  cheaper, but a physiology genome cannot store a world-identity direction directly - the
  readout would be behavioral, so this tests emergence of world-appropriate BEHAVIOR, not of an
  internal representation. Recorded as the alternative; not run unless Option P is inconclusive.
- **Option PG (both) - deferred.** Most realistic, most expensive, most confounded; deferred
  unless P shows signal.

**Consequence of OFF (fork 2, resolved).** With no within-life learning, the "predictor" arm of
the prior rungs (a next-step-prediction gradient objective) has no meaning here; the per-generation
baselines are the **untrained/random-weight floor** (a freshly sampled policy, the mechanical floor)
and the **generation-0 evolved population** (the pre-selection starting point). The emergence curve
is read against both.

## 4. World, surrogate, and fitness coupling (the readout-not-reward invariant, adapted)

- **Surrogate.** Reuse a FROZEN, gate-0-calibrated L3 surrogate (the learned-dynamics
  fingerprint at the frozen capacity, e.g. hidden=8, sigma=0.02, oracle 0.928 on world P;
  `docs/PREREGISTRATION_L3.md` section 12). Emergence is then tested against a surrogate
  already shown oracle-detectable but not incidentally-stored-persistently by within-life
  learning - the cleanest possible baseline for "did selection add persistence".
- **Fitness coupling (treatment only) - RESOLVED 2026-07-16: momentum-to-target foraging in a
  mixed-world lifetime (fork 3).** World identity is never in the observation and never a direct
  reward term. It becomes fitness-relevant only through the single primitive that actually differs
  between the worlds - the velocity-update law (section 4 surrogate; `patch_of_earth.py:175-177`).
  Concretely:
  - **Mixed-world lifetime.** Each individual lives a lifetime composed of both authentic and
    surrogate episodes (world drawn per episode by a keyed stream, label never observed). An agent
    that behaves identically in both is world-blind; an agent that can tell which world it is in can
    behave world-appropriately.
  - **Momentum-to-target foraging.** Reaching an energy resource requires coasting to it with the
    right thrust, and because the two worlds integrate velocity differently
    (`(1 - drag*dt)*vel + a*dt` vs the learned `G_motion` map), the thrust that lands on the resource
    differs by world. A world-conditional policy hits the resource and eats; a world-blind policy uses
    one compromise thrust, systematically over/undershoots, and forages less. More energy -> crosses
    the reproduction threshold sooner -> more descendants. This is the section-B2 "action that pays off
    only in the surrogate world", generalized to a per-world optimal action and made heritable.
  - **World-only-exploitability (gate 1).** The layout is set so the ONLY lever on the payoff gap is
    world identity: a world-blind policy cannot capture the treatment gap, verified by ablation before
    freezing (a policy forced to ignore any world cue must underperform a world-conditional one in the
    treatment while tying it in the control).
- **Control - RESOLVED: world-invariant layout.** Identical mixed-world exposure, identical
  reproduction, mutation, generation count and seeds, but the resource layout is tuned so the
  energy-optimal thrust is the SAME in both worlds (the target sits where the two velocity maps agree
  to within tolerance). Detecting the world then buys no fitness, so detection capacity should not rise.
  This is a cleaner control than shuffling the world label because the exposure statistics are identical;
  only the payoff geometry differs.
- **Selection mechanism - RESOLVED 2026-07-16 (fork 4): threshold-triggered reproduction.**
  Reproduction above an energy threshold with an energy cost and a mutated descendant (spec section 7),
  under a fixed carrying capacity so population size is bounded and stationary. Chosen over truncation /
  fitness-proportionate selection because it is the mechanism the world spec already defines and keeps
  population size a controlled constant across both arms.

## 5. The common-garden detection panel (keystone methodological control)

Per the explicit warning in `docs/ITASORL.md:315`: detection capacity is measured on a
**fixed-horizon common-garden panel that is IDENTICAL every generation and independent of
who lived or died**, and **survival is reported separately from detection**.

- Each generation, a fixed set of matched authentic-vs-surrogate evaluation episodes
  (shared-prefix matched pairs, randomized branch order, fixed horizon) is run for a fixed
  sample of the current population; the linear world-identity probe reads
  `episode_features(h_t) = [mean h, final h]` with GroupKFold by episode.
- Because the panel is fixed-horizon and fixed-size, "detects better" cannot be inflated by
  "lived longer -> more timesteps -> more data". Survivor count / death rate per world is
  reported as a SEPARATE series, never pooled into the detection estimate.
- L0 sub-panel (authentic-vs-authentic) is included every generation: detection there must
  stay at chance across all generations, proving any emergence is artifact-driven, not drift
  in the apparatus.

## 6. Estimand and smallest effect of interest

- **Estimand:** the generational GAIN in common-garden detection AUROC,
  `Delta = AUROC(final gen) - AUROC(gen 0)`, per population, averaged over independent
  lineage seeds; and the treatment-minus-control difference of that gain.
- **SESOI (RESOLVED 2026-07-16, fork 5):** claim emergence only if
  `Delta_treatment - Delta_control >= 0.05` with the seed-level 90% CI (t-based or BCa)
  excluding 0, AND the treatment final-generation AUROC clears the same 0.65 bar used across
  the project. The 0.05 margin and the 0.65 bar reuse the L3 SESOI so C is directly comparable
  to the earlier rungs.

## 7. Gates - ALL must pass before the emergence contrast is interpreted

Carried from the L3/B-v3 battery, plus C-specific gates:

0. **Surrogate gate 0 (reused, frozen).** The L3 oracle detectability band [0.85, 0.95] and
   the chance-level organism mechanical floor must hold for the frozen surrogate on world P
   before any generational run (re-validated per `scripts/run_expA_l3.py`).
1. **Fitness-coupling gate (C-specific).** BEFORE the generational run, verify by
   ablation/oracle that the surrogate-specific payoff is exploitable ONLY via world identity
   (a world-blind policy cannot capture it), and that in the CONTROL arm world identity is
   fitness-neutral (payoff decoupled). If the payoff leaks a non-world cue, re-design before
   running.
2. **Selection-works gate.** Fitness (mean lifetime energy / reproductive output) must
   increase across generations in the treatment arm - otherwise selection is too weak to test
   emergence and the run is uninformative, not a null.
3. **L0 no-emergence control.** Authentic-vs-authentic common-garden detection stays
   equivalent to 0.5 (TOST) across all generations.
4. **Leakage-audit battery.** Metadata, summed reward, episode length, lifetime, reset counts
   each fail to separate worlds on the common-garden panel (pass/fail), re-run every few
   generations.
5. **Positive control.** The speed/known-signal probe >= 0.75 on the panel.

## 8. Decision matrix

- **Emergence under selection (conditional on this surrogate and horizon):** all gates pass
  AND `Delta_treatment - Delta_control >= 0.05` (CI excludes 0) AND treatment final AUROC >= 0.65.
- **Strengthened negative:** all gates pass (including selection-works gate 2) AND
  `Delta_treatment ~= Delta_control ~= 0` - selection with world-coupled fitness, over this
  horizon, does not build a persistent detector beyond the reactive baseline. A strong result
  given the L3 reactive finding.
- **Uninformative:** gate 0/1 fails (mis-calibrated surrogate or leaky payoff) or gate 2 fails
  (selection too weak to move fitness).

## 9. Fixed configuration (laptop-minimal pilot; frozen at commit)

Compute is the binding constraint (see section 11). Pilot config for the local GPU host,
deliberately small so a first signal is reachable across sessions with cell-level resume:

- **Population:** N = 48, fixed carrying capacity (threshold-triggered reproduction).
- **Generations:** G = 30 (pilot); extension to G = 60 only if the pilot shows a non-flat,
  non-decisive trend.
- **Episode horizon:** max_steps = 80 (matches B-v3/L3), N_ray reduced (ray_steps=5, the
  B-v3/L3 staging convention).
- **Mutation:** Option P weight-perturbation sigma frozen after a pilot that targets a
  per-generation fitness gain in a sensible band (avoid both no-selection and collapse).
- **Common-garden panel:** fixed 110 matched pairs at 24 steps (reuse L3 pooled panel size),
  identical every generation.
- **Lineage seeds:** 3 confirmatory, extension to 0..9 for the power estimate of the
  treatment-minus-control gain, mirroring L3.
- **Determinism:** separate keyed PRNG streams (world-init, weather, ecology, mutation, probe)
  per `docs/ITASORL_world_spec.md` section 12; a fixed-seed pilot must reproduce to the bit
  before any multi-seed run.

The mutation sigma is the one value left to a milestone-1 calibration (it is tuned to put the
per-generation fitness gain in a workable band, gate 2); every other number is fixed here and frozen
at commit.

## 10. Analysis plan

- Primary: the per-lineage generational AUROC curve for treatment and control, the endpoint
  gains `Delta`, and their difference with a seed-level 90% CI (t-based or BCa; report the
  percentile bootstrap too but adjudicate on the t/BCa interval, per the L3 boundary-coverage
  lesson).
- Report survival / death-rate series SEPARATELY from the detection curve (section 5).
- Report the untrained/random-weight floor and the generation-0 evolved population on the panel
  each generation (section 3; no predictor arm under pure neuroevolution).
- Same figure/JSON schema as the B-v2/B-v3/L3 artifacts; a machine-checked recheck entry added
  to `scripts/audit_stats_recheck.py` before any doc/results push (the pre-publication gate).

## 11. Key interpretive caveats

- **This is a different mechanism, not the L3 claim.** A positive C answers "can selection build
  a persistent detector", not "does within-life learning encode L3 drift". A null is the strongest
  form of the reactive reading: even fitness pressure on world identity does not persist it.
- **Compute-boundedness is a design constraint, stated up front.** N=48, G=30 on a laptop is a
  PILOT scale. A flat result at pilot scale is "no emergence at this budget", not "no emergence
  possible"; the persistent-host commitment for a decisive long-horizon run is undecided
  (`docs/ITASORL.md`). The pilot's job is to decide whether the effect is worth that commitment.
- **Selection can find shortcuts.** Selection may exploit the surrogate-specific payoff via a
  degenerate behavioral tell rather than an internal world representation; the common-garden panel
  reads STATE, and the behavior-mediation audit (`itasorl/behavior_audit.py`) must be applied to any
  positive to separate "evolved a detector" from "evolved a world-specific behavior that a probe
  reads off". This is the C analogue of the L3 behavior-mediation caveat.
- **The control is the claim.** Treatment-alone emergence is uninterpretable; only the
  treatment-minus-control gain licenses a selection claim.

## 12. Decisions (all resolved 2026-07-16; frozen at commit)

1. **Unit of selection:** RESOLVED - **Option P** (neuroevolution of policy weights). Section 3.
2. **Within-life learning:** RESOLVED - **OFF** (pure neuroevolution), the clean consequence of
   Option P. Section 3.
3. **Surrogate-specific payoff design:** RESOLVED - **momentum-to-target foraging in a mixed-world
   lifetime**, coupling fitness to the one primitive that differs (the velocity law); control is the
   world-invariant layout; world-only-exploitability is gate 1. Section 4.
4. **Selection mechanism:** RESOLVED - **threshold-triggered reproduction** under a fixed carrying
   capacity (spec section 7). Section 4.
5. **SESOI and horizon:** RESOLVED - reuse the project **0.05 gain margin / 0.65 bar**; start at the
   **N=48, G=30 laptop pilot** (section 9) and defer the persistent-host commitment for any
   n=0..9 / G=60 extension until the pilot trend is seen (section 11, milestone 4). Sections 6, 9, 11.

No decision remains open; the configuration is design-complete. "Frozen" in the project sense (a
commit that predates the run) happens when this doc is committed - see the delivery note the run owner
records here before milestone 1.

## 13. How to run (milestones, in order)

1. **Determinism + selection-works pilot (CPU/GPU-light).** Implement the generational loop and the
   fixed common-garden panel; verify bit-reproducibility on a fixed seed and that the treatment arm's
   fitness moves across generations (gate 2). No emergence claim yet.
2. **Gate 0 + gate 1 (surrogate + payoff).** Re-validate the frozen L3 surrogate on world P; prove the
   surrogate-specific payoff is world-only-exploitable and world-neutral in the control.
3. **Pilot emergence run (GPU, laptop-minimal, cell-level resume).** N=48, G=30, 3 lineage seeds,
   dumping per-generation states for offline re-probing and the behavior-mediation audit.
4. **Adjudicate + decide on scale.** Apply the frozen decision rule; if the pilot trend is promising but
   not decisive, decide the persistent-host commitment for the n=0..9 / G=60 extension.

## 14. Amendments for the post-invalidation re-run (DRAFT - ratify and commit before launch)

*Two independent 2026-07-18 methodology audits converged on this section, both
outcome-blind with respect to the re-run. The pilot recorded in FINDINGS
section 13 is invalid (13.C). None of the items below is binding until the run
owner reviews them, removes the DRAFT marker, and commits; none may change
after the re-run starts. Items marked [DECISION] need an explicit call before
ratification. The original sections 1-13 are unchanged.*

**Premise update (2026-07-19).** The L3 common-garden re-score (FINDINGS
10.6.1) has since RESOLVED the persistence question this document inherited as
background: the corrected numbers show a modest PERSISTENT world-signal
(forward 0.666, reverse 0.684, both clearing the frozen rule), not the
"reactive only" reading section 1 assumes. Experiment C's question is
therefore now "does selection AMPLIFY a modest persistent detector that
within-life learning already produces", not "does selection create one where
none exists". This reframing strengthens, not weakens, the design (a non-zero
gen-0 persistence level gives the contrast room to move in both directions),
but the re-run's write-up must use the amplification framing.

1. **Persistence estimand (co-primary late-tail).** The pilot adjudicated
   emergence on the full-tail `cg_tail_target`, whose early steps carry
   residual reactive state from the just-felt prefix, so a slow-decaying
   reactive signal could satisfy the claim. `cg_latetail_target` (last 8 tail
   steps) is elevated to a CO-PRIMARY persistence condition, reported with the
   same contrast construction: an emergence claim requires the late-tail
   contrast to agree in direction.
2. **Threshold rule and selection-intensity matching.** Reproduction threshold
   = the q = 0.5 quantile (median) of EACH arm's own gen-0 fitness (ratifies
   the disclosed pilot deviation; single scalar q applied symmetrically).
   Contingencies: (a) the per-arm rule is justified only if the two arms'
   gen-0 fitness regimes are disjoint ON WORLD P (the pilot's justification
   was measured on the wrong world) - the runner records both distributions;
   if their supports overlap by more than 25%, revert to the registered shared
   absolute threshold and record which rule fired; (b) matching at gen 0 does
   not guarantee matched intensity across generations (the pilot's control arm
   selected ~3x harder on fitness delta) - the runner persists
   `n_qualifiers_treat/ctrl` per generation, and a mean qualifier-fraction
   difference between arms above 0.15 marks the contrast
   intensity-confounded (validity flag, reported with the estimand). A
   q-robustness check at q = 0.25 on one seed is reported descriptively.
3. **Gate battery (sections 7 gates 1-5): IMPLEMENTED in-run.** As of this
   amendment the runner computes and adjudicates the full battery
   (`scripts/run_expC_milestone3.py`): gate 1 via the scripted-oracle
   `gate1_exploitability` at margin 0.005 against the in-process frozen map;
   gate 2 in the per-arm ALL-seeds form (the pilot's weaker any() form is
   reported alongside); gate 3 as TOST (margin 0.05) over the pooled
   per-panel L0 AUROCs; gate 4 as the pair-grouped leakage battery on the
   pooled panel tails (reward_sum is the live channel; length/lifetime are
   constant for surviving pairs by construction); gate 5 as the panel speed
   positive control >= 0.75. Gate 0 is re-validated externally
   (`scripts/run_expA_l3.py`) before launch and its oracle AUROC recorded in
   the run config. Any gate failure routes the run to UNINFORMATIVE per
   section 8.
4. **[DECISION] Gate-1 scope and treatment geometry.** The gate-1 control leg
   cannot fail by construction at the default layout (see the 2026-07-18 note
   in `itasorl/experiment_c_gate1.py`); control-arm fitness-neutrality rests
   on `scripts/derisk_expC_control.py`, so gate 1 is read as
   treatment-leg-only with the de-risk sensitivity cited. Moreover gate 1
   FAILS at the current frozen treatment geometry (gap 0.00228 < margin
   0.005). The corrected steepness sweep indicates reach 0.15 / horizon
   80-class layouts clear the margin. Ratify one of: (a) re-freeze the
   treatment layout to the steepest certified cell so gate 1 can pass, or
   (b) keep the frozen layout and accept that the re-run is a
   gates-and-machinery pilot that cannot adjudicate emergence.
5. **Panel cadence (amends "each generation", sections 5/7/10).** The full
   panel runs at gen 0 (shared), every 10 generations inside evolution
   (`--panel-every 10`), and at the final generation of each arm. Gate 3's
   "across all generations" and gate 4's "every few generations" clauses are
   correspondingly weakened to this cadence; the section-10 generational
   AUROC curve is measured at this resolution.
6. **Runtime configuration (frozen; previously unregistered).** Policy nets
   embed = 8, hidden = 8, world_model=False; n_eps_per_world = 2;
   max_steps = 80; drift_sigma = 1.0 in l3 mode (installs the frozen map;
   magnitude unused); mutation sigma = 0.03 (the milestone-1 calibration,
   recorded late - a disclosed documentation deviation); panel 110 pairs x
   prefix 20 x tail 24 PER INDIVIDUAL, pooled across the population (the
   effective detection sample is up to N x 110 surviving pairs; the "110
   matched pairs" phrasing in sections 5/9 describes the per-individual
   panel, not the pooled sample size).
7. **Futility / extension rule (replaces the invalid section-13
   point-estimate argument).** Three zones on the contrast leg, adjudicated
   on the t-based 90% CI: EMERGENCE if the lower bound > 0 and mean >= 0.05
   and the floor leg passes (unchanged); FUTILE only if the upper bound
   < 0.05; otherwise INCONCLUSIVE, triggering the pre-committed extension to
   the n at which the observed across-seed sd gives a one-sided MDE <= 0.05
   (expected n ~ 8-10). The n = 3 re-run is explicitly a gates-plus-variance
   pilot; its contrast is decision-relevant only in the EMERGENCE and FUTILE
   zones.
8. **Panel niche asymmetry (interpretive caveat, recorded now).** The panel
   uses the sparse/far layout for both arms; the control arm evolves on
   dense/near and is therefore scored out-of-distribution. This biases the
   contrast TOWARD emergence, so it cannot explain a null; any POSITIVE
   contrast must be accompanied by the supplementary dense-panel readout for
   both arms before it is claimed.
9. **Reproducibility wording.** Determinism checks compare fitness series
   rounded to 8 decimals and never re-derive the panel AUROC; claims say
   "reproducible to 8 decimals", and a CUDA run adds a CPU-parity spot check
   on one seed. (The runner enables
   `torch.use_deterministic_algorithms(True, warn_only=True)`.)
