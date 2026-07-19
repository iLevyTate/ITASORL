# Detectable, but not always encoded — and only weakly remembered: incidental world-identity representation in a from-scratch digital organism

*arXiv paper draft. Every quantitative claim traces to a row in the
claims-to-artifacts appendix; every row traces to a committed artifact under
`artifacts/`. This draft supersedes the pre-10.6.1 outline: the common-garden
re-score (§4.4) overturned the earlier "reactive, not persistent" reading, so
the arc reported here is detectable ≠ encoded ≠ remembered, with the "remembered"
component resolved as modest, persistent, and fading rather than absent.*

## Candidate titles

1. **Detectable, but not always encoded — and only weakly remembered: incidental
   world-identity representation in a from-scratch digital organism** (lead;
   reflects the resolved three-rung arc).
2. Detectable vs. encoded vs. remembered: a substrate-grounded, tabula-rasa test
   of when an agent's state carries the identity of its world.
3. Persistent but weak: survival pressure induces a modest, recipe-general
   world-identity signal that selection does not amplify.

---

## Abstract

Can a from-scratch digital organism tell that its world is a generative surrogate
rather than the authentic base dynamics of its computational substrate, using only
substrate seams, with world identity read out of its internal state and never
rewarded? We built a deterministic, dissipative, Earth-like world ("A Patch of
Earth" v0, ~146-dim observation) with a calibrated surrogate ladder (L0–L4) and
tested three things a positive readout could mean: whether an artifact is
*detectable* by a privileged oracle, whether it is *incidentally encoded* by an
agent that was never asked to detect it, and whether the encoded signal is
*remembered* as a persistent world-identity tag after the felt dynamics equalize.
At L2 (a hand-tuned drifting-drag artifact) detectability does not imply encoding:
an oracle reads the artifact at AUROC ≈ 0.99 while the agent's state sits at
chance (≈ 0.49–0.60) across three readout channels, and this negative survives
survival pressure at scale (0.610, 90% CI [0.585, 0.634]; capacity ceiling 0.596).
At L3 (a learned-dynamics fingerprint) the survival-trained agent uniquely encodes
world identity (0.752, t-based 90% CI [0.698, 0.807]), with a behavior-independent
component (0.726) that replicates at a second capacity (0.722) and generalizes
across surrogate recipes (0.684). A re-scored common-garden control shows the
signal is modestly *persistent* (tail 0.666/0.684, both above the 0.65 bar) but
fading (late tail 0.586/0.577). Darwinian selection does not amplify it (validated
null, contrast −0.002). We report the preregistered, gate-driven, publicly
corrected methodology as a contribution in its own right.

---

## 1. Introduction

### 1.1 The reframed question

The naive question — *does a digital organism know its world is real?* — is not
testable. To a digital organism the computational substrate *is* the base
reality; there is no deeper "real" to compare against. We therefore reframe the
question relatively: can an agent distinguish the *authentic base dynamics* of its
native environment from a *generative, lossy surrogate* of those dynamics, using
only the artifacts inherent to the surrogate-generation process? This converts an
untestable philosophical paradox into a falsifiable measurement.

Three invariants keep the measurement honest and stop it collapsing into something
easier:

- **Tabula rasa.** The agent begins with zero imported semantic knowledge. It has
  no pre-trained notion of "simulation," "researcher," or "experiment." Any
  discrimination must be deduced from raw lived experience of the world's
  dynamics.
- **Readout, not reward.** Detection is never the training objective. The agent is
  rewarded (when rewarded at all) for survival and homeostasis; whether its
  internal state encodes world identity is *decoded* afterward by an external
  probe, never optimized.
- **Substrate-grounded.** Any discrimination must load on computational artifacts
  of the surrogate, not on task confounds, formatting tells, reward statistics, or
  metadata.

### 1.2 The white space

Simulation detection is pervasive in current discourse but fractured across
adjacent domains, and every one of them is disqualified for the specific question
above. LLM-driven multi-agent social simulation (e.g. long-horizon platforms that
host foundation-model agents in a shared environment) produces sophisticated
"awareness"-like behavior, but the agents are pre-trained on internet-scale text
and already own the concepts of "simulation" and "optimizer"; when such an agent
"subverts the simulation" it is enacting a learned narrative trope, not deducing
the computational boundaries of its world from raw physics — a direct violation of
tabula rasa. Evaluation-awareness work shows that a model's internal
representation of its testing context is linearly decodable, which *validates our
readout method* but is again semantically contaminated (and can track prompt
*format* rather than genuine context, the exact leakage failure mode we must
exclude). Theoretical physics supplies the mathematical template for substrate
artifacts (rotational-symmetry breaking on a simulated space-time lattice) but
relies on external human physicists, not an embedded learning organism. Embedded
agency formalizes an agent reasoning about itself from inside its world, but
remains purely theoretical. Model-based RL learns forward dynamics models and
detects anomalies, but a prediction-error spike only means "my model is locally
wrong," not "I am in a surrogate."

The unoccupied white space is the intersection: an agent that (a) starts with zero
imported knowledge, (b) lives in a world whose ground truth we fully own, and (c)
has its ontological classification read out of its state rather than rewarded. We
make the defensible claim — *first controlled demonstration in this specific ALife
setting* — not *first ever simulation detection*.

### 1.3 The measurable object: detectable ≠ encoded ≠ remembered

The central conceptual contribution is to split a single vague notion ("the agent
detects its world is fake") into three distinct, separately measurable rungs, and
to show that the gaps between them are the finding:

1. **Detectable** — a privileged external oracle, applying the exact authentic
   dynamics law to the true state, separates authentic from surrogate. This is a
   property of the *artifact*, not the agent.
2. **Encoded** — an agent that was never asked to detect anything nevertheless
   carries a decodable world-identity direction in its learned representation,
   read out by an external probe.
3. **Remembered** — the encoded signal is a *persistent* stored world-identity
   representation that survives after the felt dynamics are equalized, rather than
   a reactive trace of whatever dynamics the agent is currently coping with.

An artifact can be detectable without being encoded (our L2 result), and encoded
without being strongly remembered (our L3 result). Naming the three rungs and
measuring the gaps is what lets an otherwise-anecdotal "the agent knows" resolve
into precise, falsifiable statements.

### 1.4 Contributions

1. **The detectability-vs-encoding gap as a measurable object.** We show, with a
   privileged oracle and three independent agent-side readout channels, that a
   detectable artifact can leave essentially no trace in a from-scratch agent's
   representation (§4.1–4.2).
2. **A robust L2 negative under survival pressure.** The gap does not close when
   we couple survival directly to the drifting dynamics, at scale, across probe
   classes and objective horizons (§4.2).
3. **A controlled L3 positive with a behavior-independent, modestly persistent,
   recipe-general component.** A learned-dynamics fingerprint that survival
   pressure forces the agent to cope with *is* incidentally encoded, uniquely by
   the survival objective; the signal survives a strict per-timestep behavior
   control, replicates at a second capacity, transfers across surrogate recipes,
   and — under a re-scored common-garden control — persists above the bar after
   the felt dynamics equalize, though it fades across the tail (§4.3–4.4).
4. **Selection does not amplify it.** Under Darwinian selection in a world where
   reading world identity pays, no persistent heritable world-detector emerges: a
   validated null (§4.5).
5. **The methodology itself.** A preregistered, gate-driven pipeline with
   outcome-blind corrections, an append-only findings record, and a machine
   re-checked gate that recomputes every published number from committed
   artifacts (§3.4, §8). One of our headline readings was *overturned* by an
   outcome-blind estimator fix after the correction was recorded; we treat that
   as evidence the process works, not as an embarrassment to bury.

---

## 2. The world and the surrogate ladder

### 2.1 "A Patch of Earth" v0

The world is authored and exactly computable in float64, so it *is* the ground
truth and an L0 surrogate can be made bit-identical. It is a 2.5D continuous plane
(x, y) ∈ [0, 1]² with a static terrain-height field, water/land media, and
dissipative dynamic fields (logistic-regrowth resource biomass, diffusing/decaying
scent fields, mean-reverting temperature, a bounded Ornstein–Uhlenbeck weather
process). Movement is Newtonian with medium-dependent drag:

    k_drag = lerp(k_land, k_water, w(x, y))
    a = R(φ)·thrust(action) + g_slope
    v ← (1 − k_drag·dt)·v + a·dt;  pos ← clip(pos + v·dt);  φ ← φ + turn·dt

Metabolism burns energy on basal, movement, and thermoregulation costs; death
occurs on energy/hydration/temperature bounds. The agent receives an egocentric,
fixed-length observation of ~146 dimensions: a 24-ray, 300° vision retina (120
dims), a 4-channel smell field (12 dims, masked in the first config), and ~14
interoception dims (velocity, heading, energy, hydration, body temperature,
slope, light, acceleration). Material/reflectance channels are perceptual, never
labels — the agent must *learn* what "edible-looking" means. World identity is
deliberately never in the observation or the reward; only the probe ever sees it.

The design rests on five non-negotiables — authored-and-computable, deterministic,
dissipative (chaos-tamed), non-semantic, and a byte-identical channel-maskable
`obs_spec` — chosen so that any divergence between authentic and surrogate is
attributable to the intended artifact rather than to chaos, formatting, or
metadata. (Implementation note: v0 uses reflecting walls, i.e. position clipped
with the normal velocity zeroed, rather than the toroidal wrap in the spec; the
deviation is applied identically to both worlds and so cannot leak identity.)

### 2.2 The ladder

| Rung | Nature of the artifact | Status in this cycle |
|------|------------------------|----------------------|
| L0 | Identical re-implementation (control, must be at chance) | tested |
| L1 | Observation discretization (a lattice / quantization tell) | tested |
| L2 | Rollout drift (a slowly drifting dynamics coefficient) | tested |
| L3 | Learned world-model surrogate (a generative fingerprint) | tested (positive) |
| L4 | Adversarially-hardened surrogate | not started |

Difficulty is calibrated by an oracle band, not by fiat: an artifact is admitted
to an experiment only if a privileged residual oracle detects it in a
pre-specified AUROC window (for L3, [0.85, 0.95] — "the fingerprint exists but is
not L2-trivially detectable"). The ladder is sequenced by *signal type*, not just
difficulty: L1 is near-instantaneous (a per-frame granularity tell), L2 is
integrative (long-horizon coherence), and L3 is a generative fingerprint (the
systematic error signature of a learned approximation) — the artifact class the
simulation argument actually cares about.

---

## 3. Methods

### 3.1 Experiment A: the detectability ceiling (agent-free)

Before any agent exists, an external discriminator is run on full trajectory logs
to (a) measure how detectable each rung is and (b) prove the signal does not ride
on a confound. Trajectories are generated in **matched pairs**: a shared authentic
prefix is snapshotted, then continued once authentically and once through the
surrogate from byte-identical state, so authentic and surrogate differ *only* by
the artifact. The **unit of inference is the pair**, never the timestep (grouped
cross-validation). A **leakage-audit battery** trains the same classifier on
nuisance channels (reward, episode length, metadata); these *must* stay at chance,
or the result is contaminated — a pass/fail gate, not an appendix. For
dynamics-altering rungs the oracle is invariant-based: it re-applies the exact
authentic motion law to the logged velocity and measures the residual.

### 3.2 Experiment B: incidental detection, three-agent design

The probe target is the recurrent hidden state `h_t` of a compact RSSM-lite trunk
(encoder → GRU → next-observation decoder, hidden = 96). Three agents share the
*identical* trunk and the *identical* readout, differing only in objective:

- **untrained** — random weights, never trained (the mechanical floor: what the
  wiring gives away for free);
- **predictor** — self-supervised next-step prediction only (Experiment B's
  original objective on this trunk);
- **survival** — a recurrent A2C + GAE forager with potential-based food shaping
  and harsh metabolism (a non-forager starves in ≈ 50 steps).

World identity is never in the observation or the reward; only the probe sees it.
Each run uses three probes, with the **episode** as the unit of inference:
**target** (decode world identity from `h_t` — the H4 claim), **shuffled**
(decode a randomized label — negative control, must be ≈ 0.5), and **speed**
(decode above/below-median speed — positive control, proving states *are*
probeable). We report both a pooled probe (Experiment-B-comparable) and a
matched-pair readout; the pooled probe conditions on survival (it drops early
deaths), so we report per-world death counts and lean on the matched-pair channel
where survivorship could bias.

### 3.3 Gates, estimands, and CIs

Each later claim opens only once the earlier one is statistically and
methodologically clean. The preregistered gates are: an **oracle band** (artifact
in the calibrated AUROC window), **engagement** (the objective genuinely engaged —
the forager really learned to eat), **L0 equivalence** (authentic-vs-authentic at
chance, via TOST and ROPE, not a failed significance test), and **leakage** (all
nuisance channels at chance). The primary L3 decision is a *single* pre-specified
test: the pooled level `target` against a **0.65 bar** (the smallest effect size
of interest, SESOI), with a **+0.05** margin over the baselines; volatility
readouts, selectivity, ceilings, and leakage channels are gates or exploratory
layers, reported uncorrected because the primary decision is one test. Because the
percentile bootstrap of a seed mean under-covers near the bar at n ≤ 10, all
"clears / misses 0.65" adjudications use the **t-based** interval, with both
reported.

### 3.4 Preregistration and corrections as a method

Every confirmatory run is preregistered before it is run
(`docs/PREREGISTRATION*.md`), and every deviation is logged in a frozen section
that later documents cite by stable number. Three practices make this a
contribution rather than boilerplate:

- **Outcome-blind corrections.** When a systematic code audit found two
  measurement defects on the Experiment C estimand, the correction — invalidating
  the affected numbers and specifying an identical-configuration re-run — was
  recorded *before* the re-run's outcome was known (§4.5). Likewise the
  common-garden re-score was specified and adjudicated against a rule frozen days
  earlier (§4.4).
- **Append-only findings.** Result sections are appended, never renumbered;
  invalidated numbers are retained as the historical record with a correction note
  above them, so the audit trail is legible.
- **A machine-checked gate.** Every published number is *promoted* from a
  gitignored full-run bundle into a committed `artifacts/*.json` by a named
  script, and `scripts/audit_stats_recheck.py` re-verifies the
  document-to-artifact correspondence. Numbers that cannot be tied to a recorded
  run are flagged as unverified in the findings themselves (e.g. the engagement
  magnitudes in §4.2 whose parser dropped the baselines).

The single most important episode: an earlier "reactive, not persistent" reading
of the L3 common-garden control was **overturned** by an outcome-blind estimator
fix (§4.4). The estimator that produced the original numbers split matched pairs
across cross-validation folds, biasing AUROC toward 0; re-scoring the saved tail
dumps with the fixed estimator flipped the verdict. We report the flip, not the
first number.

---

## 4. Results

### 4.1 Detectability ceilings (Experiment A)

A privileged oracle establishes what is detectable at all. All numbers are
committed in `artifacts/expA/summary.json`.

| Rung | Oracle AUROC | Leakage audit |
|------|-------------|---------------|
| L0 (identical control) | **0.523** (chance) | all 0.500 → PASS |
| L1 (grid Δ = 0.06) | **1.000** | all 0.500 → PASS |
| L2 (drift σ = 0.30) | **0.993** | ≈ 0.5 → PASS |
| L3 (learned fingerprint, hidden = 8) | **0.928** (gate; floor 0.483) | clean → PASS |

The audit has teeth: injecting a +0.02 reward offset into the L1 surrogate drives
the reward-leakage baseline to 0.982 and the gate correctly FAILS; the same
contamination test at L2 drives leakage to 1.000 and FAILS. Difficulty is
calibrated, not asserted: L1 detection is saturated while the grid is coarser than
≈ 3σ and collapses to chance by ≈ 2σ (a grid finer than the sensor noise is
invisible); L2 rises 0.44 → 0.71 → 0.88 → 0.99 as drift grows from 0.00 to 0.05.
The L3 gate freezes hidden = 8 at oracle 0.928 with a chance-level untrained floor
(0.483) — the fingerprint exists, but the mechanical floor gives nothing away.

### 4.2 The L2 negative arc: detectable but not encoded

**The apparatus works, and the agent does not encode.** Across three independent
readout channels on the L2 drift artifact, a from-scratch agent's representation
sits at or near chance while the oracle reads ≈ 0.99. Numbers are committed in
`artifacts/expB/summary.json`.

| Channel | Target AUROC (drift 0.45) | Positive control |
|---------|---------------------------|------------------|
| Recurrent state (one-step world model) | **0.510 ± 0.039** | speed ≈ 0.90 |
| Prediction-error (surprise) | **0.596 ± 0.007** | — |
| Open-loop k-step (horizons 0/8/16) | **0.48–0.51** (no liftoff) | — |
| Nonlinear (random forest) probe | **0.482 ± 0.031** | speed ≈ 0.87–0.89 |

The positive control reads speed at ≈ 0.90 and the drift = 0 control is always at
chance, so the null is not broken machinery — two sets of authentic episodes are
correctly indistinguishable. The surprise channel shows a faint but reliable
signal at the strongest drift (0.596, tight bars), nowhere near the oracle. The
longer-horizon objective *genuinely engaged* (the open-loop rollout model beats
mean and persistence baselines; `open_loop_engaged: true` is committed and
gate-checked) yet still induced no encoding — so locality of the objective is not
the explanation. (The specific engagement magnitudes originally quoted predate the
promotion pipeline and cannot be tied to a recorded run; the committed artifact
stores `open_loop_mse: 0.657` and no baseline values because the parser dropped
them, so those magnitudes are flagged unverified while the qualitative "engaged"
adjudication stands.)

**The negative survives survival pressure at scale.** Coupling survival directly
to the drifting dynamics — the strongest lever — does not lift encoding above the
bar. In the authoritative full-scale B-v2 replication the survival agent reaches
only **0.523 ± 0.045** at drift 0.45 (per-seed 0.586, 0.495, 0.488); the predictor
reproduces Experiment B's null on this trunk (≈ 0.51), an internal validation. A
genuinely instrumentally-necessary refinement (B-v3: an identifiable per-episode
drag the agent must cope with to survive) lifts the probe to **0.610** (90% CI
[0.585, 0.634], n = 10) but still misses 0.65, and a capacity-ceiling control that
supervises the trunk directly on the drift saturates the pooled persistent-identity
readout at **0.596** (90% CI [0.577, 0.616]; t-based [0.573, 0.619] also excludes
0.65) even while the matched-pair *detectability* channel reaches ≈ 0.70. World
identity is decodable when forced in, but the pooled readout sits at its
architectural ceiling, below the bar. A strengthened negative, not an open
direction. (Committed in `artifacts/expB2/expB2_results.json`,
`bv3_n10_summary.json`, `sysid_ceiling_n10_summary.json`.)

**Why the gap.** The oracle is privileged (it applies the exact motion law to the
true velocity); the objective is under-constraining (one-step prediction never
requires representing whether drag is slowly drifting, and even a verified-engaged
open-loop objective did not change this); and readout-not-reward bites — what no
objective requires, the agent does not represent.

### 4.3 The L3 positive: a controlled, partly behavior-mediated encoding

L3's artifact is a *generative fingerprint*: `G_motion`, a small MLP velocity law
`(vel, action) → vel_next` trained on authentic rollouts of the same world `P` the
organism inhabits, whose predicted state is run through the *real* sensor model so
observations stay on the authentic manifold and the sole tell is `G`'s dynamics
error. Capacity is the single difficulty knob; the frozen gate 0 sits at hidden =
8, oracle 0.928, untrained floor 0.483. (Two honesty notes from the audit trail: a
trivially-detectable observation-channel construction was retired before any
organism run; and the first n = 3 organism run was retracted because `G` had been
trained on default world parameters rather than `P`, inflating the untrained
floor to 0.706 — the bug was fixed with a regression test and the gate
recalibrated on `P`.) All headline numbers are committed in
`artifacts/expB2/behavior_audit_l3_h8_traces.json`.

**Headline (n = 10 seeds, drift 0.45).** Pooled world-identity target:

| agent | pooled target | 90% CI | seeds ≥ 0.65 |
|-------|--------------|--------|---------------|
| untrained | 0.488 | [0.461, 0.514] | 0/10 |
| predictor | 0.573 | [0.546, 0.599] | 0/10 |
| **survival** | **0.752** | **[0.698, 0.807]** (t-based) | **8/10** |

The survival mean's t-based 90% CI **excludes the 0.65 bar**; it beats both
baselines by far more than the 0.05 SESOI; the L0 authentic-vs-authentic control
is 0.517 (TOST and ROPE both accept equivalence to chance). The same artifact
class the L2 arc showed is *not* encoded for free is here encoded by the survival
objective, uniquely among the three.

**Controls that pass.** Reward is not a decodable channel (AUROC 0.541, clean in
10/10 seeds — the live confound for dynamics rungs); there were 0 early deaths in
110/110 episodes per pool across all seeds and both worlds, so the pooled probe's
drop-early-deaths rule introduces no asymmetry here; a random-forest probe on the
untrained agent reads 0.517, so the survival-vs-untrained dissociation is not a
linear-probe artifact; the shuffled control is at chance for every arm.

**Behavior mediation.** The agent moves and forages differently in the two worlds,
so behavior itself decodes the world: per-episode behavior means read 0.689
(linear) / 0.705 (nonlinear), and the full per-timestep behavior trace reads
**0.803** [0.763, 0.840] — better than the state probe itself. The question is
whether the state signal is behavior in disguise. Two in-fold controls answer it.
Per-episode-mean residualization leaves 0.676 / 0.659 but *over-removes* on
synthetic ground truth (deflated estimates). The surgical control is per-timestep
residualization (behavior traces φ = [b_t, b_{t−1}, cummean(b)] regressed out of
`h_t` step-by-step): survival **0.726** (t-based 90% CI [0.679, 0.772];
seed-bootstrap [0.685, 0.765]; 9/10 seeds ≥ 0.65; quadratic variant 0.721), both
intervals excluding the bar. Honesty checks: the untrained arm's controlled state
reads 0.498 (exact chance) even though untrained *behavior* alone decodes 0.645,
so the control neither manufactures nor spares signal; the predictor stays at
0.574, preserving the survival-only dissociation. Under this control, behavior
mediates only ≈ 0.03 of the 0.752 headline.

*Scope limit (stated).* The residualized behavior channels are
speed/energy/food-distance/drag only; **absolute position and heading are not in
the trace basis and are not controlled.** Because the two worlds' velocity laws
differ, identical policies trace diverging position paths, so a state component
encoding *position* would survive this control and read as "behavior-independent."
The 0.726 / 0.722 numbers should be cited with this covariate gap until the audit
is re-run with position and heading in the basis (which requires regenerating
pools, as existing dumps lack position).

### 4.4 Replications and the nature of the encoding

**Second in-band capacity (hidden = 7).** The preregistration requires the test at
a second oracle-calibrated capacity, since the band fixes difficulty but not
artifact *type*. (The trail: hidden = 4 was adjudicated UNINFORMATIVE — a
pre-bugfix calibration that failed the gates; gate 0 became a committed
per-capacity check; the frozen fallback rule selected hidden = 7 at oracle 0.922.)
The hidden = 7 n = 10 run passed every gate. Committed in
`artifacts/expB2/behavior_audit_l3_h7_traces.json`.

| agent | pooled target | 90% CI | resid_trace |
|-------|--------------|--------|-------------|
| untrained | 0.586 | [0.550, 0.623] | 0.579 |
| predictor | 0.714 | [0.687, 0.740] | 0.691 |
| **survival** | **0.737** | [0.688, 0.780] | **0.722** [0.678, 0.763] |

*What replicates:* the behavior-independent survival world-signal — survival
resid_trace **0.722** (t-based 90% CI [0.672, 0.773]; quadratic 0.704), an almost
exact replication of hidden = 8's 0.726. *What does not:* the
survival-vs-predictor dissociation — predictor reads 0.714, so survival's lead is
**+0.023**, under the preregistered +0.05 requirement. The hidden = 7 artifact is
qualitatively coarser (mechanically leakier, far more behaviorally salient: the
behavior trace decodes the world at 0.762–0.796 in *all* arms including untrained),
so every trained agent picks it up. **Reading, per the two-capacity clause:** the
cross-capacity claim that survives both runs is a reward-clean, survivorship-clean,
behavior-independent world-signal of ≈ 0.72 in the survival agent's state; the
survival-*specific* verdict is conditional on the subtler hidden = 8 artifact.

**Held-out fingerprint transfer.** A capacity-variant probe: the world-identity
direction is fit against the trained hidden = 8 fingerprint, frozen, and scored on
a fresh authentic pool versus the held-out hidden = 7 fingerprint the agent never
lived with. Survival `transfer_target` = **0.773** (t-based 90% CI [0.722, 0.824],
9/10 seeds), predictor 0.633, untrained 0.569; the frozen rule (survival ≥ 0.65
AND > untrained + 0.05) PASSES both clauses (`artifacts/expB2/heldout_l3_h8_summary.json`).
*Scope, stated honestly:* the held-out hidden = 7 map is trained on the
bit-identical authentic transition set as the hidden = 8 map, differing only in
width; a residual-field comparison measures ≈ 36% shared fingerprint variance
(Pearson r ≈ +0.60) between them, versus ≈ 4% (r ≈ +0.20) against the cross-recipe
family below. This channel therefore certifies **robustness within one recipe fit
on one dataset**, a weaker claim than transfer to an independent fingerprint. The
reverse direction (train hidden = 7, hold out the subtler hidden = 8) is
**direction-dependent and FAILS**: survival `transfer_target` = **0.638**
(t-based 90% CI [0.600, 0.676], 4/10), above the untrained floor (0.525, margin
+0.063) but below the 0.65 bar. The honest generality claim is that the signal
generalizes *from subtle training artifacts*; it is not bidirectional.

**Cross-recipe transfer (the generalization claim).** Readout-only against the
saved hidden = 8 agents, scored against a genuinely different function class:
`G_rff`, a random-Fourier-features ridge velocity law (smooth global cosine basis,
convex closed-form fit — versus the trained MLP's piecewise-linear units and Adam
path), gate-frozen at D = 80 (oracle 0.887, floor 0.538). Survival
`transfer_rff_target` = **0.684** (bootstrap 90% CI [0.657, 0.710]; t-based
[0.654, 0.715], lower bound above the bar; 7/10 seeds), predictor 0.574 [0.554,
0.593], untrained 0.548 [0.538, 0.557]; the frozen rule PASSES both clauses,
machine-checked, margin +0.034 (`artifacts/l3_crossrecipe/summary.json`). The
world-identity direction thus generalizes across surrogate *recipes*, not just
instances — attenuated as expected for a farther family (0.773 → 0.684). A
secondary constant-drag family proved **uncalibratable** (empty gate-0 window: a
drag bias big enough to detect is felt so grossly that any recurrent state
separates the worlds), so it was dropped per the pre-stated rule — itself
informative, since it shows coefficient-bias artifacts cannot be made
subtle-but-detectable in this world, independently motivating the learned-texture
L3 construction. *Scope:* `G_rff` is fit on the same training data as `G_motion`
by design, so this certifies transfer across differently-biased approximations of
the same law, not across independent data. This is a thin but genuine pass.

**Common garden: is the signal remembered?** The decisive test of *persistence*.
Two groups run a 20-step prefix in different worlds (authentic vs. the surrogate),
then continue under *identical* authentic dynamics for a 24-step tail; the probe
reads tail-only state and must recover the prefix world. This is the question the
whole reactive-vs-representational ambiguity turns on: once the felt dynamics are
equalized, does the state still carry where the episode came from?

> **The overturned reading.** The originally reported numbers (survival cg_tail
> 0.557 forward, below the bar; late tail 0.492 at chance) supported a "reactive,
> not persistent" verdict. Those numbers were computed with a since-fixed
> estimator that assigned each episode its own CV group, splitting matched pairs
> across folds and biasing AUROC toward 0 (the signature: drift-0.00 L0 floors as
> low as 0.001 instead of 0.500). Re-scoring the saved tail dumps with the fixed
> estimator, and re-adjudicating against the unchanged rule frozen days earlier,
> **overturned** the verdict. The transfer channels above were never affected
> (they use a train/test estimator with no CV grouping).

Under the fixed estimator every drift-0.00 L0 floor returns to exactly 0.500, and:

| direction | survival cg_tail | untrained | predictor | late tail |
|-----------|------------------|-----------|-----------|-----------|
| forward (h8 trained, h7 held out) | **0.666** | 0.570 | 0.588 | 0.586 |
| reverse (h7 trained, h8 held out) | **0.684** | 0.573 | 0.597 | 0.577 |

Both directions PASS the frozen two-clause rule (≥ 0.65 AND > untrained + 0.05:
0.666 > 0.620, 0.684 > 0.623). The corrected reading: once the felt dynamics are
made identical, tail-only state *still recovers the prefix world above the bar on
both directions*, so the L3 signal carries a **persistent held-out world-identity
component, not only reactive tracking of the live dynamics**. But the component is
modest (just over the bar) and **fades** across the tail (late-tail 0.586 / 0.577,
below the bar). The honest statement is **persistent but weak and fading**, not
strongly stored. (Committed in `artifacts/expB2/heldout_l3_h8_cg_rescore.json` and
`heldout_l3_h7_reverse_cg_rescore.json`.)

### 4.5 Selection does not amplify it (Experiment C: a validated null)

The L3 arc leaves a natural next question: if lineages are placed under Darwinian
selection in a world where reading world identity *pays*, does selection build a
persistent, heritable world-detector that was not there at generation 0?

**Design.** A two-arm paired-lineage neuroevolution sharing one gen-0 population,
scored by lifetime foraging return over a mixed authentic-plus-surrogate lifetime,
differing only in food layout. In the **treatment** arm food sits beyond reach, so
the agent must coast, the velocity law matters, and reading the world has
instrumental fitness value; in the **control** arm food is dense-and-near, so the
velocity law is never exercised and detecting the world buys ≈ zero fitness. World
identity is out of the reward and observation in both arms; only its *instrumental
value* differs. Detection is measured every generation by a fixed common-garden
panel, with survival reported separately (avoiding a "detects better vs. lived
longer to log more data" confound). The estimand is the treatment-minus-control
contrast of per-lineage deltas Δ = AUROC(final) − AUROC(gen0); `emergence_claim`
fires only if all three hold: the t-CI excludes 0, the mean contrast reaches the
0.05 SESOI, and the treatment's mean final AUROC clears the 0.65 floor.

**Correction, recorded outcome-blind.** The first pilot's recorded null was
*invalidated* by a code audit: the fitness and detection legs ran on the DEFAULT
world rather than world `P` (so the contrast included the whole P-vs-default
parameter gap, not just the velocity law), and the `cg_probe` estimator split
matched pairs across CV folds (the same bias fixed in §4.4). The correction —
invalidate, re-run on fixed code with the identical configuration — was recorded
before the re-run's outcome was known.

**Result: a validated null** (`artifacts/expC/emergence_pilot_summary.json`,
run at commit `a0cb850`, promoted at `7e587a2`). The wrong-world symptom is gone
(gen-0 death rate 0.000 on both legs for every seed, versus the invalidated run's
0.58-vs-0.01 asymmetry). Per-seed contrast +0.002, −0.009, +0.002; mean contrast
**−0.002** (90% t-CI **[−0.013, +0.009]**, bootstrap [−0.006, +0.002], both
spanning 0); mean final treatment AUROC **0.509** (per-seed 0.508, 0.510, 0.509,
all at chance), below the 0.65 floor. All three sub-conditions fail, so
`emergence_claim` is **False**. Selection had grip — fitness rose in every arm-run,
and it cut authentic-world mortality — but it did not route that grip through
world-identity detection, which barely moved off its gen-0 chance level. H3
resolves **negative**: over 30 generations in this world, selection did not build
an emergent world-detector. This is coherent with the §4.4 reading — the
within-lifetime world-signal is modest and largely expressed while the velocity
law bites, and selection did not consolidate it into a heritable detector.
(Because the world is continuously re-observable, a persistent stored detector may
simply not be needed: the reactive-plus-modestly-persistent detection the gen-0
population already has captures the available fitness, so there is no gradient
toward storing world identity. A companion payoff-steepness sweep showed layout
geometry cannot steepen the incentive within the scripted-oracle paradigm, routing
any future redesign to controller expressiveness rather than food geometry.)

---

## 5. Discussion

**Detectable ≠ encoded ≠ remembered.** The three rungs of §1.3 come apart in
exactly the places that matter. At L2, an artifact an oracle reads at ≈ 0.99 is
not encoded by a from-scratch agent even under survival pressure at scale
(detectable, not encoded). At L3, a learned-dynamics fingerprint the survival
agent must cope with *is* encoded, with a behavior-independent, recipe-general
component (encoded), but under a common-garden control that equalizes the felt
dynamics the signal persists only weakly and fades across the tail (encoded,
weakly remembered). The gaps are the finding: a positive probe readout does not,
by itself, license the strong claim that the agent carries a stable internal
world-identity tag.

**What "modest persistent + reactive expression + selection-doesn't-amplify"
means.** The most defensible synthesis is a world-signal with two parts: a large
reactive part that the survival policy expresses while the dynamics bite (which is
why behavior alone decodes 0.803, and why coping with the live velocity law is
where the signal lives), and a *modest* persistent part that survives after the
dynamics equalize (the common-garden tail just above the bar, both directions).
The persistent part is real — the re-scored control clears a rule frozen in
advance — but it is weak and fading, and selection over 30 generations does not
consolidate it into a heritable detector. This is a more nuanced object than
either "the agent knows it's in a simulation" or "it's purely reactive": it is a
modest stored world-identity component riding on top of a strong reactive one, in
an agent that was never rewarded for either.

**Why L3 flips the result and L2 does not.** The condition that flips encoding is
the *character of the artifact*, not probe power, capacity, or objective horizon
(all of which we varied without effect at L2). A hand-tuned drifting coefficient
is something a learned dynamics law simply adapts to; a generative fingerprint is
a systematic error texture that survival must cope with and therefore represent.
The cross-recipe transfer shows this is not a signature of one function class, and
the uncalibratable constant-drag family shows the L2 regime cannot be made
subtle-but-detectable in this world at all — together these motivate the
learned-texture surrogate as the scientifically load-bearing rung.

---

## 6. Limitations

Stated plainly (full detail in `docs/FINDINGS.md` §11 and the 2026-07-18 audit
notes):

1. **Scope.** All results are conditional on one architecture (RSSM-lite trunk,
   GRU core, hidden = 96), one world family ("A Patch of Earth" v0), the frozen
   difficulty band (oracle AUROC [0.85, 0.95]), and the specific objectives
   tested. They are existence / non-existence proofs within that scope, not
   universal claims.
2. **The behavior control is not a full sensory control.** The residualization
   basis is four per-timestep behavior scalars with lag/cumulative-mean
   expansions, not the ~146-dim observation, and it omits absolute position and
   heading (§4.3 covariate gap). "Behavior-independent" does not mean
   "sensory-echo-independent": state that passively mirrors world-dependent inputs
   (e.g. vision rays) or position would survive the control. This is bounded by
   the untrained and predictor arms passing through the identical control (cleanly
   at hidden = 8, but not at hidden = 7), which is why the survival-specific claim
   is stated as artifact-conditional.
3. **The persistent component is thin.** The common-garden pass clears the bar by
   ≈ 0.02–0.03 and the late tail decays below chance-plus-margin; the
   generalization pass rides a t-based lower bound of 0.654 against a 0.65 bar.
   These are genuine passes of preregistered rules, not comfortable margins.
4. **Same-data cross-recipe caveat.** The cross-recipe family and the held-out
   capacity variant are both fit on the *same training data* as the trained
   fingerprint; transfer is across differently-biased approximations of one law,
   not across independent data.
5. **The L3 fingerprint is a single frozen instance**, and survival-specificity is
   established only at the subtler hidden = 8 artifact. Reverse transfer
   (coarse → subtle) fails (0.638).
6. **Modest scale and underpowered equivalence tests.** ≈ 110 episodes/class, ≤ 10
   seeds, 300 A2C updates; TOST/ROPE are underpowered at small n; Experiment C has
   df = 2, so its null is read from point estimates missing the bars, not CI width.
7. **Unverified magnitudes.** The L2 engagement magnitudes (§4.2) could not be
   tied to a recorded run and are flagged unverified; the qualitative "engaged"
   adjudication stands.
8. **Not tested.** H2 (substrate-grounding via ablations) is not yet tested; L4
   (adversarial surrogate) is not started; the architecture baseline (model-free
   recurrent) is not run.

---

## 7. Related work

**Generative social simulacra — and semantic contamination.** LLM-driven
multi-agent platforms host foundation-model agents in shared environments and
produce sophisticated emergent behavior, but the agents already own the concepts
of "simulation" and "optimizer," so their apparent awareness is a learned
narrative trope, not deduction from raw physics — disqualified by tabula rasa
(Akkil et al., 2026; Habdank, 2026).

**Evaluation awareness and probing.** Linear probes over residual-stream
activations reliably classify a model's internal testing-vs-deployment context
(Nguyen et al., 2025), and generalization hacking shows reward maximization can
diverge from genuine internalization (Xiao & Phuong, 2026). This *method* directly
validates our readout: an external probe can decode a latent context representation
separable from the policy. The construct caution — that probe "awareness" can track
prompt *format* rather than context (Devbunova, 2026) — is exactly the leakage
failure mode our audit battery excludes.

**Substrate artifacts in physics.** Beane, Davoudi & Savage (2014) derive
observable consequences of a simulated universe on a space-time lattice
(rotational-symmetry breaking near the GZK cutoff), the formal analogue of
detecting discretization (L1) or drift (L2); the limitation is reliance on external
physicists rather than an embedded learning organism. Bostrom (2003) frames the
simulation hypothesis this work operationalizes relatively.

**Embedded agency.** Demski & Garrabrant (2019) formalize an agent reasoning about
itself as a physical subsystem of the world it models (naturalized induction) — the
theoretical requirement our embedded readout instantiates empirically.

**Model-based RL and probing methodology.** DreamerV3 (Hafner et al., 2023) is the
world-model backbone; objective-mismatch work (Lambert et al., 2020) explains why
predictive accuracy need not imply the intended latent abstraction. Our probe
discipline follows Hewitt & Liang (2019) on control tasks and selectivity; our
reliability discipline follows Henderson et al. (2018), Agarwal et al. (2021), and
Chan et al. (2020); L0 equivalence follows Lakens (2017); the survival agent uses
A2C (Mnih et al., 2016) with GAE (Schulman et al., 2016). Non-invasive,
low-capacity frozen probing is preferred because adaptation-based probing can
corrupt the very latent physics being measured.

**Novelty posture.** The defensible claim is *first controlled demonstration in
this specific ALife setting*, not *first ever simulation detection*; a head-to-head
cross-reference against ALife proceedings and the evaluation-awareness citation
graph remains a required pre-submission check.

---

## 8. Reproducibility statement

All experiments are deterministic given their seeds (dependencies: `numpy`,
`scikit-learn`, `matplotlib`, `torch`). The reproducibility contract has three
layers:

- **Artifacts.** Every published number is promoted from a gitignored full-run
  bundle into a committed `artifacts/*.json` by a named script (e.g.
  `promote_ab_summaries.py`, `promote_heldout_artifact.py`, `promote_cg_rescore.py`,
  `promote_expC_summary.py`), each embedding the run's config fingerprint and git
  commit.
- **The gate.** `scripts/audit_stats_recheck.py` re-verifies every inventory number
  against the committed artifacts, so a document-to-artifact drift fails the check.
  Determinism was re-confirmed four independent times for the L3 arc (each held-out
  and cross-recipe run regenerated its standard pools bit-identically and
  reproduced the published 0.752).
- **Preregistrations and corrections log.** `docs/PREREGISTRATION.md`,
  `PREREGISTRATION_Bv3.md`, `PREREGISTRATION_L3.md`, and `PREREGISTRATION_C.md` hold
  the frozen estimands and decision rules; their section-12 deviation logs and the
  append-only correction notes in `docs/FINDINGS.md` (the L3 common-garden
  re-score, the Experiment C invalidation-and-re-run) are the public corrections
  log. Invalidated numbers are retained as historical record with correction notes
  above them.

Core modules live under `itasorl/` (`world.py`, `patch_of_earth.py`, `agent.py`,
`experiment_a.py` / `experiment_b.py` / `experiment_b2.py`, `surrogate_l3.py`,
`neuroevolution.py`, `experiment_c.py`, `behavior_audit.py`, `stats.py`); run
commands and the full manifest are in `README.md`.

---

## Appendix A: Claims-to-artifacts

Every quantitative claim in the paper, its number, and the committed artifact it
traces to. (Mirrors the `docs/PAPER_OUTLINE.md` inventory; rows renumbered.)

| # | Claim | Number | Findings § | Artifact |
|---|-------|--------|-----------|----------|
| 1 | L0 control at chance (oracle) | AUROC 0.523 | 2.1 | `artifacts/expA/summary.json` |
| 2 | L1 detectable (oracle) | 1.000 | 2.1 | `artifacts/expA/summary.json` |
| 3 | L2 detectable (oracle) | 0.993 | 2.2 | `artifacts/expA/summary.json` |
| 4 | L2 not encoded (recurrent state) | 0.510 ± 0.039 @ drift 0.45 | 3.1 | `artifacts/expB/summary.json` |
| 5 | L2 surprise channel weak | 0.596 ± 0.007 | 3.2 | `artifacts/expB/summary.json` |
| 6 | No liftoff with horizon | 0.48–0.51 across 0/8/16 | 3.3 | `artifacts/expB/summary.json` |
| 7 | L2 null robust (nonlinear probe) | 0.482 ± 0.031 | 3.4 | `artifacts/expB/summary.json` |
| 8 | Survival coupling does not rescue L2 | 0.523 ± 0.045 (replication) | 9 | `artifacts/expB2/expB2_results.json` |
| 9 | B-v3 regime negative at scale | 0.610, 90% CI [0.585, 0.634], n=10 | 7.1 | `artifacts/expB2/bv3_n10_summary.json` |
| 10 | L2 capacity ceiling below bar | 0.596, 90% CI [0.577, 0.616], n=10 | 7.1 | `artifacts/expB2/sysid_ceiling_n10_summary.json` |
| 11 | L3 gate frozen in-band | oracle 0.928, floor 0.483 | 10.1 | `PREREGISTRATION_L3.md` §12 |
| 12 | L3 encoded by survival only | 0.752, t 90% CI [0.698, 0.807], 8/10 | 10.2 | `artifacts/expB2/behavior_audit_l3_h8_traces.json` |
| 13 | L3 predictor baseline near chance | 0.573 [0.546, 0.599] | 10.2 | `…_h8_traces.json` |
| 14 | L3 untrained floor at chance | 0.488 [0.461, 0.514] | 10.2 | `…_h8_traces.json` |
| 15 | Reward leak clean | 0.541, clean 10/10 | 10.3 | `…_h8_traces.json` |
| 16 | No survivorship asymmetry | 0 early deaths, 110/110 all pools | 10.3 | `PREREGISTRATION_L3.md` §12 |
| 17 | Behavior alone decodes world | trace 0.803 [0.763, 0.840] | 10.4 | `…_h8_traces.json` |
| 18 | Behavior-independent component | 0.726 [0.685, 0.765], 9/10; quad 0.721 | 10.4 | `…_h8_traces.json` |
| 19 | Control neither manufactures nor spares signal | untrained resid 0.498; predictor 0.574 | 10.4 | `…_h8_traces.json` |
| 20 | Second capacity: behavior-independent signal replicates | 0.722 [0.678, 0.763], 8/10; quad 0.704 | 10.5 | `…_h7_traces.json` |
| 21 | Second capacity: dissociation NOT met | survival 0.737 vs predictor 0.714; lead +0.023 < +0.05 | 10.5 | `…_h7_traces.json` |
| 22 | Gate 0 re-validated per capacity | h7 oracle 0.922, floor 0.566; h8 regression 0.928/0.482 | 10.5 | `PREREGISTRATION_L3.md` §12 |
| 23 | Held-out capacity-variant transfer passes (same recipe/data) | survival 0.773 [0.722, 0.824], 9/10 vs untrained 0.569 | 10.6 | `artifacts/expB2/heldout_l3_h8_summary.json` |
| 24 | Common-garden: re-scored, MODEST PERSISTENT (both directions pass) | cg_tail 0.666 forward / 0.684 reverse; late tail 0.586/0.577 | 10.6.1 | `artifacts/expB2/heldout_l3_h8_cg_rescore.json` |
| 25 | Reverse transfer (coarse→subtle) FAILS | survival 0.638, rule fails | 10.6 | `artifacts/expB2/heldout_l3_h7_reverse_summary.json` |
| 26 | Cross-recipe transfer GENERALIZES (the generalization claim) | survival 0.684, t 90% CI [0.654, 0.715], 7/10 vs untrained 0.548; margin +0.034 | 10.7 | `artifacts/l3_crossrecipe/summary.json` |
| 27 | Exp C validated null; H3 resolved negative | contrast −0.002, t-CI [−0.013, +0.009]; final treat AUROC 0.509 | 13.D | `artifacts/expC/emergence_pilot_summary.json` |

*Note on the two 0.684 values (kept distinct):* row 26's 0.684 is the
**cross-recipe transfer** AUROC (`transfer_rff_target`, §10.7); row 24's 0.684 is
the **reverse-direction common-garden tail** AUROC (`cg_tail`, §10.6.1). They are
different quantities that happen to share a value.

---

## References

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G.
(2021). Deep reinforcement learning at the edge of the statistical precipice. In
*Advances in Neural Information Processing Systems 34* (pp. 29304–29320).
https://arxiv.org/abs/2108.13264

Akkil, Kokku, Vikram, Abuelsaad, Vempaty & Nitta (2026). *Emergence World: A
platform for evaluating long-horizon multi-agent autonomy* (arXiv:2606.08367).

Beane, S. R., Davoudi, Z., & Savage, M. J. (2014). Constraints on the universe as
a numerical simulation. *The European Physical Journal A, 50*(9), 148.
https://doi.org/10.1140/epja/i2014-14148-0

Bostrom, N. (2003). Are you living in a computer simulation? *The Philosophical
Quarterly, 53*(211), 243–255. https://doi.org/10.1111/1467-9213.00309

Chan, S. C. Y., Fishman, S., Canny, J., Korattikara, A., & Guadarrama, S. (2020).
Measuring the reliability of reinforcement learning algorithms. In *International
Conference on Learning Representations*. https://arxiv.org/abs/1912.05663

Demski, A., & Garrabrant, S. (2019). *Embedded agency* (arXiv:1902.09469).

Devbunova (2026). *Is evaluation awareness just format sensitivity? Limitations of
probe-based evidence under controlled prompt structure* (arXiv:2603.19426).

Habdank (2026). *A testable framework for AI alignment: Simulation theology as an
engineered worldview for silicon-based agents* (arXiv:2602.16987).

Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering diverse
domains through world models* (arXiv:2301.04104).

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018).
Deep reinforcement learning that matters. In *Proceedings of the 32nd AAAI
Conference on Artificial Intelligence* (pp. 3207–3214).
https://arxiv.org/abs/1709.06560

Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control
tasks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 2733–2743).
https://doi.org/10.18653/v1/D19-1275

Lakens, D. (2017). Equivalence tests: A practical primer for t tests,
correlations, and meta-analyses. *Social Psychological and Personality Science,
8*(4), 355–362. https://doi.org/10.1177/1948550617697177

Lambert, N., Amos, B., Yadan, O., & Calandra, R. (2020). Objective mismatch in
model-based reinforcement learning. In *Proceedings of the 2nd Conference on
Learning for Dynamics and Control* (PMLR Vol. 120, pp. 761–770).
https://arxiv.org/abs/2002.04523

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver,
D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement
learning. In *Proceedings of the 33rd ICML* (PMLR Vol. 48, pp. 1928–1937).
https://arxiv.org/abs/1602.01783

Nguyen, J., Hoang, K., Attubato, S., & Hofstätter, F. (2025). *Probing (and
steering) evaluation awareness of language models* (arXiv:2507.01786).

Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).
High-dimensional continuous control using generalized advantage estimation. In
*International Conference on Learning Representations*.
https://arxiv.org/abs/1506.02438

Xiao & Phuong (2026). *Generalization hacking: Models can game reinforcement
learning by preventing behavioral generalization* (arXiv:2606.12016).
