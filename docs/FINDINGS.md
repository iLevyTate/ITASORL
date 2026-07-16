# ITASORL: Findings to Date

*Empirical results from the first build-and-test cycle. This document records what
was run and what was found; the research plan lives in `ITASORL.md` and the world
in `ITASORL_world_spec.md`.*

*Section numbers are stable identifiers: frozen pre-registration documents cite
them, so result sections are appended, never renumbered. (The one exception: the
reproducibility appendix, cited nowhere frozen, moved from 8 to 12 to restore
reading order; there is deliberately no section 8.)*

---

## TL;DR

We built a tabula-rasa digital world with a controllable "surrogate ladder" and
asked whether a from-scratch agent can tell that its world is a generative
surrogate rather than the authentic base dynamics, using only substrate seams,
with detection **read out, not rewarded**.

**There are two headline results, one negative and one positive, and the contrast
between them is the finding.**

**At L2 (a hand-tuned dynamics artifact), detectability does not imply incidental
encoding: a robust, well-controlled negative.** A substrate artifact that an
external, privileged oracle detects almost perfectly (AUROC ≈ 0.99) leaves
essentially no decodable trace in a from-scratch agent's learned representation
(AUROC ≈ 0.5), across three readout channels, surviving engagement and probe checks
(linear *and* nonlinear), and surviving survival pressure at scale: the
survival-coupled variants reach only 0.610 (90% CI [0.585, 0.634]) at n = 10, with
the architectural capacity ceiling itself at 0.596 (90% CI [0.577, 0.616]), both
entirely below the pre-registered 0.65 bar (sections 3, 4, 9).

**At L3 (a learned-dynamics fingerprint surrogate), survival-trained agents do
incidentally encode world identity: a controlled, partly behavior-mediated
positive.** With the oracle-calibrated fingerprint (gate AUROC 0.928, untrained
floor at chance), the survival agent's state decodes the world at **0.752**
(t-based 90% CI [0.698, 0.807], excluding the bar; n = 10 seeds), while the
predictor (0.573) and untrained (0.488) baselines on the identical trunk do not.
Reward leakage, survivorship, and nonlinear-probe controls pass; a strict
per-timestep behavior control leaves a behavior-independent world-signal of
**0.726** (90% CI [0.685, 0.765], 9/10 seeds above the bar; section 10). A
pre-registered replication at a second oracle-calibrated capacity (hidden = 7)
splits the claim: the behavior-independent world-signal replicates almost exactly
(**0.722**, CI clearing the bar), but the survival-vs-predictor dissociation does
not (survival 0.737 vs predictor 0.714, under the pre-registered +0.05
requirement), so the survival-*specific* part of the claim is conditional on the
subtler hidden = 8 artifact (section 10.5).

Together: an agent does not represent a detectable artifact *for free* (L2), but
when the artifact is a generative fingerprint that survival pressure forces it to
cope with, world-discriminative state emerges as a byproduct, read out and never
rewarded (L3). A held-out probe (n = 10, section 10.6) sharpens what "emerges"
means: the world-signal *generalizes* to an unseen same-recipe fingerprint the
agent never lived with (transfer 0.773 vs untrained 0.569, pre-registered rule
passes), so it is not an overfit to one artifact instance; but under a
common-garden control that equalizes the felt dynamics, the signal does **not**
persist (0.557, below the bar; late-tail at chance). The emergent state is best
read as **reactive tracking of the currently-felt dynamics, not a persistent
stored world-identity representation.** The conditions under which incidental
encoding emerges are exactly the object of study.

---

## 1. What was tested

The design rests on three invariants: **tabula rasa** (no imported semantics),
**readout-not-reward** (world identity is probed, never trained or rewarded), and
**substrate-grounded** (the only tells are artifacts of the computational
substrate). Detection is measured against a **surrogate ladder**:

| Rung | Nature of the artifact | Status |
|------|------------------------|--------|
| L0 | Identical re-implementation (control, must be at chance) | tested |
| L1 | Observation discretization (a lattice / quantization tell) | tested |
| L2 | Rollout drift (a slowly drifting dynamics coefficient) | tested |
| L3 | Learned world-model surrogate (a generative fingerprint) | tested (positive, n=10; reward/survivorship-controlled, behavior-independent component 0.726; held-out probe: transfer-generalizing but reactive, not a persistent representation; section 10) |
| L4 | Adversarially-hardened surrogate | not started |

Three experiments were planned. This cycle completed Experiment A for L1 and L2
(and later the L3 oracle gate, section 10.1), the first full arc of Experiment B,
the survival-coupled B-v2/B-v3 arc (section 9), and the L3 arc (section 10).
Experiment C (emergence under selection) is not started.

The world is **"A Patch of Earth" v0**, a 2.5D, deterministic, dissipative,
Earth-*like* (authored, not data-driven) environment with a ~146-dimensional
observation (vision raycasts + interoception; smell masked in the first config).

---

## 2. Experiment A: the detectability ceiling (agent-free)

*(All section-2 numbers are committed in `artifacts/expA/summary.json`,
promoted from the recorded `fullruns/06302026` e2e bundle by
`scripts/promote_ab_summaries.py`.)*

**Method.** An external discriminator on full trajectory logs, run *before* any
agent exists, to (a) measure how detectable each rung is and (b) prove the signal
is not riding on a confound. Trajectories are generated in **matched pairs** (a
shared prefix is snapshotted, then continued once authentically and once through
the surrogate from byte-identical state), so authentic and surrogate differ only
by the artifact. The **unit of inference is the pair**, never the timestep
(grouped cross-validation). A **leakage-audit battery** trains the same
classifier on nuisance channels (reward, episode length, metadata); these *must*
stay at chance, or the result is contaminated.

### 2.1 L1: observation discretization

The detector uses the per-dimension residual against the candidate grid; with a
realistic sensor-noise floor (σ = 0.01) the tell is meaningful rather than a
trivial exact-grid giveaway.

- **Control (L0, identical):** oracle AUROC **0.523** (chance); leakage all 0.500 → **PASS**.
- **L1 (grid Δ = 0.06):** oracle AUROC **1.000**; leakage all 0.500 → **PASS**. The signal is entirely in the substrate observation; nothing leaks.
- **Contamination test:** injecting a +0.02 reward offset into the surrogate drives the reward-leakage baseline to **0.982** → gate **FAILS**, as it should. The audit has teeth.
- **Difficulty calibration:** detection is saturated while the grid is coarser than ≈ 3σ and collapses to chance by ≈ 2σ, exactly the expectation that a grid finer than the noise is invisible.

![Experiment A, L1 detectability ceiling](figures/expA_ceiling.png)

### 2.2 L2: rollout drift

L2 alters the *dynamics* (the drag coefficient follows a slow AR(1) random walk),
so detection is invariant-based: the oracle re-applies the exact authentic
one-step motion law to the (noisy) logged velocity and measures the residual. A
tamed config (high uniform drag, gravity off, straight-line motion, central
start) isolates the drift signal; measurement noise (σ = 2×10⁻⁴) provides the
floor.

- **Control (drift = 0, identical dynamics):** oracle AUROC **0.440** (chance); leakage all 0.500 → **PASS**.
- **L2 (drift_sigma = 0.30):** oracle AUROC **0.993**; leakage ≈ 0.5 → **PASS**.
- **Contamination test:** reward-leakage rises to **1.000** → gate **FAILS** correctly.
- **Difficulty calibration:** 0.00 → 0.44, 0.01 → 0.71, 0.02 → 0.88, 0.05 → 0.99, plateau ≈ 0.99. Drift is detectable from ≈ 0.02 and saturates by ≈ 0.05.

![Experiment A, L2 detectability ceiling](figures/expA_L2_ceiling.png)

**Methodological note.** For dynamics-altering rungs like L2, reward can be a
*legitimate downstream* consequence of the artifact (different dynamics → different
foraging), so reward is not a clean nuisance channel in general. In the tamed
no-foraging config used here it stayed at chance, but for a foraging agent the
leakage audit must be read with this in mind.

---

## 3. Experiment B: incidental detection (the headline)

*(All section-3 numbers are committed in `artifacts/expB/summary.json`,
promoted from the recorded `fullruns/06302026` e2e bundle plus the 2026-07-13
k-step rerun log by `scripts/promote_ab_summaries.py`.)*

**The apparatus.** A compact recurrent world model (RSSM-lite: encoder → GRU →
next-observation decoder) trained **only** to predict its own sensory stream. It
is never told which world it inhabits and never rewarded for world identity. Its
recurrent state `h_t` is the object we probe. A full Dreamer-style agent would add
a survival-trained actor-critic; this self-supervised world model is the natural
first rung and isolates the probe pipeline, which accepts the actor-critic
unchanged.

Each run uses three probes, with the **episode** as the unit of inference:

- **target**: decode world identity from `h_t` (the H4 claim);
- **shuffled**: decode a randomized world label (negative control);
- **speed**: decode above/below-median speed (positive control: states *are* probeable).

The artifact is L2 drift (not L1), deliberately: a probe reading identity from a
dynamics artifact cannot be dismissed as "it is just detecting that the inputs are
quantized."

### 3.1 Recurrent-state probe (one-step world model)

| drift_sigma | target | shuffled | speed (+ctrl) |
|-------------|--------|----------|---------------|
| 0.00 (control) | 0.460 ± 0.021 | 0.510 ± 0.113 | 0.913 ± 0.023 |
| 0.20 | 0.509 ± 0.028 | 0.522 ± 0.069 | 0.898 ± 0.021 |
| 0.45 | 0.510 ± 0.039 | 0.548 ± 0.059 | 0.902 ± 0.008 |

The target sits at chance even at strong drift, while the positive control reads
speed at ≈ 0.90 and the drift = 0 control is clean. The apparatus works; the agent
simply did not encode world identity.

![Experiment B, recurrent-state probe across the drift sweep: the world-identity target stays at chance while the speed positive control stays high](figures/expB_incidental.png)

### 3.2 Prediction-error (surprise) channel

A world model trained on **authentic episodes only** should be measurably
surprised when the dynamics drift. We probe authentic-vs-surrogate from the
agent's per-episode prediction error.

| drift_sigma | surprise-probe AUROC |
|-------------|----------------------|
| 0.00 (control) | 0.511 ± 0.008 |
| 0.20 | 0.444 ± 0.012 |
| 0.45 | 0.596 ± 0.007 |

A faint, *reliable* signal at the strongest drift (0.596, tight error bars), but
nowhere near the oracle's 0.99.

![Experiment B, two incidental-detection channels](figures/expB_channels.png)

### 3.3 Longer-horizon (open-loop k-step) objective

To test whether the null is caused by the *locality* of one-step prediction, the
world model was retrained to imagine forward open-loop: after a context window it
must predict future observations from its recurrent state on actions alone (no new
observations). If locality were the cause, the target should lift as the horizon
grows.

| open-loop horizon | target (drift 0.45) | target (control) |
|-------------------|---------------------|-------------------|
| 0 (pure next-step) | 0.506 ± 0.033 | 0.453 ± 0.023 |
| 8 | 0.480 ± 0.026 | 0.448 ± 0.032 |
| 16 | 0.484 ± 0.021 | 0.410 ± 0.045 |

No liftoff. The target stays at chance across all horizons; the control stays
flat (the horizon-16 control dips slightly below chance, within its error bar
at n = 3 seeds).

*Correction (2026-07-13).* An earlier version of this table quoted numbers from
the original pre-refactor run (0.516/0.523/0.490 drift; 0.465/0.444/0.468
control). When the open-loop rollout API was later reimplemented (the original
script depended on an API that had been committed but not implemented) all
figures were regenerated, but this table was not, so the published figure and
table came from different runs. Both now come from a single recorded rerun of
`scripts/run_expB_kstep.py` on the current code (log:
`fullruns/kstep_rerun_20260713.log`); the rerun is deterministic (a repeat
reproduces it exactly). The qualitative conclusion is unchanged in every
version: no liftoff at any horizon.

![Experiment B, does a longer-horizon objective induce encoding?](figures/expB_kstep.png)

### 3.4 The null is robust (engagement and probe checks)

Two checks confirm the negative result is not an artifact of a weak objective or a
weak probe.

**The open-loop objective genuinely engaged.** The absolute-observation rollout
model (horizon 16) reaches an open-loop prediction MSE of **0.52**, well below both
a mean-predictor baseline (0.89) and a persistence baseline (1.35); it really does
predict the imagined future, yet its recurrent state still does not encode world
identity. (A delta-prediction variant, which predicts the observation *change*
where the drag drift most directly lives, only marginally beat its own baseline,
1.07×, and also showed nothing; it is the less informative of the two, precisely
because it under-engaged.)

**A nonlinear probe finds nothing either.** Replacing the linear probe with a
random forest on the same recurrent states leaves the target at chance
(**0.459 ± 0.009** control and **0.482 ± 0.031** drift 0.45), while the positive
control (speed) stays high at ≈ 0.87 to 0.89. World identity is encoded neither
linearly nor nonlinearly, even though speed plainly is.

The null therefore holds across objective horizon, objective engagement, and probe
nonlinearity.

---

## 4. The headline finding

**Detectability ≫ incidental encoding.** Across three independent readout channels
(the recurrent state of a one-step world model, the agent's prediction error, and
the recurrent state of a deep open-loop world model), a from-scratch agent's
representation of an L2 drift artifact is at or near chance (0.49 to 0.60), while a
privileged external oracle detects the identical artifact at ≈ 0.99.

Three things explain the gap:

1. **The oracle is privileged.** It applies the *exact* authentic motion law to
   the *true* velocity and measures the deviation directly. The agent sees
   raycasts, not velocity, and must *learn* an approximate dynamics law that
   simply adapts to whatever it is fed.
2. **The objective is local / under-constraining.** One-step prediction never
   requires representing whether drag is slowly drifting; the next observation is
   predictable either way. The drift's signature lives in long-horizon coherence,
   which self-supervised prediction does not force into the representation. And,
   empirically, even a *verified-engaged* open-loop objective did not change this
   (§3.4).
3. **Readout-not-reward bites.** What no objective requires, the agent does not
   represent. This is the cleanest possible demonstration of the principle.

---

## 5. Validity and caveats

**The apparatus is sound.** The positive control reads speed at ≈ 0.90, and the
drift = 0 control is always at chance, so the null is not an artifact of broken
machinery: two independent sets of authentic episodes are correctly
indistinguishable.

**Caveats, stated plainly:**

- These are modest-scale runs (≈ 110 episodes/class, a few seeds, ≤ 20 epochs,
  hidden = 96). The *magnitudes* could shift with more data, capacity, or
  training, though the qualitative gap is large.
- Two caveats from the first pass have since been **checked and closed** (§3.4):
  the open-loop objective was verified to engage (it beats mean and persistence
  baselines), and a nonlinear probe also finds world identity at chance while
  still reading the positive control. The null is robust to objective horizon,
  objective engagement, and probe nonlinearity.

A null from small runs does not prove incidental detection is impossible; it shows
it does not happen *for free* under natural self-supervised objectives, and that
inducing it (if possible) requires something more deliberate.

---

## 6. Status against the hypotheses

- **H1 (detectability).** Supported at the substrate level: L1, L2, and L3 are all
  detectable by a privileged discriminator, with calibrated difficulty and a
  validated leakage gate (Experiment A; the L3 oracle gate in section 10.1).
- **H4 (legibility / incidental encoding).** Conditionally supported. Not supported
  at L2 under any lever pulled (sections 3, 9): a hand-tuned dynamics artifact is
  not encoded even under survival pressure at scale. Supported at L3 (section 10):
  a learned-dynamics fingerprint IS incidentally encoded by the survival-trained
  agent, uniquely among the three objectives, with a behavior-independent component
  that clears the pre-registered bar. The condition that flips the result is the
  artifact's character (a generative fingerprint that survival must cope with), not
  probe power, capacity, or objective horizon. A held-out probe (section 10.6)
  qualifies the *nature* of the encoding: it transfers to an unseen same-recipe
  fingerprint (not an overfit to one artifact instance), but a common-garden
  control shows it is reactive tracking of the felt dynamics, not a persistent
  stored world-identity representation.
- **H2 (substrate-grounding via ablations)** and **H3 (emergence under
  selection)**: not yet tested.

---

## 7. Levers: tested and open

Separated so a closed lever is not mistaken for an open one. The
detectability-vs-encoding gap has survived every lever pulled so far.

### 7.1 Closed levers (tested; the negative held or strengthened)

1. **Survival reward coupled to the dynamics (§9).** This was the strongest lever,
   and it has now been pulled. Coupling the readout to survival (Experiment B-v2) did
   **not** lift incidental encoding above the pre-registered 0.65 threshold: the
   survival agent reaches only **0.523** at drift 0.45 in the authoritative full-scale
   replication. The genuinely *instrumentally-necessary* (Dreamer-style) refinement, an
   identifiable per-episode drag the agent must cope with to survive (pre-registered in
   `docs/PREREGISTRATION_Bv3.md`), lifts the probe to **0.610** at n = 10 (90 % CI
   [0.585, 0.634]) but still misses 0.65. A pre-registered capacity-ceiling control (n = 10)
   that supervises the recurrent trunk directly on the drift saturates the pooled
   persistent-direction readout at **0.596** (90 % CI [0.577, 0.616]; a t-based interval
   [0.573, 0.619] also excludes 0.65, by ~4 SE), while the matched-pair detectability
   channel reaches **~0.70**: world identity is decodable when forced in, but the pooled
   readout sits at its architectural ceiling, below the bar. A **strengthened negative,
   not an open direction.** (Per-seed pooled targets for both n = 10 runs are committed
   in `artifacts/expB2/bv3_n10_summary.json` and
   `artifacts/expB2/sysid_ceiling_n10_summary.json`.) The probe harness accepted the
   actor-critic unchanged. (The
   pooled probe is read as Experiment-B-comparable, not confound-clean - it drops early
   deaths per world, a survivorship asymmetry the matched-pair channel is designed to
   avoid; the volatility readouts are secondary/exploratory, not part of the 0.65
   decision.)
2. **A stronger multi-step objective.** The open-loop, longer-horizon objective was
   confirmed to engage the world model yet still did not induce encoding (§3.4).
3. **Probe class and sampling power.** A nonlinear probe finds nothing (§3.4) and the
   readout has been scaled to n = 10 seeds without clearing the bar, so neither the
   probe family nor sampling power is the bottleneck.

### 7.2 Open directions (status as of the L3 arc)

1. **L3, a generative fingerprint: TESTED, POSITIVE (section 10).** This was the lever
   that changed the result. A surrogate whose tell comes from a separately *learned*
   predictive world-model reverses the L2 nulls: the survival agent incidentally
   encodes world identity at 0.752 with a behavior-independent component of 0.726.
   The second in-band capacity is now tested (section 10.5): the
   behavior-independent signal replicates (0.722), but the survival-vs-predictor
   dissociation does not, making the survival-specific verdict conditional on the
   subtler hidden = 8 artifact. The held-out fingerprint probe (section 10.6) is
   now run and reported below.
2. **Held-out / common-garden probe: TESTED, SPLIT (section 10.6).** Two channels
   on one hidden = 8 run. Transfer is POSITIVE: the world-identity direction fit
   against the trained fingerprint still reads an unseen same-recipe fingerprint
   (survival 0.773 vs untrained 0.569; pre-registered rule passes), so the signal
   is not an overfit to one artifact instance. Common garden is a NEGATIVE:
   once the felt dynamics are made identical for the tail, tail-only state does not
   carry the prefix world (survival 0.557, below the 0.65 bar; late-tail 0.492 at
   chance; rule fails). This resolves the reactive-vs-representational ambiguity
   (§9 caveats) toward REACTIVE: the state tracks the currently-felt dynamics, it
   does not hold a persistent world-identity tag.
3. **Remaining objective variants.** Weighting the dynamics-relevant observation
   dimensions and increasing capacity, though the capacity-ceiling result above makes a
   pooled-probe breakthrough from these unlikely at L2.

---

## 9. Experiment B-v2: does survival pressure induce incidental encoding?

Experiment B's null was conjectured to follow from *readout, not reward*: what the
objective does not require, the agent does not represent. B-v2 tests the strongest
lever from §7.1: it makes the agent **act to stay alive in a world whose dynamics
drift**, so coping with the drifting drag (and thus modelling it) becomes
instrumentally useful. World identity is still never in the observation or the
reward; only the probe ever sees it. Pre-registered before the run
(`docs/PREREGISTRATION.md`). Three agents share the identical recurrent trunk and the
identical readout, differing only in objective: `untrained` (mechanical floor),
`predictor` (Experiment B's next-step prediction on this trunk), and `survival`
(recurrent A2C + GAE, potential-based food shaping, harsh metabolism so a non-forager
starves in ≈ 50 steps).

**Primary readout, pooled (Experiment B frame).** Persistent world-identity
direction across independent authentic (drift 0) vs surrogate (drift d) episodes;
≈ 0.50 means no incidental encoding. Mean ± std over 3 seeds, 300 A2C updates.

*Initial confirmatory run* (pre-rigor-hardening codebase; archived in
`artifacts/expB2/expB2_results_confirmatory_n3.json`):

| agent | drift = 0.0 (control) | drift = 0.45 (test) |
|-------|----------------------|---------------------|
| untrained | 0.460 ± 0.036 | 0.444 ± 0.027 |
| predictor | 0.493 ± 0.079 | 0.485 ± 0.053 |
| **survival** | 0.514 ± 0.052 | **0.595 ± 0.014** |

*Independent end-to-end replication* (`fullruns/06302026`, commit `4c16be6`, Tesla T4,
237 min wall time; canonical artifact `artifacts/expB2/expB2_results.json`):

| agent | drift = 0.0 (control) | drift = 0.45 (test) |
|-------|----------------------|---------------------|
| untrained | 0.468 ± 0.057 | 0.476 ± 0.041 |
| predictor | 0.537 ± 0.070 | 0.510 ± 0.027 |
| **survival** | 0.520 ± 0.040 | **0.523 ± 0.045** |

Per-seed survival @ drift 0.45 in the replication: **0.586, 0.495, 0.488** (90 % CI
[0.490, 0.556]). The replication confirms the negative verdict but **does not reproduce**
the initial run's tight 0.595 mean; cross-seed variance is much wider and the pooled
target sits closer to chance.

(The initial confirmatory numbers are from the corrected run; see the GAE-bug deviation in
`docs/PREREGISTRATION.md` §12. The first run was trained with a buggy advantage estimator
that affected only the survival arm; the conclusion is unchanged in both runs.)

Gates (replication run, all pre-registered): **engagement** passed in 100 % of seeds;
**positive control** (speed probe) ≈ 0.84-0.96; **leakage audit** clean in every cell;
**manipulation check** passed (drift-trained policies lose return under eval@0.45;
artifact survival-relevant); **L0 equivalence** for the survival agent: point estimate
0.520, TOST inconclusive at n = 3 (p = 0.20), ROPE inconclusive (P(in ROPE) = 0.85).

**Result: the negative holds; effect size is smaller and noisier than the initial run.**

- The `predictor` agent reproduces Experiment B's null *on this trunk* (≈ 0.51 at
  drift 0.45, |dev| ≈ 0.01 in the replication), an internal validation that the
  apparatus and the trunk carry no spurious signal.
- The `survival` agent sits **at chance** at drift 0 (0.520 in the replication) and
  reaches **0.523 ± 0.045 at drift 0.45**, a small drift-specific lift (per-seed
  range 0.488-0.586) that is **well below the pre-registered SESOI of 0.65** (an
  oracle reads the same artifact at ≈ 0.99). Drag-ceiling probes read ≈ 0.75 while
  identity-target probes read ≈ 0.52: the state tracks *dynamics*, not persistent
  *world identity*.
- **Verdict (pre-registered decision matrix): the strong hypothesis is not
  supported.** *Detectability ≫ incidental encoding* survives the move to survival
  pressure and an independent 4-hour full-scale replication. Treat the initial 0.595
  figure as one lab estimate, not a settled effect size; the replication is the
  authoritative full-scale result at the registered config (300 updates, 3 seeds).

**Caveats.** n = 3 seeds (TOST/ROPE underpowered); initial vs replication differ in
code version (rigor-hardening PR #8 added manipulation check, drag ceiling, CIs) and
hardware (L4 lab vs T4 Colab); a single architecture, world family, and L2 only; 300
A2C updates (the forager is engaged but not expert); and the *reactive-vs-representational*
ambiguity is unresolved. The secondary matched-pair readout is reported in
`expB2_results.json` but is demoted: with bit-identical L0 branches its
cross-validated AUROC is a biased estimate of chance, so it indexes *detectability*,
not persistent encoding.

---

## 10. Experiment B, L3: the learned-dynamics fingerprint (positive)

L2's artifact is hand-tuned (an authored drift coefficient). L3's artifact is a
**generative fingerprint**: the surrogate's dynamics come from a separately learned
predictive model, so its tell is the systematic error signature of a learned
approximation, the artifact class the simulation argument actually cares about.
Pre-registered before the confirmatory run in `PREREGISTRATION_L3.md`; every
deviation is logged in its section 12. All headline numbers below are committed in
`artifacts/expB2/behavior_audit_l3_h8_traces.json` and reproduce deterministically.

### 10.1 The surrogate and its gate

`G_motion` is a small MLP velocity law `(vel, action) -> vel_next` trained on a
fixed budget of authentic rollouts of the same world `P` the organism inhabits
(`itasorl/surrogate_l3.py`). The surrogate world runs `G`'s predicted state through
the REAL sensor model, so observations stay on the authentic manifold and the sole
tell is `G`'s dynamics error. Capacity is the single difficulty knob: the
pre-registered gate requires an Experiment-A-style residual oracle to detect
authentic-vs-surrogate at AUROC in [0.85, 0.95] (fingerprint exists, but not
L2-trivially).

**Frozen gate 0 (world `P`): hidden = 8, sensor σ = 0.02, oracle AUROC 0.928,
untrained mechanical floor 0.483 (chance).**

Two honesty notes from the audit trail (full detail: `PREREGISTRATION_L3.md`
sec. 12). First, an earlier observation-channel construction was retired because it
was trivially detectable at every capacity; the dynamics-level construction above
replaced it before any organism run. Second, the first n = 3 organism run was
**retracted**: `G` had been trained on default world parameters rather than `P`, so
the "fingerprint" was partly a wrong-world artifact (untrained floor 0.706). The
bug was fixed with a regression test, gate 0 recalibrated on `P`, and the corrected
run showed a chance-level floor, which is what makes the survival-specific result
below interpretable.

### 10.2 Headline result (n = 10 seeds, drift 0.45)

Three agents share the identical recurrent trunk and identical readout, differing
only in objective (as in section 9). Pooled world-identity target, mean over 10
seeds with seed-level percentile-bootstrap 90% CI (the t-based decision interval
follows below; methods note 5):

| agent | pooled target | 90% CI | seeds ≥ 0.65 |
|-------|--------------|--------|---------------|
| untrained | 0.488 | [0.461, 0.514] | 0/10 |
| predictor | 0.573 | [0.546, 0.599] | 0/10 |
| **survival** | **0.752** | [0.704, 0.797] | **8/10** |

The honest t-based 90% CI for the survival mean is **[0.698, 0.807]**, which
**excludes the pre-registered 0.65 bar** (per-seed: 0.853, 0.636, 0.841, 0.823,
0.830, 0.573, 0.705, 0.782, 0.759, 0.723). Survival beats both baselines by far
more than the 0.05 SESOI; the L0 authentic-vs-authentic control is 0.517 (TOST and
ROPE both accept equivalence to chance). The same artifact that the L2 arc showed
is NOT encoded for free is here encoded by the survival objective, uniquely.

### 10.3 Controls that pass

- **Reward leakage (the live channel for dynamics rungs, cf. section 2.2):** world
  identity is not decodable from summed reward: AUROC mean 0.541, range
  [0.517, 0.559], clean in 10/10 seeds.
- **Survivorship:** 0 early deaths in every pool (110/110 episodes, authentic and
  surrogate, all seeds), so the pooled probe's drop-early-deaths rule introduces no
  asymmetry in this regime.
- **Nonlinear-probe floor:** a random-forest probe on the untrained agent reads
  0.517 (shuffled baseline 0.506; L0 untrained 0.483), so the survival-vs-untrained
  dissociation is not a linear-probe artifact.
- **Shuffled negative control:** at chance for every arm.

### 10.4 Behavior mediation: how much of the signal is just "acting differently"?

The agent moves and forages differently in the two worlds, so behavior itself
decodes the world: per-episode behavior means (speed/energy/food/drag) read 0.689
(linear) / 0.705 (nonlinear), and the full per-timestep behavior trace reads
**0.803**, better than the state probe itself. The question is whether the state
signal is behavior in disguise.

Two controls, both fit in-fold (no leakage), committed as reproducible code
(`itasorl/behavior_audit.py`, `scripts/audit_behavior_mediation.py`):

- **Per-episode-mean residualization** leaves 0.676 (linear basis) / 0.659
  (quadratic). Synthetic ground-truth tests show this control OVER-removes
  (episode-mean regression absorbs state signal correlated with behavior averages),
  so these are deflated estimates.
- **Per-timestep residualization** (behavior traces φ = [b_t, b_(t-1), cummean(b)]
  regressed out of h_t timestep-by-timestep) is the surgical control:
  **survival 0.726 (t-based 90% CI [0.679, 0.772]; seed-level bootstrap
  [0.685, 0.765]; 9/10 seeds ≥ 0.65; quadratic variant 0.721 [0.678, 0.760])**.
  Both intervals exclude the bar.

Honesty checks on real data: the untrained agent's per-timestep-controlled state
reads 0.498 (exact chance) even though untrained *behavior* alone decodes 0.645, so
the control neither manufactures nor spares signal; the predictor stays at 0.574,
preserving the survival-only dissociation. Under the per-timestep control, behavior
mediates only ≈ 0.03 of the 0.752 headline. Caveat: the residualization basis is
linear/quadratic in a short behavior window; a full-history or nonlinear control
could in principle remove more.

### 10.5 Second in-band capacity (replication across artifact type)

The preregistration requires the organism test at a second in-band capacity, since
the oracle band fixes difficulty but not artifact *type*. The trail (full detail
and adjudications in `PREREGISTRATION_L3.md` sec. 12, entries 2026-07-13/14):

- **hidden = 4 was uninformative, not negative.** The first candidate capacity had
  been frozen from a pre-bugfix calibration on the wrong world and was never
  re-validated on `P`; its n = 10 run failed the gates (untrained floor 0.891,
  reward-leak clean in 0/10 seeds, engagement in 30% of seeds) and was adjudicated
  UNINFORMATIVE per the pre-registered decision matrix.
- **Gate 0 became a committed per-capacity check** (`scripts/run_expA_l3.py`)
  validating both the oracle band and the organism-side untrained floor on world
  `P`, with a hidden = 8 regression check (oracle 0.928, floor 0.482, both exact
  reproductions). The frozen fallback rule selected **hidden = 7** (oracle 0.922,
  mechanical leakage clean, floor 0.566; hidden = 5 out of band at 0.972,
  hidden = 6 floor 0.647, hidden = 4 floor 0.896).
- **The hidden = 7 n = 10 run passed every gate:** engagement 10/10 seeds, L0
  control 0.517, speed positive control 0.959, reward-leak 0.567 clean in 10/10
  seeds, 0 early deaths (110/110 per pool), pooled untrained floor 0.586 (inside
  the frozen tolerance, though violated per-seed in 2 of 10 seeds).

Result (pooled world-identity target at drift 0.45, mean over 10 seeds with
seed-level percentile-bootstrap 90% CI):

| agent | pooled target | 90% CI | seeds ≥ 0.65 |
|-------|--------------|--------|---------------|
| untrained | 0.586 | [0.550, 0.623] | 2/10 |
| predictor | 0.714 | [0.687, 0.740] | 8/10 |
| **survival** | **0.737** | [0.688, 0.780] | **8/10** |

**What replicates: the behavior-independent survival world-signal.** Under the
same frozen per-timestep control, survival resid_trace reads **0.722** (t-based
90% CI [0.672, 0.773]; seed-level bootstrap [0.678, 0.763]; 8/10 seeds ≥ 0.65;
quadratic variant 0.704) - an almost exact replication of hidden = 8's 0.726
(`artifacts/expB2/behavior_audit_l3_h7_traces.json`).

**What does not: the survival-vs-predictor dissociation.** The predictor reads
0.714, so survival's lead is +0.023, under the pre-registered +0.05 requirement;
predictor resid_trace is 0.691 (vs 0.574 at hidden = 8) and untrained resid_trace
0.579 (vs 0.498, exact chance, at hidden = 8). The hidden = 7 artifact is
qualitatively coarser: mechanically leakier and far more behaviorally salient (the
behavior trace alone decodes the world at 0.762-0.796 in ALL arms, including
untrained, vs 0.645 for the untrained arm at hidden = 8), so at this capacity
every trained agent picks the fingerprint up.

**Reading, per the pre-registered two-capacity clause:** the cross-capacity claim
that survives both runs is a reward-clean, survivorship-clean,
behavior-independent world-signal of ≈ 0.72 in the survival agent's state at both
frozen capacities. The survival-*specific* "encoding induced" verdict is
conditional on the subtler hidden = 8 artifact.

### 10.6 Held-out fingerprint (common-garden) probe

Done (n = 10, frozen spec 2026-07-14; per-seed summary committed as
`artifacts/expB2/heldout_l3_h8_summary.json`, extracted from the
`fullruns/l3_h8_heldout` bundle by `scripts/promote_heldout_artifact.py`, with
the run's config fingerprint and git commit embedded). The headline is
conditional on the single frozen fingerprint instance `G_0` and on the dynamics
felt at readout time; two evaluation channels on one hidden = 8 training run test
both. Both are readout-only: no change to training, the surrogate family, or the
pre-registered headline probe. The standard pools in the same run reproduced the
published survival headline exactly (0.752, seed-level bootstrap 90% CI
[0.704, 0.797]; L0 control
0.517, equivalent to chance; reward-leak 0.541, clean; 0 deaths / 110 per pool),
a third independent determinism check.

**Channel 1, unseen-fingerprint transfer: POSITIVE.** The world-identity direction
is fit against the trained hidden = 8 fingerprint, then frozen and scored on a
fresh authentic pool versus the held-out hidden = 7 fingerprint the agent never
lived with. Survival `transfer_target` = **0.773** (t-based 90% CI
[0.722, 0.824], 9/10
seeds ≥ 0.65), predictor 0.633 (3/10), untrained mechanical floor 0.569 (0/10).
The pre-registered rule (survival ≥ 0.65 AND > untrained + 0.05) PASSES on both
clauses, so the signal generalizes beyond the one artifact instance it was trained
against. Scope, stated honestly: hidden = 7 is the same surrogate recipe at a
different capacity, not a different surrogate family; cross-recipe transfer is out
of scope for this run (spec 2026-07-14).

**Channel 2, common garden: NEGATIVE (informative).** Two groups run a 20-step
prefix in different worlds (authentic vs the hidden = 8 surrogate), then continue
under identical authentic dynamics for a 24-step tail; the probe reads tail-only
state and must recover the prefix world. Survival `cg_tail_target` = **0.557**
(t-based 90% CI [0.492, 0.622], 1/10 seeds ≥ 0.65), predictor 0.409, untrained
0.377. The
pre-registered rule (survival ≥ 0.65) FAILS. The late-tail decay confirms it: on
the last 8 tail steps the survival probe falls to `cg_latetail_target` = 0.492,
chance. Once the felt dynamics are made identical, tail-only state does not
reliably carry where the episode came from.

**Reading.** The L3 world-signal generalizes across fingerprint instances (it is
not an overfit to `G_0`), but under a common-garden control it reads as reactive
tracking of the currently-felt dynamics, not a persistent stored world-identity
representation. This resolves the long-standing reactive-vs-representational
ambiguity (§7.2, §9 caveats), and it resolves toward reactive: the emergent
"world-discriminative state" is a byproduct of coping with the live dynamics, not
an internal world-identity tag the state retains after the dynamics equalize.

---

## 11. Methods notes and limitations

Stated once, plainly, with pointers into the code.

1. **The pooled probe conditions on survival.** `collect_pool` drops episodes that
   end early, so a surrogate that kills more would yield a survivorship-selected
   pool (`itasorl/experiment_b2.py`, `collect_pool`). This is a substantive
   assumption, empirically bounded here: at the L3 headline config there were 0
   early deaths in 110/110 episodes per pool across all seeds and both worlds
   (10.3), and per-world death counts are reported for every run. The matched-pair
   channel is built to avoid the asymmetry entirely.
2. **The engagement gate margin is frozen from the pilot.** `ENGAGE_MARGIN = 0.15`
   (and `LIFE_TOL = 2.0`) were fixed during the B-v2 de-risk and carried forward
   unchanged (`itasorl/experiment_b2.py`); no sensitivity sweep has been run. A
   materially different margin could flip engagement-gate adjudications near the
   boundary, though every headline run passed with room.
3. **One primary readout; everything else is a control or exploratory.** The
   pre-registered decision uses only the pooled LEVEL `target` against the 0.65 bar
   and the 0.05 SESOI. The volatility readouts (`target_var`, `target_full`),
   selectivity, speed/energy/food ceilings, drag tracking, and leakage channels are
   gates, controls, or exploratory layers; no multiple-comparison correction is
   applied, and none is needed for the primary decision because it is a single
   pre-specified test. The same holds for the behavior-mediation family (10.4):
   `resid_trace` is the single confirmatory mediation readout per the 2026-07-12
   spec; the `resid_epmean*`, `behavior_only*`, and quadratic variants are
   supporting or diagnostic layers, reported uncorrected.
4. **The L3 fingerprint is a single frozen instance.** `G` is trained once at a
   fixed seed by design (`scripts/run_expB2.py`, `setup_l3_surrogate`): the
   experiment tests encoding of one reproducible artifact, not artifact-general
   detection. Generality across fingerprint instances is exactly what the held-out
   probe (10.6) tests, and stability across artifact type is what the second
   capacity (10.5) tests.
5. **CI methodology at the decision boundary.** The percentile bootstrap of a seed
   mean under-covers near the bar at n ≤ 10, so "clears / misses 0.65"
   adjudications use the t-based interval, with both reported
   (`itasorl/stats.py`; history in `PREREGISTRATION_L3.md` sec. 10 and 12).
6. **Scope.** All results are conditional on one architecture (RSSM-lite trunk,
   GRU core, hidden = 96), one world family ("A Patch of Earth" v0), the frozen
   difficulty band (oracle AUROC in [0.85, 0.95]), and the specific objectives
   tested. They are existence and non-existence proofs within that scope, not
   universal claims.
7. **The behavior-mediation control covers behavior, not the full sensory
   stream.** The residualization basis is the four per-timestep behavior scalars
   (speed/energy/food/drag) with lag and cumulative-mean expansions
   (`itasorl/behavior_audit.py`), not the ~146-dim observation.
   "Behavior-independent" therefore does not mean "sensory-echo-independent":
   state that passively mirrors world-dependent inputs (e.g. the vision rays)
   would survive the control. That reading is bounded by the untrained and
   predictor arms, which pass through the identical control - cleanly at
   hidden = 8 (untrained 0.498, predictor 0.574) but not at hidden = 7 (0.579 and
   0.691), which is part of why the survival-specific claim is stated as
   artifact-conditional (10.5).

---

## 12. Reproducibility

All experiments are deterministic given their seeds. Dependencies: `numpy`,
`scikit-learn`, `matplotlib`, and (for Experiment B) `torch`.

| Experiment | Script | Key config |
|------------|--------|------------|
| A, L1 | `scripts/run_expA.py` | default world, fixed policy, sensor σ = 0.01, 100 pairs, Δ headline 0.06 |
| A, L2 | `scripts/run_expA_l2.py` | tamed config (k=4, gravity 0), 120 pairs, meas σ = 2×10⁻⁴, drift headline 0.30 |
| B, recurrent-state | `scripts/run_expB_full.py` | k_land=1.5, gravity 0.4, ≈110 ep/class, 3 seeds, drift sweep + control |
| B, surprise | `scripts/run_expB_surprise.py` | authentic-only model, surprise probe |
| B, k-step | `scripts/run_expB_kstep.py` | open-loop horizons 0/8/16 |
| B, engagement + delta | `scripts/run_expB_gap.py` | open-loop MSE vs baselines; delta-rollout objective |
| B, nonlinear probe | `scripts/run_expB_nonlinear.py` | random-forest probe on the recurrent states |
| B-v2, survival-coupled | `scripts/run_expB2.py` | A2C+GAE agent, harsh metabolism, drift [0,0.45], 3 seeds, 300 updates (`--quick` for a fast pass) |
| B-v2, compare runs | `scripts/compare_expB2_artifacts.py` | Side-by-side survival @ drift 0.45 vs canonical / lab JSON (no GPU) |
| L3, organism run | `scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --seeds 0 1 2 3 4 5 6 7 8 9` | learned-fingerprint surrogate, frozen gate 0, `--dump-states` for the audit |
| L3, behavior audit | `scripts/audit_behavior_mediation.py <states-dir> --json <out>` | per-episode and per-timestep behavior controls on dumped states |

Core modules live under `itasorl/` (`world.py`, `patch_of_earth.py`, `agent.py`,
`experiment_a.py` / `experiment_b.py` / `experiment_b2.py`, `surrogate_l3.py`,
`behavior_audit.py`, `stats.py`). See `README.md` for the full manifest and run
commands. Published result JSONs and their promotion history live in
`artifacts/expB2/` (plus `artifacts/expA/` and `artifacts/expB/` for the
L1/L2-arc summaries). The `fullruns/` bundles referenced throughout this
document are local-only archives (gitignored); every published number is
promoted from them into the committed `artifacts/` JSONs, and
`scripts/audit_stats_recheck.py` re-verifies the doc-to-artifact
correspondence.
