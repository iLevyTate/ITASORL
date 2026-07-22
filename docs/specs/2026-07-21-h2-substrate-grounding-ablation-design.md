# Design Spec - H2 Substrate-Grounding Ablation (DRAFT, not frozen)

**Status:** FROZEN for the A1 launch (2026-07-21). Section 4 mechanics verified against
source (file:line anchors inline). Open decisions (section 9) are RESOLVED with the
recommended defaults. This launch runs **A1 only** (the primary necessity sweep,
readout-only on the saved hidden=8 bundle); A2 is re-scoped to a rollout-level obs
ablation and deferred, A3 is deferred (needs a train). Runner:
`scripts/run_expH2_ablation.py`.

**Date drafted:** 2026-07-21.

## 0. One-line purpose

Close H2 for the L3 rung: show the incidentally-encoded world-identity signal
(pooled survival target 0.752 at hidden=8) is *driven by the computational substrate
artifact* - the learned velocity-law divergence - and not by any task confound, by
adding the one piece the existing audit battery is missing: a positive **necessity
ablation** that neutralizes the substrate seam and shows the signal collapses to chance.

## 1. Background - what H2 needs, and what is already banked

Project invariant (docs/ITASORL.md:26): "Substrate-grounded. Any discrimination must
load on computational artifacts of the surrogate, not on task confounds, formatting
tells, or metadata." Hypothesis (docs/ITASORL.md:115): "H2 - Substrate-grounding.
Detection is driven by computational artifacts, not task confounds - established
through strict ablations." The minimal-result bar (docs/ITASORL.md:163) asks for "an
ablation showing detection loads entirely on the structural artifact."

H2 has two halves. The **"not task confounds"** half is already largely established for
the L3 hidden=8 headline by the existing, committed audits - this spec inventories them
rather than re-running them:

| Confound ruled out | Control (committed) | Result |
|---|---|---|
| Apparatus manufactures signal | L0 authentic-vs-authentic (TOST/ROPE) | 0.517, equivalent to chance |
| Reward coupling | pooled reward-leak audit | 0.541, clean 10/10 seeds |
| Survivorship selection | per-world death counts | 0 early deaths, 110/110 both pools |
| Behavior (speed/energy/food/drag) | per-timestep residualization (10.4) | resid_trace 0.726, CI clears bar |
| Absolute position + heading | 7-channel control (10.4.1) | resid_trace 0.723, gap closed |
| Linear-probe overfit | random-forest probe on untrained | 0.517, near chance |
| Length / metadata | mechanical-channel leakage | clean at 0.5 |

What is **NOT** yet established is the **"driven by computational artifacts"** half as a
*positive necessity* claim. Every control above is of the form "control for confound X,
signal survives." None removes the **substrate artifact itself** and shows the signal
dies. That causal-necessity ablation is the H2 capstone this spec adds. Without it, a
skeptic can grant every confound control and still ask "granted it is not those things -
but have you shown it IS the substrate seam, rather than some fourth unmeasured channel?"

## 2. Hypothesis

- **H2 (necessity):** graded neutralization of the substrate seam (the G_motion
  velocity-law divergence) drives the pooled world-identity target monotonically to the
  L0 chance floor; with the seam fully removed the signal is equivalent to chance.
- **H2 (localization, secondary):** the signal loads on the substrate-derived
  observation channels (motion/velocity-sensitive dims), not on static terrain/food
  channels; masking the former collapses the signal, masking an equal count of the
  latter does not.
- **Null / failure mode:** the target stays above the floor even with the seam fully
  neutralized -> the signal is NOT substrate-grounded (it is loading on an unmeasured
  non-substrate channel), which would be a genuine and important negative for H2 and
  would reopen the "fourth channel" question.

## 3. Design principle - readout-only on the saved agents

This is an **evaluation-only** design on the frozen, saved hidden=8 agents
(`fullruns/l3_h8_heldout` bundle, the same 60 agents the held-out and cross-recipe
probes reused), mirroring the 2026-07-14 held-out/common-garden and 2026-07-15
cross-recipe specs: no retraining, no change to the surrogate family, no change to the
headline probe. VERIFIED feasible: the surrogate velocity law is attached per-world via
a single hook, `w._g_motion` (`experiment_b2.py:104-105`), which the agent never sees;
swapping in a graded law changes only the world the frozen agent is dropped into, not
the agent. This keeps H2 cheap, bit-reproducible, and free of the "new training run
changed something" risk. The saved agents were trained at the full surrogate (alpha=1);
H2 reads how their existing world-identity direction responds when the seam is dialed
down at evaluation - a dose-response of the readout, not a new training regime.

## 4. Ablation battery (mechanics verified against source)

The substrate seam has a single root cause: the per-step velocity update, authentic
`vel_next = (1 - drag*dt)*vel + a*dt` (`patch_of_earth.py:177`) versus the learned map
`vel_next = _g_motion(vel, a, drag)` (`:175`), called inside `_integrate_motion`. In
world P drag is the exact constant 1.5, so the authentic law is exactly linear and
`GMotion`'s systematic error is the whole fingerprint. Every ablation below targets that
one channel or its downstream path into the observation.

### A1 - Graded substrate-seam neutralization (PRIMARY, necessity)

Attach a graded velocity law in place of the plain `GMotion` net:

    g_alpha(vel, a, drag) = (1 - alpha) * true_law(vel, a, drag) + alpha * G(vel, a, drag)

where `true_law` is the exact arithmetic of `patch_of_earth.py:177` and `G` is the
frozen `GMotion` for that seed. `g_alpha` satisfies the same `(vel, a, drag) -> vel_next`
hook contract, so it drops into `w._g_motion` with no other change. Then run the standard
`pooled_readout` (authentic pool seed 800_000 at drift 0, surrogate pool seed 850_000
under `g_alpha`) per seed, re-fitting the probe per alpha as every pooled readout does.

- **alpha = 1.0** -> `g_alpha == G` exactly: reproduces the published 0.752 (determinism
  gate 0).
- **alpha = 0.0** -> `g_alpha == true_law` exactly: both pools now run the authentic law,
  so this IS an authentic-vs-authentic pair and must read the L0 floor (~0.517). This is
  the same construction the codebase already uses for L0 (the net is attached only when
  `drift_sigma > 0`, `experiment_b2.py:104`).
- Sweep alpha over the FROZEN grid {0.0, 0.1, 0.25, 0.5, 0.75, 1.0} (extra point near 0
  because the collapse shape near the floor is the object of interest); score the pooled
  `target` at each.
- **Prediction if substrate-grounded:** monotone non-increasing dose-response, target
  declining from ~0.752 at alpha=1 to ~0.517 at alpha=0. A non-monotone curve, or a
  floor that stays above L0, falsifies the substrate-necessity claim.

### A2 - Observation-channel localization (SECONDARY, DEFERRED from this launch)

The obs vector is 146 dims with a fixed layout (`world.py:105-111`): vision (24 rays x 5)
+ smell (12, masked to zero in v0) + intero (14). VERIFIED motion-derived channels: the
5th slot of each ray, `radial_velocity = -vel . d` (`patch_of_earth.py:253`), i.e. the 24
ray indices {4, 9, 14, ...}; plus intero `vel_x, vel_y` (obs 132-133) and `accel_x,
accel_y` (obs 144-145) - a **28-channel motion set**, with a matched 28-channel static
control set available (24 per-ray `dist` slots {0, 5, 10, ...} + 4 terrain intero dims).

DESIGN CORRECTION (2026-07-21): the probe reads the recurrent state `h_t`, not the
observation, so masking obs channels CANNOT be done post-hoc on the saved `h_t` stacks.
A faithful localization requires RE-RUNNING rollouts with the masked observation fed to
the frozen agent (no training, but new forward passes that change the trajectory). That
is a legitimate but more invasive ablation than A1's readout-only sweep, and it answers a
subtly different question ("does the agent still form the world-signal when it cannot see
velocity"). It is therefore deferred to a second pass and not part of this launch, which
runs A1 alone. Prediction to test later: masking the 28 motion channels drives the signal
toward chance more than masking the matched 28 static channels; not expected to reach the
L0 floor because the seam also propagates through position into later raycasts.

### A3 - Alternate-substrate-primitive positive control (DEFERRED, needs a train)

Introduce the divergence through a *different* substrate primitive - the L1
discretization rung (`world.py:346-361`, quantize obs to a 1/64 grid) - and confirm a
survival agent still incidentally encodes it, showing "substrate artifacts" generically
rather than this one velocity law. This requires TRAINING an L1-surrogate organism (no
saved L1 agents exist), so it is a separate multi-hour run and is DEFERRED out of the
first H2 pass. Recorded here so it is on the table, not smuggled in later; A1+A2 on the
existing bundle are the tight first pass.

## 5. Estimand and decision rule

- **Estimand:** the pooled per-episode world-identity target AUROC (the frozen L3
  headline readout - episode_features = [mean h, final h], GroupKFold), evaluated per
  alpha / per mask, averaged over the 10 saved seeds.
- **Reuse the frozen bar (0.65) and CI convention (t-based 90% on per-seed AUROCs).**
- **A1 pass (H2 necessity confirmed):** alpha=1 reproduces the headline within
  determinism tolerance; the target is monotone non-increasing across the alpha grid
  (report Spearman rho with a pre-set threshold); and alpha=0 is equivalent to the L0
  floor by TOST/ROPE (same equivalence machinery as the L0 control).
- **A2 pass (localization):** motion-mask target < 0.65 (drops below the bar) AND
  static-mask target >= 0.65 (holds), with the two masks matched on channel count.
- **Failure:** alpha=0 stays above the floor (non-equivalent), or the sweep is
  non-monotone -> H2 not confirmed; record as a substantive negative and reopen the
  unmeasured-channel question.

## 6. Gates (all must pass before A1/A2 are interpreted)

0. **Determinism gate.** alpha=1 (full surrogate) reproduces the published hidden=8
   pooled target 0.752 bit-identically on the saved agents (the fifth independent
   determinism check on this bundle).
1. **L0 anchor.** alpha=0 reproduces the authentic-vs-authentic floor (~0.517),
   independently confirming the graded hook collapses to authentic at alpha=0.
2. **Probe-honesty.** shuffled-label control at chance for every alpha/mask (the signal
   is real, not probe overfitting).

## 7. Interpretive caveats

- **This closes H2 for the L3 rung at hidden=8, conditional on that artifact.** It does
  not speak to L1/L2 (different substrate primitives) or to H3 (already a validated
  null). The hidden=7 second capacity is leakier and is reported as a secondary check
  only if A1 is run there.
- **Necessity, not exhaustive sufficiency.** A monotone collapse to the floor shows the
  seam is necessary and dose-controls the signal. A2 localizes it to substrate-derived
  channels. Together they are the "loads entirely on the structural artifact" evidence
  the minimal-result bar asks for; they do not prove no other substrate channel could
  ever carry signal in a different design.
- **Reactive reading is unchanged.** The common-garden result (signal is reactive
  tracking of currently-felt dynamics, not a persistent stored representation) stands;
  H2 is about WHAT the signal loads on, not about persistence. A1 neutralizes the felt
  dynamics divergence, so a collapse is fully consistent with the reactive reading.

## 8. What this run costs

Readout-only on the saved 60 agents; NO training anywhere. A1 is 5 pooled-readout passes
(one per alpha) over the 10 saved seeds - each pass is frozen deterministic rollouts to
build the 800_000/850_000 pools plus a logistic probe, i.e. the same unit of work the
held-out (2026-07-14) and cross-recipe (2026-07-15) probes did, x5. A2 adds no new
rollouts at all: it re-fits the probe on already-collected states under two channel
masks, so it is near-free. The dominant cost is the 5x pool regeneration; on CPU this is
expected to be well under a training run (the held-out probe was a single such pass).
RAM preflight per the standing rule before any launch; run background + monitor.

## 9. Decisions - RESOLVED at freeze (2026-07-21)

1. **alpha grid: {0, 0.1, 0.25, 0.5, 0.75, 1.0}** - extra point near 0 to resolve the
   collapse shape; the extra cell is nearly free on a readout-only run.
2. **A3 deferred** - A1 (+ the later A2) close H2 for the L3 rung; A3 needs a fresh train.
3. **hidden=8 only** for the first pass; hidden=7 (leakier) added later as robustness if
   A1 lands clean.
4. **Kept as a `docs/specs/` design**, not promoted to `PREREGISTRATION_H2.md`: this is a
   readout-only evaluation channel on the existing L3 rung, matching the held-out,
   common-garden, and cross-recipe probe specs (all in `docs/specs/`).

Additionally at launch: **A2 deferred** to a second pass (needs rollout-level obs masking,
not a post-hoc h_t mask - see section 4 A2 design correction). This launch = A1 only.
