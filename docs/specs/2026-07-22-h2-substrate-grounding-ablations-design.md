# H2 substrate-grounding ablations design

Date: 2026-07-22
Status: frozen (approved for implementation 2026-07-23; no science run until
tests pass and gate-0 calibrations are recorded)

## Purpose

H2 (substrate-grounding) is the last untested hypothesis: "detection is driven
by computational artifacts, not task confounds - established through strict
ablations" (`docs/ITASORL.md`). The L3 arc's published controls each killed one
named confound (reward leak clean 10/10, survivorship 0/220 deaths,
per-timestep behavior control leaves 0.726, nonlinear untrained floor at
chance), but none tested whether the world-identity direction loads on the
artifact's STRUCTURE. The surviving skeptic story is:

> "The survival state separates the worlds because the surrogate is
> differently perturbing to live in. Any equally-detectable dynamics
> perturbation - structured or not - would separate them just as well."

If that is true, the published "generative fingerprint is incidentally
encoded" wording overclaims: the direction reads generic
dynamics-perturbedness, a task-level confound, not the substrate texture.

This probe adjudicates H2 with two readout-only ablations against the saved
`fullruns/l3_h8_heldout` agents (no training anywhere):

1. **Structure knockout (PRIMARY).** Score the frozen world-identity direction
   on a gate-calibrated surrogate whose tell is *unstructured* - iid Gaussian
   velocity jitter tuned into the same oracle band [0.85, 0.95] as the trained
   fingerprint (0.928) and the cross-recipe RFF family (0.887). Equal
   detectability, zero learned texture. The published cross-recipe result
   (rff 0.684, rule passes) is the matched-band STRUCTURED comparator; it is
   cited, not rerun.
2. **Dose-response titration (SECONDARY).** Score the same frozen direction on
   the same-recipe artifact at higher `G` capacities (hidden 16, 32, 64), which
   shrink the fingerprint below the band (prior sweeps: hidden>=16 oracle
   <0.75, hidden=32 ~0.70, hidden=128 ~0.63; measured fresh here). If the
   signal is substrate-grounded, its reading should decay with the artifact's
   oracle detectability.

Both channels reuse the cross-recipe machinery (`transfer_readout`, gate-0
battery, integrity gate) unchanged in form.

## The ablation logic, stated once

The frozen direction already reads, at matched in-band detectability:

- the trained MLP fingerprint (standard probe 0.752),
- an unseen same-recipe capacity variant (transfer 0.773),
- an unseen different-recipe fitted texture (rff transfer 0.684).

All three are SYSTEMATIC approximation textures of the authentic law. The
knockout asks: remove the systematicity, keep the detectability. A direction
grounded in the substrate artifact reads textures, not magnitude; a direction
reading a task confound (perturbedness, altered movement statistics) reads
anything in the band. The capacity ladder then asks the complementary
question: keep the texture's RECIPE, shrink its magnitude - does the reading
co-decay with the oracle?

## Held-out ablation family: `G_gn` (Gaussian-noise velocity law)

- `vel_next = (1 - drag0*dt)*vel + a*dt + eta`, `eta ~ N(0, sigma_v^2 I2)`,
  where `drag0*dt` uses world-P's exact authentic constants - the DETERMINISTIC
  part of the law is authentic to machine precision, so the ONLY tell is the
  iid jitter. Satisfies the `GMotion` hook contract (`(vel, a, drag) ->
  vel_next`, drag ignored, numpy-only per step).
- At `sigma_v = 0` the law equals the authentic world-P law exactly (unit
  test, mirroring `GConstantDrag` eps=0).
- **No learned structure by construction:** eta is white in time, isotropic in
  velocity space, and independent of `(vel, a)`. There is nothing for a
  texture-reading direction to read except the felt consequences of being
  buffeted - which is exactly the task-confound channel the knockout isolates.
- **Determinism:** `G_gn` holds its own `np.random.Generator(Philox(seed))`;
  pool collection is sequential and single-threaded, so a fixed collection
  order gives bit-reproducible pools. A double-run bit-identity test is
  required (same bar as the cd calibration double-runs).
- Difficulty knob: `sigma_v`, oracle monotone increasing. Calibrate on the
  coarse grid {0.0025, 0.005, 0.01, 0.02, 0.04}, then bisect; freeze the FIRST
  in-band value. (The residual oracle reads the per-step velocity residual
  against the exact authentic law at sigma_meas = 0.02, so the tell IS eta;
  the grid brackets sigma_meas deliberately.)
- **Rung honesty:** iid dynamics noise is not on the L1-L4 ladder; it is a
  deliberate off-ladder ablation. It does not collide with L1 (no grid
  structure), L2 (no coefficient bias; the deterministic part is exact), or L3
  (no learned texture).

## Gate 0 per channel (before any transfer scoring)

Run the `run_expA_l3.py` battery (new `--family gn` and the mlp ladder), on
world P at the frozen sigma_meas = 0.02:

- **`G_gn`:** oracle AUROC in [0.85, 0.95]; mechanical leakage battery pass;
  untrained pooled floor |target - 0.5| < 0.10 at drift 0.45. Transfer claims
  are only read against `G_gn`'s own untrained floor, never against 0.5.
- **Capacity ladder (hidden 16, 32, 64):** NO band requirement - sub-band
  detectability is the point of the titration. Required per capacity:
  mechanical leakage pass AND untrained floor |target - 0.5| < 0.10 at drift
  0.45 (the post-fix floor sweep covered 8-32 at chance, but 64 has never been
  floor-checked on world P; all three are measured fresh). Record each
  capacity's oracle AUROC - these are the x-axis of the dose-response. The
  battery's composite `passes_gate0` field (which requires the band) is
  IGNORED for ladder capacities; only `mech_leak_pass` and `floor_ok` gate
  them, and the H2 runner reads exactly those two fields from the ladder JSON.

Pre-stated drop rules: a ladder capacity failing its floor/leakage check is
dropped and recorded (the titration reads on the surviving capacities). If
`G_gn` has an empty calibration window (floor exceeds tolerance at every
in-band sigma_v, the `G_cd` failure mode), the knockout is dropped and
recorded - itself informative (unstructured jitter cannot be made
subtle-but-detectable in world P either) - and the PROMOTION rule below makes
the titration the primary H2 channel.

## Reuse protocol (readout-only, no retraining)

Identical in form to the cross-recipe probe (spec 2026-07-15):

- Agents: the 60 saved checkpoints in `fullruns/l3_h8_heldout/agents/`.
- Integrity gate, must pass before any new pool counts: regenerate the
  standard pools for EVERY reloaded agent with the original seed bases
  (800_000/850_000) against the bit-identical rebuilt hidden=8 GMotion
  (seed 0, params P, `--device cuda`); require bit-identical state arrays
  against `fullruns/l3_h8_heldout/states/` and the drift-0.45 pooled survival
  mean to reproduce the published 0.752 exactly (fifth independent determinism
  check).
- Probe: refit the pooled linear probe per cell on the SAVED standard pools,
  freeze, score on new pools (`transfer_readout`, identical code path).
- New pools per drift-0.45 cell per arm per channel: one FRESH authentic pool
  vs one ablation pool (n = 110, steps = 24). Frozen seed bases, distinct from
  every base used in any prior run: gn 960_000/970_000; ladder h16
  980_000/990_000, h32 1_000_000/1_010_000, h64 1_020_000/1_030_000.
- Per-pool survivor/death counts reported (the jitter could re-introduce
  deaths; `transfer_readout` already reports them).

## Decision rules (frozen in advance)

**Channel 1, PRIMARY (`G_gn` structure knockout).** Evaluate the standard
positive rule on per-arm means at drift 0.45: survival `transfer_gn_target`
>= 0.65 AND > untrained `transfer_gn_target` + 0.05.

| Positive rule on `G_gn` | Reading |
|---|---|
| Fails BOTH clauses (survival < 0.65 AND within +0.05 of the untrained floor) | **H2 SUPPORTED (texture-specific).** The direction reads matched-band structured textures (0.752 / 0.773 / 0.684) but not matched-band unstructured noise; detection loads on the computational artifact, not perturbedness. FINDINGS sec.6 H2 flips to supported-at-L3. |
| Fails the absolute bar but clears the floor by > 0.05 | **PARTIAL.** A sub-bar unstructured reading exists; H2 supported in weakened form (texture-preferring, not texture-exclusive); wording caveated. |
| PASSES both clauses | **H2 NEGATIVE (perturbation-generic).** The direction reads any equally-detectable dynamics perturbation; the "fingerprint is encoded" wording is downgraded to "dynamics-perturbation is encoded" in FINDINGS sec.6, sec.10, and README. |
| Clears the bar but sits within +0.05 of an untrained floor >= 0.60 | **UNINFORMATIVE.** The unstructured world is mechanically separable at readout despite passing the (3-seed) gate-0 floor check - the cd-style failure surfacing late. Recorded; no knockout claim either way; the promotion rule applies. |

**Channel 2, SECONDARY (capacity ladder).** Report per-capacity survival /
predictor / untrained transfer means with t-based 90% CIs alongside each
capacity's fresh oracle AUROC. Exploratory reading, no binary rule (house
multiplicity discipline): monotone co-decay of the survival reading with
oracle detectability supports grounding; a flat survival reading across a
falling oracle supports the confound story. Stated expectation: in-band
same-recipe transfer read 0.773; the hidden=64 reading should sit well below
it if the signal is dose-responsive.

**Promotion rule (only if `G_gn` drops at gate 0 or lands UNINFORMATIVE).**
The ladder becomes primary with this frozen rule: survival
`transfer_h64_target` < 0.65 AND `transfer_h16_target` >
`transfer_h64_target` + 0.05 -> dose-response demonstrated, H2 supported in
the graded form; otherwise not demonstrated. If the `G_gn` channel reaches
any of the first three table rows, this rule is never evaluated (no
double-dipping).

## CLI and code touch-points

- `itasorl/surrogate_l3_families.py`: add `GNoise` (Philox generator, seeded
  at construction; `reseed(seed)` method so the runner can reset the stream
  per pool for reproducibility) and `make_g_gn(*, sigma_v, params, seed)`;
  extend `gate0_candidates` with family `"gn"` sweeping `GN_SWEEP =
  (0.0025, 0.005, 0.01, 0.02, 0.04)`.
- `scripts/run_expA_l3.py`: extend `--family` choices with `gn`; the mlp path
  already accepts `--hiddens 16 32 64` for the ladder battery. Default
  behavior byte-identical (no-op regression).
- `scripts/run_l3_h2_ablations.py` (new, readout-only): modeled on
  `run_l3_crossrecipe.py` - integrity gate, then per-channel transfer pools
  and frozen-probe scoring. Flags: `--agents-dir`, `--states-dir`,
  `--out-dir`, `--channels {gn, ladder}`, `--gn-json`, `--ladder-json`,
  `--device`, `--quick`.
- `scripts/run_expB2.py` training path: UNTOUCHED.
- Results: `fullruns/l3_h2_ablations` with per-cell keys
  `transfer_gn_target`, `transfer_h16_target`, `transfer_h32_target`,
  `transfer_h64_target` (+ CI fields matching existing naming); state dumps
  suffixed `_gntransfer.npz`, `_h16transfer.npz`, etc., so the behavior-
  mediation audit can run post-hoc without recollection. Aggregate carries
  the machine-checked `gn_rule_pass` / verdict fields like `rff_rule_pass`.

## Testing (all before any launch)

1. Synthetic ground truth: a direction fit on a structured texture must NOT
   read a matched-magnitude iid-noise construction, and must still read a
   shared-component structured one (tests the scorer's discrimination, not
   the science).
2. `GNoise` unit tests: sigma_v=0 equals the authentic world-P law to float
   precision; fixed seed gives bit-identical streams across processes;
   `reseed` restores the stream.
3. Loader integrity test on one saved agent (bit-identical regenerated states
   on a small pool).
4. Gate-0 calibration determinism: identical JSON on two runs with fixed
   seeds (including the gn family, which exercises the Philox path).
5. No-op regression: `run_expA_l3.py` without `--family` matches current
   output byte-for-byte.
6. `--quick` smoke of `run_l3_h2_ablations.py` end-to-end on CUDA.
7. `ruff check .` clean.

## Run commands (after implementation + tests)

    python scripts/run_expA_l3.py --family gn --json fullruns/l3_h2_ablations/gate0_gn.json
    python scripts/run_expA_l3.py --hiddens 16 32 64 --json fullruns/l3_h2_ablations/gate0_ladder.json
    python scripts/run_l3_h2_ablations.py \
        --agents-dir fullruns/l3_h8_heldout/agents \
        --states-dir fullruns/l3_h8_heldout/states \
        --channels gn ladder --device cuda \
        --gn-json fullruns/l3_h2_ablations/gate0_gn.json \
        --ladder-json fullruns/l3_h2_ablations/gate0_ladder.json \
        --out-dir fullruns/l3_h2_ablations

Launch policy per standing preference: RAM preflight, background with log
tee, monitor on checkpoints. Results recorded as a dated entry in
`docs/PREREGISTRATION_L3.md` sec.12 (the ablations act on the L3 result), a
new FINDINGS section ("Experiment B, H2: substrate-grounding ablations"),
and the FINDINGS sec.6 H2 line, with the frozen rules above applied.

## Out of scope

- Any change to training, the trained surrogate `G`, or the pre-registered
  headline probe.
- New training runs (readout-only by construction). Retrain-based ablations
  (agents trained IN ablated worlds) answer "what induces encoding", already
  partly covered by the hidden=7/hidden=4 arcs, at ~10x compute.
- State-side lesions / observation-dimension attributions on the saved dumps:
  correlational, no new world contrast; a possible post-hoc addendum, not
  part of the H2 adjudication.
- Temporally-correlated (AR(1)) noise family: collides with the L2 drift
  rung's construction; the knockout needs the cleanest structureless case.
- Common-garden channels for the ablation families: YAGNI; persistence was
  adjudicated in 10.6.1 and this probe asks about grounding, not persistence.
- L4 (adversarially hardened surrogate): separate rung, separate
  preregistration.
