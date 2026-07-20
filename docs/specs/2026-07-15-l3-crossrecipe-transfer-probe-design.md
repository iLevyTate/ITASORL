# L3 cross-recipe transfer probe design

Date: 2026-07-15
Status: approved (design review in session; follows the held-out/common-garden result)

## Purpose

The L3 arc closed with a split result (PREREGISTRATION_L3.md sec.12, 2026-07-14
entry): the world-identity direction fit against the trained hidden=8 fingerprint
transfers to an unseen same-recipe fingerprint (survival transfer_target 0.773 vs
untrained floor 0.569; frozen rule passes), and the common-garden control resolved
the signal as reactive tracking of the felt dynamics, not a persistent stored
representation.

One skeptic escape hatch on the transfer channel remains, named by the result
itself: "hidden=7 is the SAME surrogate recipe at a different capacity, not a
different surrogate family; cross-recipe transfer is out of scope for this run."
This probe closes it: does the direction fit against the trained MLP fingerprint
still read a surrogate family with a different function class and a different fit
procedure? A positive means the probe reads systematic dynamics error in general;
a negative means the published transfer claim is recipe-scoped, and the docs say
so.

Both channels are readout-only. No training, no change to the surrogate G used
for training, no change to the pre-registered headline probe.

## World-P scope note (stated up front)

In the frozen organism world P (k_land = k_water = 1.5, gravity = 0.4;
run_expA_l3.py:50), drag is constant, so the authentic velocity law is EXACTLY
linear in (vel, a) (patch_of_earth.py:168-177). Verified consequences:

- Any family that can represent a linear map exactly (linear or polynomial least
  squares) fits the authentic law to machine precision: no fingerprint, gate-dead
  from below. The linear family is INVALID here, not merely weak.
- Every valid fingerprint in world P is systematic approximation texture of an
  exactly linear law. "Cross-recipe" therefore means a different approximator
  class and fit procedure, not a different information constraint. This scope is
  stated in every writeup of the result.
- A globally mis-set constant drag is degenerate L2-regime
  (patch_of_earth.py:154-172: regime = per-episode constant drag scaling). This
  is owned and used deliberately: see Family 2.

## Held-out families

Family 1, PRIMARY: `G_rff`, a random-Fourier-features ridge regression velocity
law.

- Features z(x) = sqrt(2/D) * cos(W x_norm + b), rows of W drawn N(0, I/ell^2),
  b drawn U[0, 2pi), frozen feature seed 0. Closed-form ridge solve for the
  output weights. Per-step inference is numpy-only (two matmuls plus cos),
  matching the GMotion no-torch constraint.
- Same training data as G_motion: authentic transitions, 250 eps x 40 steps,
  seed 0, world P, same normalization treatment.
- Frozen knob defaults: ridge lambda = 1e-3, bandwidth ell = 1.0 on normalized
  inputs. Difficulty knob: D (feature count), swept ascending {8, 16, 32, 64,
  128}; oracle expected monotone decreasing in D; freeze the FIRST in-band D.
  Fallback mirrors prereg sec.9: if the sweep steps over the band, bisect on D;
  if bisection cannot land, raise lambda one decade and re-sweep (recorded).
- Why this is a different recipe: smooth global sinusoidal basis vs piecewise
  linear ReLU units; convex closed-form fit vs Adam; deterministic given seed
  with no optimizer path dependence. No rung collision: the residual is smooth
  (no L1 comb structure) and a texture field (not an L2 coefficient bias).

Family 2, SECONDARY: `G_cd`, a mis-set constant-drag analytic law.

- vel_next = (1 - c*dt) * vel + a*dt with c = 1.5 * (1 + eps). At eps = 0 this
  IS the authentic world-P law, so only a mis-set constant has a fingerprint: a
  pure velocity-decay-rate bias.
- Difficulty knob: eps, oracle monotone increasing in |eps|. Calibrate on the
  coarse grid {0.05, 0.1, 0.2, 0.4, 0.8}, then bisect; freeze the first in-band
  value.
- Honest framing, pre-registered: this family is degenerate L2-regime, and that
  is its value. The L2 arc showed a drag-coefficient artifact is NOT incidentally
  encoded when trained on; this channel asks whether the L3-fit direction READS
  such an artifact anyway. Cross-rung, secondary/exploratory, not part of the
  primary decision.

## Gate 0 per family (before any transfer scoring)

Run the run_expA_l3.py battery per family: oracle AUROC in [0.85, 0.95] with
sigma_meas = 0.02; mechanical leakage battery pass; untrained pooled floor
|target - 0.5| < 0.10 at drift 0.45. Floors are per-family and become that
family's mechanical transfer floor (transfer claims are only ever read against
the same family's untrained floor, never against 0.5).

Pre-stated drop rules: a family that cannot calibrate into the band is dropped
and recorded in prereg sec.12 (no transfer claim, no penalty to the other
family). If both families drop, nothing launches and the failure is recorded.

## Reuse protocol (readout-only, no retraining)

- Agents: the 60 saved checkpoints in `fullruns/l3_h8_heldout/agents/`
  (state_dict + frozen obs-norm; 2 drifts x 10 seeds x 3 arms). A loader is new
  code; saving already exists.
- Integrity gate, must pass before any new pool counts: with `--device cuda`
  (matching the original run) and the h8 training surrogate rebuilt bit-identically
  (train_g_motion seed 0, params P, same device), regenerate the standard
  authentic and h8-surrogate pools for EVERY reloaded agent, using the original
  run's pool seed bases, and require bit-identical state arrays against the
  saved dumps in `fullruns/l3_h8_heldout/states/`. Refitting the probe on those
  pools must reproduce the published drift-0.45 pooled survival target 0.752
  exactly. This doubles as the fourth independent determinism check.
- Probe: refit the pooled linear probe per cell (drift, seed) per arm on the
  SAVED standard pools (auth vs h8 features), freeze, then score on new pools.
  Identical fit code path as the heldout run's transfer channel.
- New pools, per drift-0.45 cell, per arm, per family: one FRESH authentic pool
  vs one held-out-family pool (n = 110, steps = 24), seed bases distinct from
  every base used in the original run, so no episode the probe trained on is
  ever scored. Transfer is evaluated at drift 0.45 (the organism surrogate
  strength), mirroring the heldout transfer channel exactly.

## Decision rules (frozen in advance)

Channel 1, PRIMARY (`G_rff`): survival `transfer_rff_target` >= 0.65 AND
> untrained `transfer_rff_target` + 0.05 -> the world-signal generalizes across
surrogate recipes. Below either bar -> the h8-learned direction is
recipe-specific: an informative negative; FINDINGS 10.7 and the sec.6 H4 wording
get the caveat.

Channel 2, SECONDARY (`G_cd`): the same metrics are computed and reported with
CIs, with no headline pass/fail. Pre-stated interpretation table:

| G_rff (primary) | G_cd (secondary) | Reading |
|---|---|---|
| passes | passes | Strongest: the direction reads systematic dynamics error in general, across learned-nonlinear, fitted-smooth, and analytic-bias signatures |
| passes | fails | Primary claim stands; boundary mapped: reads fitted-approximation textures but not a pure coefficient bias |
| fails | passes | Recipe-specific for textures yet reads coefficient bias: points to a low-dimensional felt-dynamics direction; headline caveat updated |
| fails | fails | The h8 direction is fingerprint-recipe-specific; the transfer claim stays scoped to same-recipe capacities (current published scope) |

## CLI and code touch-points

- `itasorl/surrogate_l3_families.py` (new): `GRff` and `GConstantDrag`, both
  callable as `(vel, a, drag) -> vel_next` ignoring drag, numpy-only per step;
  builders `fit_g_rff(...)` and `make_g_cd(...)`.
- `scripts/run_expA_l3.py`: new `--family {mlp, rff, cd}` flag. Default `mlp`
  preserves current behavior byte-for-byte (no-op regression). `rff` sweeps D,
  `cd` sweeps eps; JSON calibration table as today.
- `scripts/run_l3_crossrecipe.py` (new, readout-only): loads agents, runs the
  integrity gate, collects transfer pools for the gate-passing families, scores
  the frozen probe. Flags: `--agents-dir`, `--states-dir`, `--out-dir`,
  `--families`, `--device`, `--quick`.
- `scripts/run_expB2.py` training path: UNTOUCHED.
- Results: `fullruns/l3_crossrecipe` with per-cell JSON keys
  `transfer_rff_target`, `transfer_cd_target` (+ CI fields matching existing
  naming). State dumps suffixed `_rfftransfer.npz`, `_cdtransfer.npz` so
  post-hoc audits (e.g. behavior mediation) can run without recollection.

## Testing (all before any launch)

1. Synthetic ground truth, mirroring the heldout tests: a direction fit on one
   texture must transfer to a shared-component construction and must NOT
   transfer to an orthogonal texture.
2. Family unit tests: `GRff` deterministic across processes for a fixed seed;
   `GConstantDrag` with eps = 0 equals the authentic law to float precision.
3. Loader integrity test on one saved agent: bit-identical regenerated states on
   a small pool.
4. Gate-0 calibration determinism: identical JSON on two runs with fixed seeds.
5. No-op regression: `run_expA_l3.py` without `--family` matches the current
   output schema exactly.
6. `--quick` smoke of `run_l3_crossrecipe.py` end-to-end on CUDA.
7. `ruff check .` clean (CI runs it on Py 3.10-3.12).

## Run commands (after implementation + tests)

    python scripts/run_expA_l3.py --family rff --json fullruns/l3_crossrecipe/gate0_rff.json
    python scripts/run_expA_l3.py --family cd  --json fullruns/l3_crossrecipe/gate0_cd.json
    python scripts/run_l3_crossrecipe.py \
        --agents-dir fullruns/l3_h8_heldout/agents \
        --states-dir fullruns/l3_h8_heldout/states \
        --families rff cd --device cuda \
        --out-dir fullruns/l3_crossrecipe

Launch policy per standing preference: RAM preflight, background with log tee,
monitor on checkpoints; results recorded in PREREGISTRATION_L3.md sec.12 (new
dated entry) and FINDINGS.md sec.10.7 with the frozen rules above applied.

## Out of scope

- Any change to training, the trained surrogate G, or the pre-registered
  headline probe.
- New training runs (readout-only by construction).
- k-NN / grid-quantization family: rejected; its fingerprint recapitulates the
  L1 quantization-residual detection statistic.
- Linear / polynomial least-squares family: invalid in world P (exact fit, no
  fingerprint; see the scope note).
- Cross-WORLD transfer (different WorldParams): a different axis, not this
  probe.
- Common-garden channels for the new families: YAGNI until the transfer channel
  reads out.
- L4 (adversarially hardened surrogate): separate rung, separate
  preregistration.
