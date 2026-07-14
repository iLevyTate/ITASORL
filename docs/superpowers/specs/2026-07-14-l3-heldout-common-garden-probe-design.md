# L3 held-out fingerprint + common-garden probe design

Date: 2026-07-14
Status: approved (design review in session; implements the last owed L3 item)

## Purpose

The L3 result now stands at: a reward-clean, survivorship-clean,
behavior-independent world-signal of ~0.72 in the survival agent's state at
both frozen capacities (hidden=8 resid_trace 0.726; hidden=7 replication
0.722), with the survival-vs-predictor dissociation holding only at the
subtler hidden=8 artifact (PREREGISTRATION_L3.md sec.12, 2026-07-14 entry).

Two skeptic escape hatches remain, and the owed "held-out/common-garden
probe" (FINDINGS.md sec.7.2; prereg sec.12 STILL OWED) closes both:

1. "The probe just reads THIS particular artifact." Answered by a transfer
   test: does the world-identity direction learned against the trained
   fingerprint generalize to a fingerprint the agent never saw?
2. "The state just reflects what the dynamics feel like right now"
   (reactive vs representational, FINDINGS.md sec.7.2 item 2). Answered by
   a common-garden test: after the felt dynamics are made identical, does
   the state still carry where the episode came from?

Both are EVALUATION channels on top of the frozen training protocol; no
change to training, the surrogate G family, or the pre-registered headline
probe. Because trained agents are never persisted (no torch.save anywhere
in the repo), the channels run inside one new training run.

## Run profile (frozen)

- One n=10 run: seeds 0-9, drifts [0.0, 0.45], 300 updates, the standard
  frozen organism config (prereg sec.9), RTX 4050, `--device cuda` passed
  EXPLICITLY so a CUDA failure aborts rather than silently training on CPU.
- Agents train against the hidden=8 surrogate (headline capacity, cleanest
  gates: untrained floor 0.482). hidden=7 (the other validated in-band
  capacity, oracle 0.922, floor 0.566) is the held-out fingerprint.
- All existing gates computed in-run, unchanged: L0 control, engagement,
  speed positive control, pooled reward-leak, untrained mechanical floor.
- The standard headline pools are unchanged, so the run doubles as a third
  determinism check on the published survival 0.752 at hidden=8.
- Out dir: `fullruns/l3_h8_heldout` with `--dump-states` and
  `--save-agents`. Cell-level checkpoints/resume as usual.

## Channel 1: unseen-fingerprint transfer

Per cell (drift, seed) and per arm (untrained / predictor / survival),
after the standard pools:

- Collect one extra surrogate pool (n=110, steps=24) with the hidden=7
  surrogate active, and one FRESH authentic pool (same size, distinct seed
  base). Fresh, because the transfer test must not score authentic episodes
  the probe already trained on.
- Fit the pooled linear probe on the standard auth-vs-h8-surr episode
  features (fit once on the full standard pools; the test set is disjoint),
  then evaluate it FROZEN on fresh-auth-vs-h7-surr.
- New metric per arm: `transfer_target` AUROC (+ bootstrap CI, as the
  existing pooled metrics report).

Decision rule (stated in advance): survival `transfer_target` >= 0.65 AND
> untrained `transfer_target` + 0.05 -> the world-signal generalizes beyond
the trained fingerprint. The untrained arm is the MECHANICAL transfer
floor, not 0.5: hidden=7 is mechanically leakier (organism-run floor 0.586
when probed directly), so transfer claims are only ever read against it.
Survival below untrained + 0.05 -> the h8-learned direction is
fingerprint-specific (informative negative, recorded as such).

## Channel 2: common garden

Per cell and per arm:

- 220 episodes, 110 per prefix world (matching the standard pool's
  per-world power). Prefix: 20 steps in either the authentic world or the
  hidden=8 surrogate world. At the switch point BOTH groups
  continue under identical authentic dynamics for a 24-step tail. Same
  world params; only the prefix dynamics differ, and they end at the
  switch.
- Probe reads TAIL-ONLY states (episode features [mean h, final h] over the
  tail window), labels = prefix world, GroupKFold as in the standard
  readout. New metric per arm: `cg_tail_target`.
- Decay check: the same probe computed on the LAST 8 tail steps only
  (`cg_latetail_target`). A reactive signal should wash out along the tail;
  a persistent representation should not.

Decision rule (stated in advance): survival `cg_tail_target` >= 0.65 AND
> untrained `cg_tail_target` + 0.05 -> persistent world identity
(representational). Below either bar -> the signal is reactive
moment-to-moment tracking; that is an informative negative for the
representational reading, not a gate failure.

Honesty note baked into the design: ANY recurrent state mechanically
integrates prefix history, so the untrained arm's tail score may sit well
above 0.5. The claim is only ever survival vs that floor.

## CLI and code touch-points

Flags on `scripts/run_expB2.py`, all strict no-ops unless set:

- `--heldout-evals` (requires `--drift-mode l3`): enables both channels.
- `--heldout-hidden 7`, `--cg-prefix 20`, `--cg-steps 24`: frozen defaults.
- `--save-agents`: torch.save each arm's state_dict + frozen obs-norm to
  `<out-dir>/agents/agent_d{drift:.2f}_s{seed}_{arm}.pt` (a few MB each).
  Prevents a repeat of the situation that forced this retraining: trained
  agents were never persisted anywhere.

`itasorl/experiment_b2.py`:

- `transfer_readout(...)` (new): swaps the global L3 surrogate to
  `heldout_hidden` via `setup_l3_surrogate`, collects the transfer pools
  with `collect_pool` (distinct seed bases; no episode reuse), restores the
  training surrogate in a try/finally so the global surrogate state can
  never leak into later evals, then scores the frozen probe.
- `common_garden_rollout(...)` (new): prefix in world X, dynamics switched
  to authentic for the tail; reuses the `_run_branch` pattern from
  `matched_pair_recurrent_rollout`. Returns tail-only state sequences.
- Probe: `probe_world_identity` unchanged for the tail probe; transfer
  scoring is a small frozen-evaluation variant beside it.

Results schema: new per-arm keys in the cell JSON and aggregate results -
`transfer_target`, `cg_tail_target`, `cg_latetail_target` (+ CI fields
matching the existing naming). New keys change the config fingerprint;
correct, since heldout cells must never mix with old checkpoints.

State dumps: standard dumps unchanged (behavior-audit compatible). Transfer
and CG pools dumped alongside with suffixed names
(`states_d*_s*_{arm}_h7transfer.npz`, `states_d*_s*_{arm}_cg.npz`) so
post-hoc audits (e.g. behavior mediation on the transfer pool) can run
later without retraining.

## Testing (all before any launch)

1. Unit tests on synthetic ground truth, mirroring
   `2026-07-12-l3-behavior-audit-design.md`: (a) a fingerprint-general
   synthetic signal must transfer (high `transfer_target`) and a
   fingerprint-specific one must not; (b) a synthetic persistent state must
   score high on the tail probe and a purely reactive one must fall to
   chance in the tail.
2. `--quick` smoke run exercising both channels end-to-end on CUDA.
3. No-op regression: with the new flags off, the cell JSON schema is
   byte-identical to the current schema.
4. `ruff check .` clean.

## Run command (after implementation + tests)

    python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 \
        --heldout-evals --heldout-hidden 7 --save-agents \
        --seeds 0 1 2 3 4 5 6 7 8 9 --device cuda \
        --out-dir fullruns/l3_h8_heldout \
        --dump-states fullruns/l3_h8_heldout/states

Launch policy per standing preference: RAM preflight (>= 4 GB free),
background with log tee, monitor on cell checkpoints; results recorded in
PREREGISTRATION_L3.md sec.12 with the decision rules above applied as
frozen.

## Out of scope

- Any change to training, the surrogate G, or the pre-registered headline
  probe.
- The train-at-h7 / hold-out-h8 direction (stricter transfer test): staged
  as a possible follow-up ONLY if the h8-trained transfer result is
  positive; not part of this run.
- Common-garden across capacities (h7 prefix vs h8 prefix): YAGNI until the
  basic channel reads out.
