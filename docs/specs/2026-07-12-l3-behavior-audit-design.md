# L3 behavior-mediation audit: reproducible script, per-timestep control, hidden=4 prep

Date: 2026-07-12
Status: approved (scope confirmed in session; owner launches all GPU runs)

## Problem

The L3 n=10 positive (survival pooled AUROC 0.752) carries a post-hoc caveat:
behavior alone (per-episode mean speed/energy/food/drag) decodes the world at
about 0.69, and controlling for those means in-fold leaves a
behavior-independent signal of about 0.66 (0.676 linear, 0.659 quadratic).
Three problems:

1. The audit that produced 0.69/0.66 was ad hoc and never committed. The
   published numbers (README, PREREGISTRATION_L3.md section 12, commit
   ba68fef) have no code behind them.
2. The control is a soft upper bound. Only 4 per-episode means were
   controlled; per-timestep behavior could push 0.66 lower, possibly to
   chance. This is the test most likely to kill the claim, so it must be run,
   and at the headline capacity (hidden=8), not only at hidden=4.
3. The existing dumps (fullruns/l3_n10_audited/states) contain per-timestep
   hidden states (110, 24, 96) but only per-episode behavior scalars, and no
   agent checkpoints. The per-timestep control therefore requires a re-run
   with an extended dump.

The pre-registration (PREREGISTRATION_L3.md, "STILL OWED") lists: (a) richer
per-timestep behavior control, (b) hidden=4 second-capacity replication,
(c) held-out/common-garden probe. This step covers (a) fully, preps (b), and
defers (c) until (a)'s outcome is known.

## Deliverables

1. `itasorl/behavior_audit.py`: pure, unit-testable analysis functions.
2. `scripts/audit_behavior_mediation.py`: CLI that runs the audit over a
   dump directory and prints per-cell and per-seed-mean tables (optional
   JSON output).
3. Dump extension in `itasorl/experiment_b2.py`: persist per-timestep
   behavior traces `bta`/`bts` with shape (k, steps, 4), channels
   (speed, energy, food_dist, drag). Backward compatible: the audit detects
   trace presence and skips the per-timestep control on old dumps.
4. Run recipe for the owner to launch: hidden=8 re-run (per-timestep control
   at the headline capacity) and hidden=4 n=10 (pre-registered second
   capacity), both with `--dump-states`.
5. Audit executed on `fullruns/l3_n10_audited/states`; recorded numbers
   compared against the published 0.689/0.704 (behavior-only) and
   0.676/0.659 (controlled). If the committed script disagrees, the script
   is canonical and the docs are corrected in the same PR.

## Methodology

All probes reuse the existing machinery verbatim: `episode_features`
([mean_t h, final h]), `grouped_auroc` (GroupKFold 5-fold, StandardScaler +
LogisticRegression fit on train folds only), groups = episode index.

### Per-episode control (reproduces the ad hoc audit)

Per cell (drift, seed, agent): B = per-episode behavior means (2k, 4),
y = world label, X = episode_features(H).

- Behavior-only decode: grouped CV AUROC of B -> y, linear (logistic) and
  nonlinear (random forest). Targets the published 0.689/0.704.
- Controlled probe: inside each CV fold, fit a linear regression B_train ->
  X_train, residualize BOTH train and test features with that train-fit
  model, then fit the standard probe on residuals. Out-of-fold AUROC.
  Quadratic variant: augment B with squares and pairwise interactions
  (14 columns). Targets the published 0.676/0.659.
- In-sample residualization (fit on all data) is known to over-remove
  (deflated to 0.56-0.63 in the ad hoc audit) and is NOT used.

### Per-timestep control (new, requires extended dumps)

Given traces Bt (k, T, 4) aligned with H (k, T, Hdim): per fold, fit a
linear map phi(b)_t -> h_t pooled over train-episode timesteps, residualize
every h_t (train and test) with the train-fit map, rebuild
episode_features from the residual states, probe as usual.

phi(b)_t = [b_t, b_{t-1} (edge-padded), cummean_{<=t}(b)]: 12 columns.
Including contemporaneous b_t is deliberately conservative (removes slightly
more than causally available to h_t); a surviving signal is then harder to
dismiss. Quadratic variant applies degree-2 expansion to b_t only.

Also reported with traces: behavior-trace-only decode (episode_features_full
applied to Bt -> y), the richer behavior ceiling.

### Aggregation

Per-seed cell values, then across-seed mean with `mean_ci` (90% bootstrap),
plus count of seeds clearing the 0.65 bar, mirroring the headline reporting.

## Dump format change

`collect_pool` already computes sp/en/fd/dg per timestep and keeps only
means. Change: also return the stacked trace (k, steps, 4) when
return_anchors=True; `pooled_readout` persists them as `bta` (authentic)
and `bts` (surrogate). Alignment note: H[e, t] is the state that produced
action t; Bt[e, t] is measured just after that action resolves. Old dumps
lack `bta`/`bts` and remain fully readable.

## Testing (all synthetic, CPU, fast)

1. Behavior-mediated-only world signal: uncontrolled probe high, per-episode
   controlled probe falls to about 0.5.
2. Orthogonal genuine world direction plus behavior mediation: controlled
   probe stays high (control does not over-remove).
3. Pure noise: controlled probe about 0.5 (no leakage-driven inflation).
4. Timing-pattern signal (world flips behavior ordering within episodes but
   preserves episode means): per-episode control leaves AUROC high, the
   per-timestep control drives it to about 0.5. This proves the new control
   is strictly stronger.
5. Dump round-trip: pooled_readout writes bta/bts with the right shapes;
   audit loads both old-format (no traces) and new-format dumps.

## Run recipe (owner launches)

    # hidden=8 re-run with traces (per-timestep control on the headline)
    python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 \
        --dump-states fullruns/l3_h8_traces/states  [n=10 profile]
    # hidden=4 second in-band capacity (prereg section 11)
    python scripts/run_expB2.py --drift-mode l3 --l3-hidden 4 \
        --dump-states fullruns/l3_h4_traces/states  [n=10 profile]

Exact flags to be confirmed against run_expB2.py/run_expB2_n10.sh during
implementation and recorded in scripts/README.md.

## Out of scope

- Common-garden / held-out fingerprint probe (owed item c): deferred; its
  design depends on whether the behavior-independent signal survives the
  per-timestep control.
- Any change to training, the surrogate G, or the pre-registered headline
  probe itself.

## Decision rule (stated in advance)

On the new hidden=8 dumps: if the per-timestep-controlled survival AUROC
(across-seed mean) stays at or above 0.65, the behavior-independent claim
strengthens; if it lands in [0.60, 0.65), the claim weakens to "a weak
behavior-independent trace, below the pre-registered bar"; if it lands below
0.60, the honest headline becomes "L3 world-identity signal is largely
behavior-mediated". In every case README, FINDINGS, and PREREGISTRATION_L3
are updated to say which outcome occurred.
