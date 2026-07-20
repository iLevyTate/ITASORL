# Scripts

Deterministic reproduction runners. Execute from the **repo root**:

```bash
python scripts/run_e2e.py --quick     # pytest + all experiments (recorded)
python scripts/run_e2e.py --profile bv3_regime   # named preset (see run_e2e.PROFILES)
python scripts/run_expA.py            # Experiment A, L1
python scripts/run_expB2.py           # Experiment B-v2 (GPU if available)
python -m itasorl.experiment_b        # Experiment B smoke test
```

`run_e2e.py --profile` is what the Colab notebook runs: the profile table,
Drive mirroring (`--drive-sync DIR`) and auto-resume of an unfinished run of
the same profile all live in `run_e2e.py`, so the notebook stays a thin shell.

See the root `README.md` for the full list.

## run_local.py

Run any Colab notebook `RUN_PROFILE` locally with resume support.

```bash
python scripts/run_local.py bv3_regime_n10          # first run
python scripts/run_local.py bv3_regime_n10 --resume  # after any interruption
```

Preflights: CUDA must be visible (override with `--allow-cpu`) and at least 4 GB free RAM. For expB2, progress checkpoints are written per (drift, seed) cell under `<run>/artifacts/cells/`.

  If the latest-run pointer is lost or moved, resume a specific run directly:
  `python scripts/run_local.py <profile> --resume fullruns/<dir>` (or
  `python scripts/run_e2e.py --resume fullruns/<dir> --only expb2`).

- `--b2-dump-states auto` places recurrent-state dumps under
  `<run_dir>/artifacts/states` so they are mirrored to Drive, included in
  `bundle.zip`, and survive resume on a different machine. All profiles with
  `dump_states=True` use `auto` (Colab and `run_local.py` alike).

## audit_behavior_mediation.py

Offline (no GPU) behavior-mediation audit of a pooled-states dump: can
behavior alone decode the world, and how much world-signal survives the
in-fold behavior controls? Old dumps get the per-episode-mean control; dumps
written after the trace extension (keys `bta`/`bts`) also get the strictly
stronger per-timestep control.

```bash
python scripts/audit_behavior_mediation.py fullruns/l3_n10_audited/states \
    --json fullruns/l3_n10_audited/behavior_audit.json
```

## L3 owed runs (human-launched, GPU)

Pre-registered follow-ups needing fresh dumps (the audited n=10 run predates
the trace extension: no `bta`/`bts`, no checkpoints):

```bash
# DONE 2026-07-13 (fullruns/l3_h8_traces): hidden=8 re-run, per-timestep
# behavior control at the HEADLINE capacity. Outcome: resid_trace 0.726
# [0.685, 0.765] -> claim STRENGTHENED (artifacts/expB2/behavior_audit_l3_h8_traces.json).
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out-dir fullruns/l3_h8_traces --dump-states fullruns/l3_h8_traces/states

# DONE 2026-07-13 (fullruns/l3_h4_traces): hidden=4 completed but UNINFORMATIVE
# per the decision matrix - gate failures (untrained floor 0.891, reward-leak
# clean 0/10 seeds, engagement 30%). hidden=4 was never re-validated on world P
# after the world-params fix; see the 2026-07-13 gate-0 re-freeze entry in
# docs/PREREGISTRATION_L3.md sec.12.
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 4 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out-dir fullruns/l3_h4_traces --dump-states fullruns/l3_h4_traces/states

# Gate-0 recalibration on world P (validates oracle band AND untrained floor per
# capacity, hidden=8 regression check; selects the second capacity). Result
# 2026-07-13: hidden=7 is the only valid second capacity (oracle 0.922, floor
# 0.566); table in fullruns/l3_gate0_recal/calibration.json.
python scripts/run_expA_l3.py --json fullruns/l3_gate0_recal/calibration.json

# DONE 2026-07-14 (fullruns/l3_h7_traces): hidden=7 second-capacity replication,
# all gates pass. Survival 0.737 [0.688, 0.780] with resid_trace 0.722 (replicates
# hidden=8's 0.726) but predictor reads 0.714, so the survival-vs-predictor
# dissociation does NOT replicate at this coarser artifact; see the 2026-07-14
# entry in docs/PREREGISTRATION_L3.md sec.12.
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 7 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out-dir fullruns/l3_h7_traces --dump-states fullruns/l3_h7_traces/states

# Behavior-mediation audit for the hidden=7 run (artifact committed):
python scripts/audit_behavior_mediation.py fullruns/l3_h7_traces/states \
    --json artifacts/expB2/behavior_audit_l3_h7_traces.json

# DONE 2026-07-14 (fullruns/l3_h8_heldout): held-out fingerprint + common-garden
# run, all gates pass, published h8 table reproduced exactly. Transfer to the
# unseen hidden=7 fingerprint: survival 0.773 [0.728, 0.815] -> GENERALIZES
# (frozen rule met). Common garden: survival 0.557 [0.500, 0.611] < 0.65 ->
# REACTIVE tracking (informative negative). See the second 2026-07-14 entry in
# docs/PREREGISTRATION_L3.md sec.12; spec in
# docs/specs/2026-07-14-l3-heldout-common-garden-probe-design.md.
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 \
    --heldout-evals --heldout-hidden 7 --save-agents \
    --seeds 0 1 2 3 4 5 6 7 8 9 --device cuda \
    --out-dir fullruns/l3_h8_heldout \
    --dump-states fullruns/l3_h8_heldout/states
```

Then run `audit_behavior_mediation.py` on each `states/` directory. Decision
rule (fixed in advance, see
`docs/specs/2026-07-12-l3-behavior-audit-design.md`): survival
`resid_trace` mean >= 0.65 strengthens the behavior-independent claim;
[0.60, 0.65) weakens it to a below-bar trace; < 0.60 means the L3 signal is
largely behavior-mediated.
