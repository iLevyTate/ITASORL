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

# OWED - hidden=4: pre-registered second in-band capacity (PREREGISTRATION_L3.md sec.11)
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 4 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out-dir fullruns/l3_h4_traces --dump-states fullruns/l3_h4_traces/states
```

Then run `audit_behavior_mediation.py` on each `states/` directory. Decision
rule (fixed in advance, see
`docs/superpowers/specs/2026-07-12-l3-behavior-audit-design.md`): survival
`resid_trace` mean >= 0.65 strengthens the behavior-independent claim;
[0.60, 0.65) weakens it to a below-bar trace; < 0.60 means the L3 signal is
largely behavior-mediated.
