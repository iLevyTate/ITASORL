# Scripts

Deterministic reproduction runners. Execute from the **repo root**:

```bash
python scripts/run_e2e.py --quick     # pytest + all experiments (recorded)
python scripts/run_expA.py            # Experiment A, L1
python scripts/run_expB2.py           # Experiment B-v2 (GPU if available)
python -m itasorl.experiment_b        # Experiment B smoke test
```

See the root `README.md` for the full list.

## run_local.py

Run any Colab notebook `RUN_PROFILE` locally with resume support.

```bash
python scripts/run_local.py bv3_regime_n10          # first run
python scripts/run_local.py bv3_regime_n10 --resume  # after any interruption
```

Preflights: CUDA must be visible (override with `--allow-cpu`) and at least 4 GB free RAM. For expB2, progress checkpoints are written per (drift, seed) cell under `<run>/artifacts/cells/`.
