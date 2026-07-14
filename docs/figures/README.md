# Figure provenance

Every committed figure in this directory is produced by a deterministic script that
writes into `docs/figures/` in place (run from the repo root). Figures are committed
only deliberately, together with the numbers they support, and a provenance row here
is updated in the same commit.

| Figure | Generating script | What it shows | Backing doc section |
|--------|-------------------|---------------|---------------------|
| `expA_ceiling.png` | `scripts/run_expA.py` | L1 detectability ceiling vs grid spacing | FINDINGS.md sec. 2 |
| `expA_L2_ceiling.png` | `scripts/run_expA_l2.py` | L2 detectability ceiling vs drift strength | FINDINGS.md sec. 2 |
| `expB_incidental.png` | `scripts/run_expB_full.py` | Recurrent-state probe across the drift sweep | FINDINGS.md sec. 3 |
| `expB_channels.png` | `scripts/run_expB_surprise.py` | Recurrent-state vs prediction-error channels | FINDINGS.md sec. 3 |
| `expB_kstep.png` | `scripts/run_expB_kstep.py` | Effect of open-loop horizon on encoding | FINDINGS.md sec. 3.3 (regenerated 2026-07-13 with the table; log `fullruns/kstep_rerun_20260713.log`) |

Regenerate all of the above in one recorded pass:

```bash
python scripts/run_e2e.py --quick
```

Individual `scripts/run_exp*.py` runners rewrite only their own figure. Experiment
B-v2 / L3 figures are run artifacts: they are written into the run's `--out-dir`
under `fullruns/` and promoted to `artifacts/expB2/` (see the promotion history in
`artifacts/expB2/README.md`), not into this directory.

The regeneration contract: if a figure changes when regenerated, either the code
changed (find the commit and record it) or the environment differs (report torch,
numpy, and Python versions). A silently changed figure is a bug in process, not a
result.
