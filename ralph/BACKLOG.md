# Ralph Backlog

Living list of bugs, gaps, and issues. The loop reads this first and works the
highest-priority **Open** item each run. Keep entries one line where possible:
`SEV | file:line | description`.

Severity: **P0** crash/build-break · **P1** correctness/security · **P2** missing
test/contract gap · **P3** footgun/cleanup.

## Open
<!-- Ralph adds discovered issues here, highest severity first. -->
P2 | experiment_b2.py:compute_gae | The A2C advantage math (per-episode GAE, padding/masking, bootstrap = 0 for terminated vs last value for truncated) is load-bearing and untested. Add correctness tests on hand-checkable inputs: a 2-step terminated episode (bootstrap 0), a truncated episode (bootstrap = last value), and that padded steps beyond `lengths` contribute zero. Hunt for off-by-one / mask leakage.
P2 | experiment_b2.py:RunningNorm | The Welford observation normalizer is untested. Verify its running mean/var match `numpy` batch statistics over a stream to tolerance, and that `freeze()` halts further updates so the probe sees the same normalization the agent trained with. Hunt for numerical drift in the parallel-variance update.
P2 | experiment_b2.py + agent_ac.py | CUDA/CPU parity + determinism: the matched-pair L0 readout must be bit-identical and the pooled readout deterministic given seeds on BOTH devices. Add a device-parametrized test (skip `cuda` when unavailable) asserting identical h_t / probe AUROC across a repeat run. This guards the keystone confound control on GPU.
P2 | experiment_b2.py:pooled_readout/collect_pool | Drop-on-early-death can silently shrink the pools under the harsh B-v2 metabolism; verify `collect_pool` only returns full-length survivors and that the `too_few_survivors` guard fires (returns NaN, not a crash) when survivors < 5. Add a test.
P3 | agent_ac.py:to_env_action | Confirm the squashed env action stays within the world's action ranges (thrust in [0,1], turn in [-1,1], binaries in {0,1}) for EXTREME raw latents (e.g. +/-50), not just typical samples. Add an extreme-value test.
P3 | run_expA.py, run_expA_l2.py, run_expB_*.py | Run scripts have no `if __name__ == "__main__"` guard - importing one executes the whole (multi-minute) experiment. Harmless as scripts but a footgun and blocks importing them in tests. Deferred: wrapping 7 files in main() is churn for no current bug. Optional cleanup.

## In progress
<!-- The item currently being worked, if any. -->

## Done
<!-- Resolved items, newest first. Include the fixing commit's short SHA. -->
P2 | tests/test_logschema.py | Hunt of logschema.py (Parquet writer) + all run scripts found NO bug: StepWriter round-trips correctly (pyarrow casts inferred arrays to the float32/int32/list<float32> schema; empty default lists + multi-batch buffering both work), manifest round-trips via JSON, and all 7 run scripts execute end-to-end (full/surprise/nonlinear reproduce FINDINGS qualitatively). Added 4 logschema round-trip tests. 19 tests pass; ruff clean. 2026-06-27.
P2 | tests/ | Deep correctness pass on world dynamics + oracles found NO new bug (dynamics, L1/L2 oracle math, snapshot/restore, surrogate ladder all verified correct; the L1 float32 dtype contract holds via numpy weak-scalar casting). Strengthened coverage with 5 tests: reactive-policy L0 bit-identity, L1 obs-format/dtype identity (test_world.py), and the Experiment A oracle + leakage gate (new test_experiment_a.py). 15 tests pass; ruff clean. 2026-06-27.
P2 | tests/ + pytest.ini | No test suite existed (CI's pytest step skipped itself). Added 10 regression tests: obs-format contract, get/set_state exact roundtrip, matched-pair L0 bit-identity, L1 quantization-only divergence (test_world.py), and the open-loop rollout API incl. future-obs independence (test_agent.py). All pass; ruff clean. Fixed 2026-06-27.
P2 | requirements.txt + .github/workflows/ci.yml | No dependency manifest existed (CI install was stubbed, pip cache disabled). Added requirements.txt (numpy/scikit-learn/matplotlib/torch, loose lower bounds) and re-enabled `cache: pip` keyed on it. Validated current env satisfies all bounds. Fixed 2026-06-27.
P1 | ruff.toml + .github/workflows/ci.yml | CI `python` job was RED: ruff lints with defaults but the repo uses a deliberate compact style (semicolons, multi-import lines). Decision (human-approved: keep style): added ruff.toml ignoring the stylistic E rules (E401/E701/E702/E731/E741) while keeping all F/E9 bug-catchers, and dropped the `ruff format --check` step (formatter can't preserve the style). `ruff check .` now passes. Fixed 2026-06-27.
P3 | run_expA.py, run_expA_l2.py | Dead code flagged by CI's ruff gate: 2 unused `numpy` imports (F401) + 1 placeholder-less f-string (F541). Auto-fixed; `ruff check --select F` now clean. Fixed 2026-06-27.
P0 | experiment_b.py + agent.py | `run_expB_gap.py` & `run_expB_kstep.py` crashed (missing `train_world_model(rollout_context=, delta=)` and `RecurrentWorldModel.forward_rollout`). Implemented the open-loop rollout API; both scripts now run and reproduce FINDINGS.md (gap: engaged, delta 1.07x; kstep: no liftoff). Fixed 2026-06-27.

## Questions / needs a human
<!-- Ambiguous or product-decision items Ralph should NOT decide alone. -->
<!-- (none open) -->
