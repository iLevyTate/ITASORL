# Ralph Journal

Append-only log of what each run did. Newest entries go at the bottom. The next
run reads the last few entries to avoid repeating work.

Format per entry:

```
## YYYY-MM-DD HH:MM — <short title>
- Found: <the bug/gap and how it was detected>
- Fix:   <what changed>
- Verify: <build/test command + result>
- Commit: <short SHA>
```

---
<!-- Ralph appends below this line. -->

## 2026-06-27 — open-loop rollout API (run_expB_gap / run_expB_kstep crashes)
- Found: `run_expB_gap.py` and `run_expB_kstep.py` (both documented runnable in
  README.md:93 and FINDINGS.md §3.3/§3.4/§8) crash immediately —
  `train_world_model()` had no `rollout_context`/`delta` params and
  `RecurrentWorldModel` had no `forward_rollout`. Reproduced with a direct call
  (`TypeError: unexpected keyword argument 'rollout_context'`).
- Fix:   added `RecurrentWorldModel.forward_rollout` (teacher-force `context`
  steps, then imagine open-loop feeding own predictions back) + `rollout_loss`,
  and a `self.delta` output-convention flag (absolute vs observation-change);
  wired `rollout_context`/`delta` kwargs into `train_world_model`. Default path
  (rollout_context=None) is unchanged.
- Verify: `python run_expB_gap.py` -> open-loop MSE 0.666 < mean 0.889 <
  persistence 1.348 (ENGAGED; baselines match FINDINGS to 3 dp), delta engagement
  1.07x/1.06x (matches FINDINGS "1.07x"), target AUROC at chance.
  `python run_expB_kstep.py` -> no liftoff, target 0.506/0.481/0.482 across
  horizons (FINDINGS 0.516/0.523/0.490). Both exit 0. `experiment_b.py` smoke and
  AST syntax check still pass (no regression to the default-signature callers).
- Commit: 796c9b0.

## 2026-06-27 — CI ruff gate: fix real (F) issues, defer style policy
- Found: CI's `python` job (ci.yml) runs `ruff check .` + `ruff format --check .`
  but there is no ruff config, so it lints with defaults and is RED — 57 errors.
  Breakdown: 46 E702 (semicolons) + 5 E401 (multi-import) + 1 E701 + 1 E731 +
  1 E741 (all deliberate compact style) and 3 genuine issues: 2 F401 unused
  `numpy` imports (run_expA.py, run_expA_l2.py) + 1 F541 empty f-string
  (run_expA.py).
- Fix:   auto-fixed only the 3 F-code issues (`ruff check --select F401,F541
  --fix`); `ruff check --select F` is now clean. Did NOT touch the 53 stylistic
  errors: reformatting 14 files or relaxing the CI gate is a style-policy call,
  recorded under BACKLOG "Questions / needs a human" with both options.
- Verify: `python -m py_compile run_expA.py run_expA_l2.py` OK; diff shows only
  the dead import lines + the f-string prefix changed (runtime unchanged).
- Commit: d1184af.

## 2026-06-27 — green the ruff CI gate (style-preserving, human-approved)
- Found: CI `python` job red at the ruff step (53 remaining stylistic errors);
  resolving it was a style-policy decision. Human chose option A (keep the
  compact style, configure the tool).
- Fix:   added `ruff.toml` ignoring the stylistic E rules (E401/E701/E702/E731/
  E741) while keeping all F + E9 bug-catchers; removed the `ruff format --check`
  step from ci.yml (the formatter cannot preserve the one-line style). Put the
  config in ruff.toml, NOT pyproject.toml, to avoid tripping CI's `pip install
  -e .` and release.yml's `python -m build` (no packaging metadata exists).
- Verify: `python -m ruff check .` -> "All checks passed!".
- Commit: 1b1b39b.

## 2026-06-27 — add requirements.txt + re-enable pip cache
- Found: no dependency manifest; CI's install step was stubbed (`if [ -f
  requirements.txt ]`) and the pip cache was disabled with a note to re-enable
  once a manifest existed.
- Fix:   added requirements.txt (numpy>=1.24, scikit-learn>=1.3, matplotlib>=3.7,
  torch>=2.0; bounds loose for the 3.10-3.12 matrix) and set `cache: pip` +
  `cache-dependency-path: requirements.txt` in ci.yml.
- Verify: parsed every requirement with packaging.Requirement and confirmed the
  installed versions (numpy 1.26.4 / sklearn 1.5.2 / matplotlib 3.9.4 / torch
  2.7.0) satisfy the bounds -> ALL OK.
- Commit: 0da972d.

## 2026-06-27 — add the first regression test suite
- Found: no tests existed; CI's pytest step skipped itself, and the keystone
  invariants (matched-pair L0 identity, exact snapshot/restore, the new rollout
  API) had no guard.
- Fix:   added pytest.ini (pythonpath=., testpaths=tests) and tests/ with 10
  tests - obs-format contract, get/set_state exact roundtrip, matched-pair L0
  bit-identity, L1 quantization-only divergence, and the open-loop rollout
  (teacher-forced prefix == forward, full context == forward, finite loss in
  both modes, open-loop ignores future obs).
- Verify: `python -m pytest -q` -> 10 passed in ~2.6s; `python -m ruff check .`
  -> All checks passed.
- Commit: 9cc085c.

## 2026-06-27 — deep correctness pass (world dynamics + oracles)
- Found: NO new bug. Reviewed terrain gradient + slope/gravity coupling, drag
  AR(1) stability, thermoregulation bounds, the L1 and L2 oracle math, and RNG
  determinism in get_state/set_state. One suspicion - L1Discretize emitting
  float64 and breaking the float32 obs contract - was DISPROVEN empirically:
  numpy treats the Python-float delta as a weak scalar, so float32/float stays
  float32 (holds under numpy 1.x value-based casting and 2.0 NEP 50).
- Evidence: snapshot/restore is bit-exact even under an obs-dependent policy over
  60 steps (max diff 0.0); run_expA.py validates L0 oracle 0.509 (chance), L1
  @delta=0.06 AUROC 1.000, ceiling falls to chance as delta->sigma, leakage gate
  PASS clean / FAIL on contamination.
- Fix:   converted the verified invariants into 5 regression tests (reactive-
  policy L0 identity, L1 obs-format/dtype identity, Experiment A oracle + leakage
  gate). These cover gaps the prior const-policy tests missed.
- Verify: `python -m pytest -q` -> 15 passed (~12s); `python -m ruff check .`
  -> All checks passed.
- Commit: cf33f64.

## 2026-06-27 — hunt: logschema Parquet writer + run scripts (no bug)
- Found: NO bug. Suspected StepWriter._flush would crash because pa.array infers
  float64/int64/list<double> that mismatch the float32/int32/list<float32>
  schema - DISPROVEN: pyarrow's record_batch(arrays, schema=...) casts them.
  Verified empty default lists, multi-batch buffering, activation schema, and
  manifest JSON round-trip all work. Ran the 3 remaining run scripts
  (full/surprise/nonlinear): all exit 0 and reproduce FINDINGS qualitatively, so
  all 7 run scripts now execute end-to-end.
- Fix:   added tests/test_logschema.py (4 tests: StepWriter row/type round-trip,
  incomplete-record rejection, manifest round-trip, record template coverage).
  Recorded the run-scripts no-main-guard footgun as a deferred P3 (churn for no
  current bug).
- Verify: `python -m pytest -q` -> 19 passed; `python -m ruff check .` -> All
  checks passed.
- Commit: this run.
