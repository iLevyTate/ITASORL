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

## 2026-06-28 — compute_gae mask leakage (real correctness bug, A2C advantages)
- Found: top open backlog item asked to test `experiment_b2.compute_gae` and hunt
  for mask leakage. Found a REAL bug, not just a gap. The GAE accumulator gated
  its carry with the CURRENT step's mask (`gae = delta + gamma*lam*m*gae`). The
  correct GAE recursion needs the NEXT step's mask there (reset the carry across
  the episode boundary). For any episode shorter than Tmax, the loop visits the
  padded steps first (m=0, so `gae` becomes that step's delta = -V(pad)), then at
  the last valid step (m=1) multiplies that garbage in: the last in-episode
  advantage (and value target) gained a spurious `-gamma*lam*V(pad)` term.
  Full-length episodes (no padding) were correct, which hid the bug. Under the
  harsh B-v2 metabolism most episodes die early, so most episodes were affected.
- Reproduced: oracle test vs an independent textbook per-episode GAE on a 2-episode
  batch (ep0 len 2 padded to Tmax 3 with value[0,2]=0.7). Buggy last-step adv =
  1.0417 (terminated) / 1.3386 (truncated) vs oracle 1.7 / 1.997 - exactly the
  predicted -gamma*lam*0.7 leak.
- Fix: carry `next_mask` (mask of t+1, init 0) and gate the accumulator with it;
  dropped the now-redundant `nonterm` factor on the bootstrap in `delta` (next_v
  already stays at `boot` across padding, so the bootstrap was already correct and
  is unchanged). Shapes / return signature / truncation semantics all preserved.
- Verify: `python -m pytest -q` -> 33 passed (2 new compute_gae oracle tests RED
  before, GREEN after); `python -m ruff check .` -> All checks passed;
  `python experiment_b2.py` smoke runs end-to-end on CUDA, exit 0.
- Commit: 679fee6.

## 2026-06-28 — RunningNorm Welford normalizer test gap (no bug)
- Found: top open backlog item - the Welford observation normalizer
  (`experiment_b2.RunningNorm`) was untested. NO bug. The parallel-variance update
  is the standard Chan et al. merge tracking population variance directly
  (`var = (var*n_A + bv*n_B + delta^2*n_A*n_B/total)/total`), and the
  `count=1e-4, var=1` init is the deliberate Stable-Baselines3 epsilon prior that
  avoids a first-update divide-by-zero.
- Evidence: streamed 5000 samples (heterogeneous per-dim scale) in random 1..16
  batches -> running mean matches numpy batch mean to 2.0e-7, var to 1.5e-6
  absolute; single-sample stream vs one big batch agree to ~1e-14 (merge is
  associative); freeze() leaves mean/var/count bit-unchanged under a later
  large-shift update; normalized output is ~N(0,1). The only deviation is the
  prior's O(1e-4/(var*N)) bias on tiny-variance dims (~2% rel at var~1e-6) - by
  design, not drift.
- Fix:   added 4 characterization tests to tests/test_experiment_b2.py
  (streaming==numpy stats with atol on mean / rtol+atol on var, batch-size
  independence, freeze() no-op incl. returns-self, normalize-to-unit-std). These
  lock in the contract the frozen probe relies on.
- Verify: `python -m pytest -q` -> 37 passed (was 33; +4 new); `python -m ruff
  check .` -> All checks passed.
- Commit: this run.

## 2026-07-01 — Ralph pivot: experiment status + research queue
- Found: Colab full run (`fullruns/06302026`) analyzed; canonical B-v2 artifact
  promoted; FINDINGS §9 updated. Ralph was bug-only and had no read of current
  results or prioritized next steps.
- Fix:   added `ralph/EXPERIMENT_STATUS.md` (results snapshot), `ralph/NEXT_STEPS.md`
  (tiered action queue with human gates for long GPU runs), rewrote `PROMPT.md`
  to orient on status + next steps each run (bugs still P0/P1 first), extended
  `BACKLOG.md` with replication tooling + doc items and human-gated questions.
- Verify: files are self-consistent with `docs/FINDINGS.md` §9 and
  `artifacts/expB2/README.md` promotion record.
- Commit: (this session)

## 2026-07-07 - B-v3 regime n=10 power extension (H_B3 not met)
- Found: the pre-registered n=10 power extension (profile `bv3_regime_n10`,
  `fullruns/07062026`, RTX 4050, 337 min, commit 820849f) completed. Survival identity
  probe **0.610 +/- 0.047** @ drift 0.45, 90% CI [0.585, 0.634] (per-seed 0.669, 0.514,
  0.594, 0.642, 0.611, 0.586, 0.615, 0.638, 0.558, 0.677; 2/10 cross 0.65). Beats
  untrained 0.500 and predictor 0.513 by >0.05 but misses the 0.65 SESOI; volatility
  readout 0.535/0.611 also below. L0 equivalent to chance (TOST p=0.010, ROPE 0.999);
  manipulation check passed; identity target 0.610 > drag-tracking ceiling 0.487.
  Verdict: **H_B3 NOT met**, intermediate zone; the n=10 CI excludes 0.65 so the
  adjudication is decisive. Resolves the open Power (n=10) question.
- Verify: `python scripts/compare_expB2_artifacts.py --run fullruns/07062026` -> this run
  0.610 +/- 0.047 (drag ceiling 0.487) vs canonical Colab 0.523 and initial lab 0.595;
  all FAIL vs 0.65. Numbers match `fullruns/07062026/SUMMARY.md`.
- Next: report the pre-registered sysid-aux capacity ceiling (profile `bv3_ceiling`,
  ~3.7 hr; PREREG_Bv3 sec.10) to test whether 0.65 is reachable by this trunk.
  Human-gated GPU run; command prepared, not launched.
- Commit: docs-only (`ralph/EXPERIMENT_STATUS.md` + this entry); `fullruns/` is gitignored.

## 2026-07-07 - sysid-aux CEILING run + verdict-label fix (H_B2 null confirmed)
- Found: the pre-registered sysid-aux capacity ceiling (profile `bv3_ceiling`,
  `fullruns/07072026`, T4, 206 min, commit 0e69d3d; `--updates 300 --drift-mode regime
  --sysid-aux`, n=3) completed. Supervising the survival trunk directly on drag (a capacity
  control, NOT H_B2 evidence) lifts the pooled target to only **0.622** @ drift 0.45 (90% CI
  [0.576, 0.667]), barely above the unsupervised n=10 headline (0.610), while the
  detectability-style matched-pair channel reaches **~0.80** (0.785, 0.771, 0.849);
  ceiling(drag) 0.556; L0 controls at chance; leakage clean. Reading: identity IS decodable
  when forced in (matched-pair 0.80), but the pooled persistent-direction readout saturates
  ~0.62, so the 0.610 headline sits at the pooled probe's architectural ceiling. **Confirms
  the null** (not a probe-capacity floor a stronger pooled probe would clear); does not
  license chasing 0.65. L2/pooled line concluded; the "L3 if L2 null holds under scale" gate
  is now MET.
- Fix: the summary pipeline mislabeled a sysid-aux survival number as "weak / H_B2 NOT met".
  Added `_b2_used_sysid_aux()` (reads `b2_flags.json`) in `itasorl/results_io.py`; ceiling
  runs now stamp `sysid_aux` and relabel the survival verdict "ceiling" in `steps/expB2.json`,
  and `build_summary` renders a CEILING banner + headline instead of the weak-trace verdict
  (untrained/predictor controls untouched, numbers unchanged). New regression tests in
  `tests/test_results_io_ceiling.py`.
- Verify: `python -m pytest -q` -> 129 passed (3 new ceiling tests); `python -m ruff check .`
  clean. Re-rendered `fullruns/07072026` SUMMARY: headline flips to CEILING, survival rows ->
  ceiling, pooled 0.622 unchanged. Numbers match `SUMMARY.md` / `artifacts/expB2_results.json`.
- Next: L3 generative-fingerprint scope spec in BACKLOG Questions for human sign-off (no
  coding until approved, Tier-3 rule). Note: run finalize did not fully close (status.json
  running:true, no bundle.zip); science complete (manifest expB2 ok, exit 0).
- Commit: this session (`itasorl/results_io.py`, `tests/test_results_io_ceiling.py`,
  `ralph/EXPERIMENT_STATUS.md`, `ralph/BACKLOG.md`, this entry); `fullruns/` is gitignored.

## 2026-07-08 - CUDA/CPU parity + cross-run determinism test (replication-gap guard)
- Found: highest-priority [ready] item (NEXT_STEPS Tier 1 / BACKLOG P2). The keystone
  matched-pair L0 confound control and the pooled readout were only ever tested on CPU,
  yet every `fullruns/` run is on GPU and the top open question is the 0.595 (lab) vs
  0.523 (Colab) replication gap. No test pinned down what SHOULD be device-invariant.
- Fix: added two device-parametrized tests in `tests/test_experiment_b2.py`
  (DEVICES = cpu + cuda-if-present, so CI skips cuda): (1)
  `test_matched_pair_L0_bit_identical_on_device` - the L0 auth==surrogate branch
  bit-identity holds on the GPU path too (a broken GPU kernel would fabricate a
  world-identity signal only on GPU); (2) `test_readout_states_deterministic_across_runs`
  - same device + same seeds yields bit-identical `collect_pool` recurrent states across a
  repeat run, so every downstream probe AUROC is identical by construction. Cross-DEVICE
  equality is deliberately NOT asserted (CPU vs CUDA float reduction order differs); that
  FP nondeterminism is itself a plausible contributor to the replication gap, so the test
  localizes the gap to device/seed/code rather than our readout.
- Verify: pre-flight scratch confirmed 6/6 and 8/8 survivors and bit-identical repeat
  runs on BOTH cpu and cuda (RTX 4050), plus matched-pair L0 bit-identity on both.
  `python -m pytest -q` -> 135 passed (was 131; +4 = 2 tests x 2 devices);
  `python -m ruff check .` -> All checks passed.
- Next: unchanged - L3 generative-fingerprint scope spec awaits human sign-off (BACKLOG
  Questions, Tier-3 gate). Remaining in-loop [ready] items: collect_pool early-death guard
  test (P2), extreme-latent action-bounds test (P3), FINDINGS sec.7 wording (P3).
- Commit: this session (`tests/test_experiment_b2.py`, `ralph/NEXT_STEPS.md`,
  `ralph/BACKLOG.md`, this entry).

## 2026-07-08 - FINDINGS sec.7 rewrite: closed vs open levers (human-directed)
- Found: BACKLOG P3 / NEXT_STEPS Tier 2. FINDINGS sec.7 ("Next steps") listed the
  survival-coupling lever as item 1 yet closed with "the real Dreamer-style version of
  the claim", reading like an OPEN lever even though B-v2 (0.523) and the B-v3
  identifiable-regime + sysid-aux ceiling follow-ups had already returned it negative.
- Fix: human chose "all three" (ship + inline B-v3 numbers + reorganize). Retitled sec.7
  "Levers: tested and open" and split into 7.1 Closed levers (survival coupling as a
  strengthened negative with canonical 0.523 plus inlined B-v3 0.610 @ n=10, 90% CI
  [0.585, 0.634], and sysid-aux ceiling ~0.62 pooled vs ~0.80 matched-pair; the
  multi-step objective and probe-class/power items folded in as closed) and 7.2 Open
  directions (L3 generative fingerprint, held-out/common-garden probe, remaining
  objective variants). The reorg also makes sec.9's pre-existing "lever from sec.7.1"
  reference resolve for the first time; fixed the EXPERIMENT_STATUS sec.7.3->7.2 L3
  pointer to match.
- Verify: docs-only, no code touched (pytest/ruff unaffected; suite last green at 138).
  Re-read the rendered sec.7 and confirmed all numbers match EXPERIMENT_STATUS
  (0.523 canonical B-v2, 0.610 n=10 B-v3, ceiling 0.622/~0.80) and honor the no-long-dash
  house style. B-v3 artifact remains un-promoted (candidate only); numbers cited with the
  PREREGISTRATION_Bv3 provenance rather than implying promotion.
- Next: no in-loop [ready] items remain. Open work is human-gated (L3 scope sign-off) or
  tracking-only (stale BACKLOG line-13 artifact-comparison helper; script already exists).
- Commit: this session (`docs/FINDINGS.md`, `ralph/EXPERIMENT_STATUS.md`,
  `ralph/NEXT_STEPS.md`, `ralph/BACKLOG.md`, this entry).

## 2026-07-08 - extreme-latent env-action bounds safety test
- Found: BACKLOG P3 / NEXT_STEPS Tier 2. `agent_ac.to_env_action` squashes the unbounded
  Gaussian policy latent into the world's actuator ranges via sigmoid/tanh, but the only
  test (`test_act_returns_valid_env_action_and_finite_logp`) exercised TYPICAL obs. Nothing
  confirmed the bounds hold for pathological latents, and an escaped bound could NaN the
  physics integrator mid-run.
- Fix: added `tests/test_agent_ac.py::test_env_action_bounded_for_extreme_latents` - feeds
  raw latents at magnitudes 50, 1e3, 1e4, 1e8 (both signs on both continuous dims) through
  `to_env_action` and asserts thrust in [0,1], turn in [-1,1], binaries in {0,1}, all finite.
- Verify: pre-flight scratch confirmed the squash saturates gracefully at every magnitude up
  to 1e8 (sigmoid underflows to exactly 0.0, tanh clamps to +/-1.0, binaries pass through),
  so no clamp() is required. `python -m pytest -q` -> 138 passed (was 137; +1);
  `python -m ruff check .` -> clean.
- Next: only P3 doc wording remains in-loop (FINDINGS sec.7 item 1 tightening) plus the stale
  BACKLOG line-13 "artifact comparison helper" (script already exists) - both left for human
  review of scientific phrasing / tracking cleanup. L3 scope still blocked on human sign-off.
- Commit: this session (`tests/test_agent_ac.py`, `ralph/NEXT_STEPS.md`, `ralph/BACKLOG.md`,
  this entry).

## 2026-07-08 - collect_pool early-death / too_few_survivors guard tests
- Found: next-highest [ready] item (NEXT_STEPS Tier 2 / BACKLOG P2). Under the harsh B-v2
  metabolism, `collect_pool` drops episodes that die before `steps`, and `pooled_readout`
  returns a NaN `too_few_survivors` sentinel when either pool has <5 survivors. Neither the
  drop-only-full-length invariant nor the guard branch had a test, so a regression could
  silently dilute the pool or emit a spurious AUROC from a near-empty pool.
- Fix: added two tests to `tests/test_experiment_b2.py`. (1)
  `test_collect_pool_excludes_early_deaths` monkeypatches `SURVIVAL_METAB` E0 to 0.2 to force
  a real MIX (6/10 survive at steps=12), then asserts only full-length survivors are returned
  (no truncated/padded dead episodes) with H/speed aligned. (2)
  `test_pooled_readout_too_few_survivors_guard` stubs `collect_pool` to return 3 (<5)
  survivors and asserts `pooled_readout` returns `too_few_survivors is True` with every core
  metric NaN and no exception - decoupling the guard branch from the metabolism so it is fast
  and platform-independent.
- Verify: pre-flight scratch confirmed the death regime is stable (6/10 survivors across E0
  0.08-0.2 at steps=12) and the stub triggers the guard with all core keys NaN.
  `python -m pytest -q` -> 137 passed (was 135; +2); `python -m ruff check .` -> clean.
- Next: remaining in-loop [ready] items are the extreme-latent action-bounds test (P3,
  agent_ac.to_env_action) and the FINDINGS sec.7 wording tightening (P3). L3 scope still
  blocked on human sign-off. Also noticed the "artifact comparison helper" BACKLOG P2 (line
  13) looks stale - `scripts/compare_expB2_artifacts.py` already exists and NEXT_STEPS marks
  it [done]; left untouched (out of scope for this atomic change).
- Commit: this session (`tests/test_experiment_b2.py`, `ralph/NEXT_STEPS.md`,
  `ralph/BACKLOG.md`, this entry).
