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
- Commit: this run.
