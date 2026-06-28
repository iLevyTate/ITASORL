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
- Commit: this run.
