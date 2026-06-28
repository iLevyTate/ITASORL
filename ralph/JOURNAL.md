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
- Commit: this run.
