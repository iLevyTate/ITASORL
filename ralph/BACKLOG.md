# Ralph Backlog

Living list of bugs, gaps, and issues. The loop reads this first and works the
highest-priority **Open** item each run. Keep entries one line where possible:
`SEV | file:line | description`.

Severity: **P0** crash/build-break · **P1** correctness/security · **P2** missing
test/contract gap · **P3** footgun/cleanup.

## Open
<!-- Ralph adds discovered issues here, highest severity first. -->

## In progress
<!-- The item currently being worked, if any. -->

## Done
<!-- Resolved items, newest first. Include the fixing commit's short SHA. -->
P3 | run_expA.py, run_expA_l2.py | Dead code flagged by CI's ruff gate: 2 unused `numpy` imports (F401) + 1 placeholder-less f-string (F541). Auto-fixed; `ruff check --select F` now clean. Fixed 2026-06-27.
P0 | experiment_b.py + agent.py | `run_expB_gap.py` & `run_expB_kstep.py` crashed (missing `train_world_model(rollout_context=, delta=)` and `RecurrentWorldModel.forward_rollout`). Implemented the open-loop rollout API; both scripts now run and reproduce FINDINGS.md (gap: engaged, delta 1.07x; kstep: no liftoff). Fixed 2026-06-27.

## Questions / needs a human
<!-- Ambiguous or product-decision items Ralph should NOT decide alone. -->
P1 | .github/workflows/ci.yml + all *.py | **CI `python` job is RED.** It runs `ruff check .` and `ruff format --check .` (default rules), but the codebase uses a deliberate compact style throughout (46 `E702` semicolons, 5 `E401` multi-import lines, plus `E701`/`E731`/`E741`). 53 lint errors remain after the F-code fixes, and `ruff format` will also reject the compact style. **This needs a style-policy decision before CI can go green** — options: (a) add `pyproject.toml [tool.ruff]` ignoring the stylistic E rules AND drop/relax the `ruff format --check` step (keeps the author's style; the formatter has no "keep semicolons" knob); or (b) run `ruff format .` + `ruff check --fix` to reformat all 14 files to ruff defaults (rewrites the established style — churn). Ralph will not impose either unilaterally.
P2 | repo root | No `requirements.txt`/`pyproject.toml` (CI's pip-cache + install steps are stubbed waiting for one) and no test suite (CI's pytest step skips itself). Both are gated behind the ruff decision above, since the CI `python` job fails at the ruff step before reaching install/test. Worth adding once the lint policy is set.
