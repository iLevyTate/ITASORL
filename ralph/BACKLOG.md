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
P0 | experiment_b.py + agent.py | `run_expB_gap.py` & `run_expB_kstep.py` crashed (missing `train_world_model(rollout_context=, delta=)` and `RecurrentWorldModel.forward_rollout`). Implemented the open-loop rollout API; both scripts now run and reproduce FINDINGS.md (gap: engaged, delta 1.07x; kstep: no liftoff). Fixed 2026-06-27.

## Questions / needs a human
<!-- Ambiguous or product-decision items Ralph should NOT decide alone. -->
