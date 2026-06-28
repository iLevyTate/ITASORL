# Ralph — Senior-Dev Bug-Hunting Loop

A [Ralph Wiggum loop](https://awesomeclaude.ai/ralph-wiggum): feed Claude Code
the same prompt over and over in a **fresh context each time** and let it
iterate toward a goal. Progress accumulates in **files and git history**, not in
the model's context window.

This loop runs Claude as a **senior developer** whose single job is to find real
bugs, gaps, and issues — and fix them, one atomic, tested commit at a time.

## What's here

| File | Purpose |
|------|---------|
| `PROMPT.md` | The senior-dev persona, priorities, per-run procedure, and stop signal. This is the brain. |
| `ralph.sh` | The loop. Calls Claude with `PROMPT.md`, stops on completion / stall / max-iterations. |
| `BACKLOG.md` | Discovered issues. Ralph reads this first and works the top item. |
| `JOURNAL.md` | Append-only record of what each run did. |
| `logs/` | Per-iteration stdout, timestamped. |

## The command to run locally

From the repo root, on a feature branch (the script refuses to run on
`main`/`master`):

```bash
chmod +x ralph/ralph.sh          # first time only
MAX_ITERATIONS=30 ./ralph/ralph.sh
```

That's it. Watch it work; logs stream to your terminal and to `ralph/logs/`.

### Tuning knobs (all optional env vars)

```bash
MAX_ITERATIONS=30 \   # hard cap; the loop can never run forever (default 20)
STALL_LIMIT=2 \       # stop after N runs with no new commit (default 2)
SLEEP_BETWEEN=3 \     # seconds between iterations (default 3)
./ralph/ralph.sh

DRY_RUN=1 ./ralph/ralph.sh    # print the command without calling Claude
```

### How it stops

The loop exits on the first of:
1. **Completion** — the agent prints `RALPH_COMPLETE` (nothing left to fix).
2. **Stall** — `STALL_LIMIT` iterations in a row produce no new commit.
3. **Cap** — `MAX_ITERATIONS` reached.

Stop it yourself any time with **Ctrl-C**; committed work is already safe in git.

## Raw one-liner (no script)

If you want the bare Ralph in your shell:

```bash
for i in $(seq 1 20); do
  claude -p "$(cat ralph/PROMPT.md)" --dangerously-skip-permissions --permission-mode acceptEdits
done
```

The wrapper script is preferred — it adds the completion sentinel, stall
detection, the `main`-branch guard, and per-iteration logs.

## Safety notes

- Always run on a **feature branch**, never `main`. The script enforces this.
- `--dangerously-skip-permissions` lets Claude run unattended. It only ever
  executes the reviewed `PROMPT.md`, but treat the branch as untrusted until you
  review the diff: `git log --oneline` then read each commit.
- Each fix is its own commit, so anything bad is easy to revert.
