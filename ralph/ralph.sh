#!/usr/bin/env bash
#
# ralph.sh — the Ralph Wiggum loop, senior-dev bug-hunting edition.
#
# Feeds ralph/PROMPT.md to Claude Code over and over in a fresh context each
# time. State accumulates in files and git history (BACKLOG.md / JOURNAL.md),
# NOT in the model's context window. The loop stops when the agent prints the
# completion sentinel, when it stops making progress, or when MAX_ITERATIONS
# is hit — whichever comes first.
#
#   "Me fail English? That's unpossible." — Ralph Wiggum
#
# Usage:
#   ./ralph/ralph.sh                 # run with defaults
#   MAX_ITERATIONS=30 ./ralph/ralph.sh
#   DRY_RUN=1 ./ralph/ralph.sh       # print the command, don't call Claude
#
set -uo pipefail

# ---- config (override via env) ---------------------------------------------
PROMPT_FILE="${PROMPT_FILE:-ralph/PROMPT.md}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"      # hard cap so it can never run forever
SLEEP_BETWEEN="${SLEEP_BETWEEN:-3}"         # pause between iterations (seconds)
STALL_LIMIT="${STALL_LIMIT:-2}"             # stop after N runs with no git change
SENTINEL="${SENTINEL:-RALPH_COMPLETE}"      # completion signal from the agent
LOG_DIR="${LOG_DIR:-ralph/logs}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
# Flags passed to Claude. --dangerously-skip-permissions lets it run unattended;
# the loop only ever runs the senior-dev PROMPT.md, on a feature branch.
# Override by exporting CLAUDE_FLAGS as a space-separated string.
read -r -a CLAUDE_FLAGS <<< "${CLAUDE_FLAGS:---dangerously-skip-permissions --permission-mode acceptEdits}"

# ---- preflight --------------------------------------------------------------
cd "$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ralph: not inside a git repo. Aborting." >&2; exit 1; }

[ -f "$PROMPT_FILE" ] || { echo "ralph: prompt file '$PROMPT_FILE' not found." >&2; exit 1; }

if [ "${DRY_RUN:-0}" != "1" ] && ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "ralph: '$CLAUDE_BIN' not found on PATH. Install Claude Code first." >&2; exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  echo "ralph: refusing to run on '$BRANCH'. Switch to a feature branch first." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
PROMPT="$(cat "$PROMPT_FILE")"
START_TS="$(date +%Y%m%d-%H%M%S)"

echo "==========================================================="
echo " Ralph loop starting"
echo "   branch:      $BRANCH"
echo "   prompt:      $PROMPT_FILE"
echo "   max iters:   $MAX_ITERATIONS   (stall limit: $STALL_LIMIT)"
echo "   logs:        $LOG_DIR/"
echo "   stop with:   Ctrl-C   (or sentinel '$SENTINEL')"
echo "==========================================================="

stalls=0
for (( i=1; i<=MAX_ITERATIONS; i++ )); do
  log="$LOG_DIR/run-${START_TS}-iter-$(printf '%02d' "$i").log"
  before="$(git rev-parse HEAD)"

  echo ""
  echo "----- iteration $i/$MAX_ITERATIONS  ($(date '+%H:%M:%S'))  -> $log"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] $CLAUDE_BIN -p '<PROMPT.md>' ${CLAUDE_FLAGS[*]}" | tee "$log"
  else
    # Fresh context every iteration; tee so you can watch live and grep later.
    "$CLAUDE_BIN" -p "$PROMPT" "${CLAUDE_FLAGS[@]}" 2>&1 | tee "$log"
  fi

  # 1) Completion sentinel?
  if grep -q "$SENTINEL" "$log" 2>/dev/null; then
    echo ""
    echo "ralph: agent signalled '$SENTINEL'. Nothing left to do. Stopping."
    break
  fi

  # 2) Progress check — did this iteration produce a commit?
  after="$(git rev-parse HEAD)"
  if [ "$before" = "$after" ]; then
    stalls=$((stalls + 1))
    echo "ralph: no new commit this iteration ($stalls/$STALL_LIMIT stalls)."
    if [ "$stalls" -ge "$STALL_LIMIT" ]; then
      echo "ralph: stalled $STALL_LIMIT times in a row. Stopping to avoid spinning."
      break
    fi
  else
    stalls=0
    echo "ralph: committed $(git rev-parse --short "$after"). Progress made."
  fi

  [ "$i" -lt "$MAX_ITERATIONS" ] && sleep "$SLEEP_BETWEEN"
done

echo ""
echo "==========================================================="
echo " Ralph loop finished after up to $MAX_ITERATIONS iterations."
echo " Review the work:   git log --oneline   |   cat ralph/JOURNAL.md"
echo "==========================================================="
