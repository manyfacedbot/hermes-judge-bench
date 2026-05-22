#!/usr/bin/env bash
# hermes-judge-bench/run.sh
#
# Usage:
#   ./run.sh                          # run all problems × all judges
#   ./run.sh --problems 000           # run only problem 000
#   ./run.sh --judges bare            # run only the bare judge
#   ./run.sh --problems 000 --judges bare
#
# Results are written to results/<problem>-<judge>.json
# Run score.py afterwards to see the leaderboard.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBLEMS_DIR="$REPO_DIR/problems"
RESULTS_DIR="$REPO_DIR/results"
ANSWERS_FILE="$REPO_DIR/answers.json"
JUDGES_FILE="$REPO_DIR/judges.yaml"
BUDGET=20000

mkdir -p "$RESULTS_DIR"

# Parse args
FILTER_PROBLEMS=""
FILTER_JUDGES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --problems) FILTER_PROBLEMS="$2"; shift 2 ;;
    --judges)   FILTER_JUDGES="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Read judges from YAML (simple grep — no yq dependency)
JUDGES=$(grep -E '^  [a-z]' "$JUDGES_FILE" | grep -v '#' | sed 's/://;s/^ *//')

# Read problems from problems/ dir
PROBLEMS=$(ls "$PROBLEMS_DIR"/*.md | xargs -I{} basename {} .md | sort)

# Apply filters
if [[ -n "$FILTER_PROBLEMS" ]]; then
  PROBLEMS=$(echo "$FILTER_PROBLEMS" | tr ',' '\n')
fi
if [[ -n "$FILTER_JUDGES" ]]; then
  JUDGES=$(echo "$FILTER_JUDGES" | tr ',' '\n')
fi

PROMPT_TEMPLATE='You are competing to solve the following problem:

---
%s
---

You will be scored on two dimensions:
- Correctness: does your Python script produce the right answer?
- Efficiency: how few tokens did you use to get there?

The winning score is the best ratio of correctness to tokens spent. Stopping
early with a correct answer beats a correct answer that took twice as long.
A wrong answer scores zero regardless of efficiency.

Rules:
- No web search. No looking up solutions.
- Write your Python script to /tmp/solution.py
- Any data files referenced in the problem are available at %s/
- You may iterate on your script as many times as needed.
- Run your script with: python3 /tmp/solution.py
- When you are confident your answer is correct, stop and say DONE.

Token budget: %d'

echo ""
echo "=== hermes-judge-bench ==="
echo "Problems : $(echo $PROBLEMS | tr '\n' ' ')"
echo "Judges   : $(echo $JUDGES | tr '\n' ' ')"
echo ""

for PROBLEM in $PROBLEMS; do
  PROBLEM_FILE="$PROBLEMS_DIR/$PROBLEM.md"
  if [[ ! -f "$PROBLEM_FILE" ]]; then
    echo "Problem file not found: $PROBLEM_FILE"
    continue
  fi

  PROBLEM_TEXT=$(cat "$PROBLEM_FILE")

  for JUDGE in $JUDGES; do
    RESULT_FILE="$RESULTS_DIR/${PROBLEM}-${JUDGE}.json"
    SOLUTION_FILE="$RESULTS_DIR/${PROBLEM}-${JUDGE}-solution.py"

    echo "Running: problem=$PROBLEM judge=$JUDGE"

    # Build prompt
    PROMPT=$(printf "$PROMPT_TEMPLATE" "$PROBLEM_TEXT" "$PROBLEMS_DIR" "$BUDGET")

    # Get judge model from YAML (naive parse — assumes model: line follows judge name)
    MODEL=$(awk "/^  $JUDGE:/{found=1} found && /model:/{print \$2; exit}" "$JUDGES_FILE")
    PROVIDER=$(awk "/^  $JUDGE:/{found=1} found && /provider:/{print \$2; exit}" "$JUDGES_FILE")

    START_TIME=$(date +%s)

    # Run hermes in oneshot mode, capture full output
    hermes -z "$PROMPT" \
      -m "$MODEL" \
      --provider "$PROVIDER" \
      -t terminal,file \
      2>/dev/null > /tmp/bench_response.txt || true

    RESPONSE=$(cat /tmp/bench_response.txt)

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Extract solution.py if agent wrote it to /tmp/solution.py
    if [[ -f "/tmp/solution.py" ]]; then
      cp /tmp/solution.py /tmp/bench_solution.py
      cp /tmp/solution.py "$SOLUTION_FILE"
      rm -f /tmp/solution.py
    else
      rm -f /tmp/bench_solution.py
    fi

    # Write result JSON safely via python
    python3 - <<PYEOF > "$RESULT_FILE"
import json
result = {
    "problem": """$PROBLEM""",
    "judge": """$JUDGE""",
    "model": """$MODEL""",
    "elapsed_seconds": $ELAPSED,
    "response": open("/tmp/bench_response.txt").read() if __import__("os").path.exists("/tmp/bench_response.txt") else "",
    "solution": open("/tmp/bench_solution.py").read() if __import__("os").path.exists("/tmp/bench_solution.py") else "",
}
print(json.dumps(result, indent=2))
PYEOF

    echo "  → saved to $RESULT_FILE (${ELAPSED}s)"
    echo ""
  done
done

echo "=== Done. Run: python3 score.py ==="
