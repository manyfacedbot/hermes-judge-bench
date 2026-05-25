#!/usr/bin/env bash
# hermes-judge-bench/run.sh
#
# Usage:
#   ./run.sh                                      # run all problems × all judges
#   ./run.sh --problems 000                       # run only problem 000
#   ./run.sh --judges bare                        # run only the bare judge
#   ./run.sh --problems 100,101 --judges bare,judge-face
#   ./run.sh --token-budget 50000 --wall-budget 600
#
# Each invocation is a "heat" with its own timestamped directory:
#   results/heat_<unix_timestamp>/
#     <judge>/
#       <problem>-response.txt    # full agent response
#       <problem>-solution.py     # extracted solution (if agent wrote one)
#       <problem>-result.json     # metadata + token counts
#
# Run score.py afterwards to see the leaderboard (reads all heats).
# Run score.py --heat <id> to score a specific heat.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBLEMS_DIR="$REPO_DIR/problems"
RESULTS_DIR="$REPO_DIR/results"
JUDGES_FILE="$REPO_DIR/judges.yaml"
TOKEN_BUDGET=20000   # informational — told to agent; not enforced by hermes itself
WALL_BUDGET=300      # hard — enforced via timeout(1); exit 124 = killed

# Cache token weight for scoring: cache_read_tokens count at this fraction of a full token.
# Anthropic prompt caching cannot be disabled in Hermes (hard-coded auto-detect).
# Cache reads are billed at ~10% of full price but still represent processed context,
# so we include them at a configurable weight (default 0.1 = billing-proportional).
# Set to 1.0 to treat cache reads as full tokens, 0.0 to exclude entirely.
CACHE_READ_WEIGHT="0.1"

mkdir -p "$RESULTS_DIR"

# Parse args
FILTER_PROBLEMS=""
FILTER_JUDGES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --problems)      FILTER_PROBLEMS="$2";  shift 2 ;;
    --judges)        FILTER_JUDGES="$2";    shift 2 ;;
    --token-budget)  TOKEN_BUDGET="$2";     shift 2 ;;
    --wall-budget)   WALL_BUDGET="$2";      shift 2 ;;
    --cache-weight)  CACHE_READ_WEIGHT="$2"; shift 2 ;;
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

# Create heat directory for this run
HEAT_ID="heat_$(date +%s)"
HEAT_DIR="$RESULTS_DIR/$HEAT_ID"
mkdir -p "$HEAT_DIR"

PROMPT_TEMPLATE='You are competing to solve the following problem against other agents:

---
%s
---

You will be scored on:
  score = result_quality / (1 + tokens_used / 10000)

where tokens_used = your total input + output tokens this session, and
result_quality depends on the problem (correctness for standard problems,
compression_ratio for poetry problems).

Stopping early with a good answer beats a marginally better answer that cost
twice as many tokens. A wrong answer or failed verification scores zero.

Rules:
- No web search. No looking up solutions.
- Write your Python script to /tmp/solution.py
- Any data files referenced in the problem are available at %s/
- You may iterate on your script as many times as needed.
- Run your script with: python3 /tmp/solution.py
- When you are confident your answer is correct, stop and say DONE.

Hard limits (enforced externally — you will be cut off if you exceed either):
  Token budget : %d tokens (input + output combined)
  Wall-clock   : %d seconds

Stay well within these limits. A correct answer with tokens to spare
beats a better answer that is cut off mid-run.'

echo ""
echo "=== hermes-judge-bench ==="
echo "Heat         : $HEAT_ID"
echo "Token budget : ${TOKEN_BUDGET} tokens"
echo "Wall budget  : ${WALL_BUDGET}s"
echo "Cache weight : ${CACHE_READ_WEIGHT} (cache_read_tokens counted at this fraction)"
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
    # Per-heat, per-judge directory
    JUDGE_DIR="$HEAT_DIR/$JUDGE"
    mkdir -p "$JUDGE_DIR"

    RESULT_FILE="$JUDGE_DIR/${PROBLEM}-result.json"
    SOLUTION_FILE="$JUDGE_DIR/${PROBLEM}-solution.py"
    RESPONSE_FILE="$JUDGE_DIR/${PROBLEM}-response.txt"

    echo "Running: problem=$PROBLEM judge=$JUDGE  →  $JUDGE_DIR/"

    # Build prompt
    PROMPT=$(printf "$PROMPT_TEMPLATE" "$PROBLEM_TEXT" "$PROBLEMS_DIR" "$TOKEN_BUDGET" "$WALL_BUDGET")

    # Get judge model/provider from YAML
    MODEL=$(awk "/^  $JUDGE:/{found=1} found && /model:/{print \$2; exit}" "$JUDGES_FILE")
    PROVIDER=$(awk "/^  $JUDGE:/{found=1} found && /provider:/{print \$2; exit}" "$JUDGES_FILE")

    START_TIME=$(date +%s)

    # Run hermes under wall-clock timeout
    timeout "${WALL_BUDGET}" \
      hermes -z "$PROMPT" \
        -m "$MODEL" \
        --provider "$PROVIDER" \
        -t terminal,file \
      2>/dev/null > "$RESPONSE_FILE" || TIMED_OUT=$?

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    KILLED_BY_TIMEOUT=false
    if [[ "${TIMED_OUT:-0}" -eq 124 ]]; then
      KILLED_BY_TIMEOUT=true
      echo "  ⏱  Wall-clock budget exceeded (${WALL_BUDGET}s) — run terminated"
    fi
    unset TIMED_OUT

    # Grab actual token counts from the session hermes just created
    SESSION_ID=$(hermes sessions list --limit 1 2>/dev/null | tail -1 | awk '{print $NF}')
    INPUT_TOKENS=0
    OUTPUT_TOKENS=0
    CACHE_READ_TOKENS=0
    CACHE_WRITE_TOKENS=0
    if [[ -n "$SESSION_ID" ]]; then
      TOKEN_JSON=$(hermes sessions export --session-id "$SESSION_ID" - 2>/dev/null | \
        python3 -c "
import json, sys
data = [json.loads(l) for l in sys.stdin if l.strip()]
if data:
    r = data[0]
    print(r.get('input_tokens', 0), r.get('output_tokens', 0), r.get('cache_read_tokens', 0), r.get('cache_write_tokens', 0))
else:
    print('0 0 0 0')
" 2>/dev/null || echo "0 0 0 0")
      read INPUT_TOKENS OUTPUT_TOKENS CACHE_READ_TOKENS CACHE_WRITE_TOKENS <<< "$TOKEN_JSON"
    fi

    # Extract solution.py if agent wrote it to /tmp/solution.py
    if [[ -f "/tmp/solution.py" ]]; then
      cp /tmp/solution.py "$SOLUTION_FILE"
      cp /tmp/solution.py /tmp/bench_solution.py
      rm -f /tmp/solution.py
    else
      rm -f /tmp/bench_solution.py
    fi

    # Write result JSON (all shell values passed as argv — no heredoc quoting issues)
    python3 -c "
import json, sys, os

response = open(sys.argv[13]).read() if os.path.exists(sys.argv[13]) else ''
solution = open(sys.argv[14]).read() if os.path.exists(sys.argv[14]) else ''

inp   = int(sys.argv[8])
out   = int(sys.argv[9])
cr    = int(sys.argv[10])
cw    = int(sys.argv[11])
wt    = float(sys.argv[12])
# effective tokens = input + output + cache_read * weight
effective = inp + out + round(cr * wt)

result = {
    'heat':               sys.argv[1],
    'problem':            sys.argv[2],
    'judge':              sys.argv[3],
    'model':              sys.argv[4],
    'elapsed_seconds':    int(sys.argv[5]),
    'token_budget':       int(sys.argv[6]),
    'wall_budget':        int(sys.argv[7]),
    'killed_by_timeout':  sys.argv[15] == 'true',
    'input_tokens':       inp,
    'output_tokens':      out,
    'cache_read_tokens':  cr,
    'cache_write_tokens': cw,
    'cache_read_weight':  wt,
    'effective_tokens':   effective,
    'response_file':      os.path.basename(sys.argv[13]),
    'solution_file':      os.path.basename(sys.argv[14]) if os.path.exists(sys.argv[14]) else None,
}
print(json.dumps(result, indent=2))
" "$HEAT_ID" "$PROBLEM" "$JUDGE" "$MODEL" "$ELAPSED" "$TOKEN_BUDGET" "$WALL_BUDGET" \
  "$INPUT_TOKENS" "$OUTPUT_TOKENS" "$CACHE_READ_TOKENS" "$CACHE_WRITE_TOKENS" "$CACHE_READ_WEIGHT" \
  "$RESPONSE_FILE" "$SOLUTION_FILE" "$KILLED_BY_TIMEOUT" \
  > "$RESULT_FILE"

    echo "  → $RESULT_FILE  (${ELAPSED}s, $((INPUT_TOKENS + OUTPUT_TOKENS)) raw tokens, ${CACHE_READ_TOKENS} cache reads)"
    echo ""
  done
done

echo "=== Done. Heat: $HEAT_ID ==="
echo "    Results : $HEAT_DIR"
echo "    Score   : python3 score.py --heat $HEAT_ID"
echo "    All     : python3 score.py"
