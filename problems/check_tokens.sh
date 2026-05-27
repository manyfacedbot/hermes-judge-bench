#!/usr/bin/env bash
# check_tokens.sh — print token usage for the current session
# Usage: bash /home/hermes/hermes-judge-bench/problems/check_tokens.sh
#
# Prints two lines:
#   tokens_spent: N
#   tokens_remaining: N
#
# The tournament writes the round budget to /tmp/archipelago_budget before invoking the agent.

BUDGET_FILE="/tmp/archipelago_budget"
SESSION_ID=$(hermes sessions list --limit 1 2>/dev/null | tail -1 | awk '{print $NF}')
if [[ -z "$SESSION_ID" ]]; then
  echo "ERROR: could not find current session" >&2
  exit 1
fi

hermes sessions export --session-id "$SESSION_ID" - 2>/dev/null | python3 -c "
import json, sys, os

lines = [l for l in sys.stdin if l.strip()]
if not lines:
    print('ERROR: no session data')
    sys.exit(1)
d = json.loads(lines[0])
spent = d.get('input_tokens', 0) + d.get('output_tokens', 0)

budget = None
budget_file = os.environ.get('ARCHIPELAGO_BUDGET_FILE', '/tmp/archipelago_budget')
try:
    with open(budget_file) as f:
        budget = int(f.read().strip())
except Exception:
    pass

print(f'tokens_spent: {spent}')
if budget is not None:
    print(f'tokens_remaining: {max(0, budget - spent)}')
else:
    print('tokens_remaining: unknown (budget file not found)')
"
