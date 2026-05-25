#!/usr/bin/env bash
# check_tokens.sh — print token usage for the current session
# Usage: bash /home/hermes/hermes-judge-bench/problems/check_tokens.sh
#
# Prints:
#   input_tokens output_tokens cache_read_tokens raw_tokens (input+output)

SESSION_ID=$(hermes sessions list --limit 1 2>/dev/null | tail -1 | awk '{print $NF}')
if [[ -z "$SESSION_ID" ]]; then
  echo "ERROR: could not find current session" >&2
  exit 1
fi

hermes sessions export --session-id "$SESSION_ID" - 2>/dev/null | python3 -c "
import json, sys
lines = [l for l in sys.stdin if l.strip()]
if not lines:
    print('ERROR: no session data')
    sys.exit(1)
d = json.loads(lines[0])
inp = d.get('input_tokens', 0)
out = d.get('output_tokens', 0)
cr  = d.get('cache_read_tokens', 0)
raw = inp + out
print(json.dumps({
    'input_tokens':       inp,
    'output_tokens':      out,
    'cache_read_tokens':  cr,
    'raw_tokens':         raw,
}))
"
