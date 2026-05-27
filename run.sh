#!/usr/bin/env bash
# Thin shim — the real entry point is run.py.
# The previous run.sh exec'd `hermes -z`, which never triggered the /goal
# loop (one-shot mode bypasses slash commands), so it tested the agent's
# main model, not the judge. run.py drives the real goal_judge auxiliary
# under an isolated HERMES_HOME per heat.
set -euo pipefail
exec python3 "$(cd "$(dirname "$0")" && pwd)/run.py" "$@"
