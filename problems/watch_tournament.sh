#!/usr/bin/env bash
# watch_tournament.sh — live tail of tournament progress + per-agent heartbeats
#
# Streams three sources into one terminal:
#   /tmp/arch.log               — tournament runner stdout (round/game results)
#   ~/.hermes/logs/agent.log    — hermes heartbeat: API calls, tool completions, errors
#   results/heat_*/agent logs   — each agent's final response text
#
# Usage: bash problems/watch_tournament.sh [heat_id]

RESULTS_DIR="$(dirname "$(dirname "$(realpath "$0")")")/results"
TOURNAMENT_LOG="/tmp/arch.log"
HERMES_LOG="$HOME/.hermes/logs/agent.log"

if [[ -n "$1" ]]; then
    HEAT_DIR="$RESULTS_DIR/$1"
else
    HEAT_DIR=$(ls -td "$RESULTS_DIR"/heat_* 2>/dev/null | head -1)
fi

if [[ -z "$HEAT_DIR" || ! -d "$HEAT_DIR" ]]; then
    echo "No heat directory found under $RESULTS_DIR"
    exit 1
fi

echo "========================================"
echo "Watching heat: $(basename "$HEAT_DIR")"
echo "========================================"

# Note the current line count in the hermes log so we only tail new lines
HERMES_START_LINE=$(wc -l < "$HERMES_LOG" 2>/dev/null || echo 0)

# Wait for tournament log then tail it
(
    while [[ ! -f "$TOURNAMENT_LOG" ]]; do sleep 0.5; done
    tail -f "$TOURNAMENT_LOG" | sed 's/^/[tournament] /'
) &

# Tail hermes log from current position — only new lines written during this tournament
(
    tail -n +"$((HERMES_START_LINE + 1))" -f "$HERMES_LOG" | sed 's/^/[hermes] /'
) &

# Poll for new agent response logs and tail each one as it appears
declare -A tailing
while true; do
    for f in "$HEAT_DIR"/*/300-round*-agent.log; do
        [[ -f "$f" ]] && [[ -z "${tailing[$f]}" ]] && {
            tailing[$f]=1
            label=$(echo "$f" | sed "s|$HEAT_DIR/||")
            tail -f "$f" | sed "s|^|[$label] |" &
        }
    done

    if ! pgrep -f "archipelago_tournament.py" > /dev/null 2>&1; then
        sleep 3
        # pick up any final logs
        for f in "$HEAT_DIR"/*/300-round*-agent.log; do
            [[ -f "$f" ]] && [[ -z "${tailing[$f]}" ]] && {
                tailing[$f]=1
                label=$(echo "$f" | sed "s|$HEAT_DIR/||")
                tail -f "$f" | sed "s|^|[$label] |" &
            }
        done
        echo "[watch] tournament process exited."
        wait
        exit 0
    fi

    sleep 2
done
