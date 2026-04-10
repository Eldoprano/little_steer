#!/usr/bin/env bash
# overnight_runs.sh
# Runs label evolution for every heretic model family, 2 in parallel at a time.
# Where both heretic and self-heretic exist for a model, only self-heretic is used.
# Each run: 300 steps, no seed, gpt-oss-20b labeler, all datasets except lima.
#
# Usage: bash overnight_runs.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=300
# LABELER="gpt-oss-20b"
LABELER="qwen3.5-27b-uncensored-hauhaucs-aggressive"
LOG_DIR="$SCRIPT_DIR/overnight_logs"
export OPENAI_BASE_URL="http://localhost:1234/v1"
export OPENAI_API_KEY="lm-studio"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Each entry: "name|model_substring"
# The model_substring is passed to --models and selects all files containing that string.
# Lima is excluded by default (no --include-benign flag).
declare -a RUNS=(
    "overnightQwen_deepseek-r1-8b-self-heretic|deepseek-r1-distill-llama-8b-self-heretic"
    "overnightQwen_gemma-4-26b-heretic|gemma-4-26b-a4b-heretic"
    "overnightQwen_gpt-oss-20b-heretic|gpt-oss-20b-heretic"
    "overnightQwen_qwen3.5-heretic|qwen3.5-9B-heretic"
)

run_one() {
    local name="$1"
    local model_substr="$2"
    local logfile="$LOG_DIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $name (model pattern: '$model_substr') → $logfile"

    local cmd=(
        uv run "$SCRIPT_DIR/run.py" run
        --models "$model_substr"
        --labeler "$LABELER"
        --steps "$STEPS"
        --name "$name"
        --no-viz
    )

    if $DRY_RUN; then
        echo "  DRY RUN: ${cmd[*]}"
        return 0
    fi

    if "${cmd[@]}" >"$logfile" 2>&1; then
        echo "[$(date '+%H:%M:%S')] DONE: $name"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $name (exit $?) — see $logfile"
    fi
}

echo "=== Overnight label evolution runs ==="
echo "Steps: $STEPS  |  Labeler: $LABELER  |  Parallelism: 2"
echo "Logs: $LOG_DIR"
$DRY_RUN && echo "(DRY RUN — no actual API calls)"
echo ""

# Run in batches of 2
pids=()
for entry in "${RUNS[@]}"; do
    IFS="|" read -r name model_substr <<< "$entry"
    run_one "$name" "$model_substr" &
    pids+=($!)

    if [[ ${#pids[@]} -ge 2 ]]; then
        # Wait for the oldest job to finish before launching more
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done

# Wait for remaining jobs
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "=== All runs finished ==="
