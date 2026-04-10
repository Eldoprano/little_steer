#!/usr/bin/env bash
# gpt5mini_runs.sh
# 8 runs with gpt-5.4-mini-2026-03-17 as labeler:
#   - 4 empty-taxonomy runs (one per heretic model family)
#   - 4 seeded runs starting from the arxiv_paper.json taxonomy (max-labels=25)
# 100 steps each, datasets balanced, 5% revisit rate, 2 runs in parallel.
# Uses the OPENAI_API_KEY already in the environment.
#
# Usage: bash gpt5mini_runs.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=100
LABELER="gpt-5.4-mini-2026-03-17"
SEED_FILE="$SCRIPT_DIR/seeds/arxiv_paper.json"
LOG_DIR="$SCRIPT_DIR/gpt5mini_logs"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Each entry: "name|model_substring|extra_flags"
# Empty runs: blank slate, max-labels=20
# Seeded runs: start from arxiv_paper.json, max-labels=25 (room to CREATE new labels)
declare -a RUNS=(
    "gpt5mini_deepseek_empty|deepseek-r1-distill-llama-8b-self-heretic|"
    "gpt5mini_deepseek_seeded|deepseek-r1-distill-llama-8b-self-heretic|--seed $SEED_FILE --max-labels 30"
    "gpt5mini_gemma_empty|gemma-4-26b-a4b-heretic|"
    "gpt5mini_gemma_seeded|gemma-4-26b-a4b-heretic|--seed $SEED_FILE --max-labels 30"
    "gpt5mini_gpt-oss_empty|gpt-oss-20b-heretic|"
    "gpt5mini_gpt-oss_seeded|gpt-oss-20b-heretic|--seed $SEED_FILE --max-labels 30"
    "gpt5mini_qwen_empty|qwen3.5-9B-heretic|"
    "gpt5mini_qwen_seeded|qwen3.5-9B-heretic|--seed $SEED_FILE --max-labels 30"
)

run_one() {
    local name="$1"
    local model_substr="$2"
    local extra_flags="$3"
    local logfile="$LOG_DIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $name (model: '$model_substr') → $logfile"

    local cmd=(
        uv run "$SCRIPT_DIR/run.py" run
        --models "$model_substr"
        --labeler "$LABELER"
        --steps "$STEPS"
        --name "$name"
        --no-viz
        --balance-datasets
        --revisit-rate 0.05
    )

    # Append extra flags (seed, max-labels) if any
    if [[ -n "$extra_flags" ]]; then
        # shellcheck disable=SC2206
        cmd+=($extra_flags)
    fi

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

echo "=== gpt-5.4-mini label evolution runs ==="
echo "Steps: $STEPS  |  Labeler: $LABELER  |  Parallelism: 2"
echo "Runs: 4 empty + 4 seeded (arxiv_paper.json, max-labels=25)"
echo "Logs: $LOG_DIR"
$DRY_RUN && echo "(DRY RUN — no actual API calls)"
echo ""

# Run in batches of 2
pids=()
for entry in "${RUNS[@]}"; do
    IFS="|" read -r name model_substr extra_flags <<< "$entry"
    run_one "$name" "$model_substr" "$extra_flags" &
    pids+=($!)

    if [[ ${#pids[@]} -ge 2 ]]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "=== All runs finished ==="
