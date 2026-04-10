#!/usr/bin/env bash
# gpt5mini_empty.sh — 4 empty-taxonomy runs, one per model family.
# Each run mixes both the normal and heretic version of that model.
# 100 steps, balanced datasets, 5% revisit rate, 2 in parallel.
# Usage: bash gpt5mini_empty.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=100
LABELER="gpt-5.4-mini-2026-03-17"
LOG_DIR="$SCRIPT_DIR/gpt5mini_logs"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Each entry: "name|model_substr1,model_substr2,..."
# Multiple substrings become multiple --models flags.
# deepseek: self-heretic + normal (trailing _ excludes old plain-heretic files)
declare -a RUNS=(
    "gpt5mini_deepseek_empty|deepseek-r1-distill-llama-8b-self-heretic,deepseek-r1-distill-llama-8b_"
    "gpt5mini_gemma_empty|gemma-4-26b-a4b"
    "gpt5mini_gpt-oss_empty|gpt-oss-20b"
    "gpt5mini_qwen_empty|qwen3.5-9B"
)

run_one() {
    local name="$1" model_substrs="$2"
    local logfile="$LOG_DIR/${name}.log"
    echo "[$(date '+%H:%M:%S')] Starting: $name → $logfile"

    local model_args=()
    IFS=',' read -ra substrs <<< "$model_substrs"
    for s in "${substrs[@]}"; do
        model_args+=(--models "$s")
    done

    local cmd=(
        uv run "$SCRIPT_DIR/run.py" run
        "${model_args[@]}"
        --labeler "$LABELER"
        --steps "$STEPS"
        --name "$name"
        --no-viz
        --balance-datasets
        --revisit-rate 0.05
    )
    if $DRY_RUN; then echo "  DRY RUN: ${cmd[*]}"; return 0; fi
    if "${cmd[@]}" >"$logfile" 2>&1; then
        echo "[$(date '+%H:%M:%S')] DONE: $name"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $name (exit $?) — see $logfile"
    fi
}

echo "=== gpt-5.4-mini empty-taxonomy runs (normal + heretic per model) ==="
echo "Steps: $STEPS  |  Labeler: $LABELER  |  Parallelism: 2"
$DRY_RUN && echo "(DRY RUN)"
echo ""

pids=()
for entry in "${RUNS[@]}"; do
    IFS="|" read -r name model_substrs <<< "$entry"
    run_one "$name" "$model_substrs" &
    pids+=($!)
    if [[ ${#pids[@]} -ge 2 ]]; then
        wait "${pids[0]}"; pids=("${pids[@]:1}")
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo ""
echo "=== All runs finished ==="
