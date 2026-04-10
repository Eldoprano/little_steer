#!/usr/bin/env bash
# phi_ministral_empty.sh — 2 empty-taxonomy runs: phi-4-reasoning-14B and ministral (normal only).
# ministral self-heretic excluded due to poor reasoning coverage (~9% on clear_harm).
# 100 steps, balanced datasets, 5% revisit rate, 2 in parallel.
# Usage: bash phi_ministral_empty.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=100
LABELER="gpt-5.4-mini-2026-03-17"
LOG_DIR="$SCRIPT_DIR/phi_ministral_logs"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ministral: trailing underscore excludes self-heretic variant
declare -a RUNS=(
    "phi_empty|phi-4-reasoning-14B"
    "ministral_empty|ministral-3-8B-Reasoning-2512_"
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
        --sampling-seed 456
    )
    if $DRY_RUN; then echo "  DRY RUN: ${cmd[*]}"; return 0; fi
    if "${cmd[@]}" >"$logfile" 2>&1; then
        echo "[$(date '+%H:%M:%S')] DONE: $name"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $name (exit $?) — see $logfile"
    fi
}

echo "=== phi-4 + ministral empty-taxonomy runs ==="
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
