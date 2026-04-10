#!/usr/bin/env bash
# resume_unconverged.sh — Resume deepseek and ministral empty runs for 100 more steps.
# Both had new labels appearing at/near step 100, indicating they had not converged.
# Usage: bash resume_unconverged.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=100
LABELER="gpt-5.4-mini-2026-03-17"
LOG_DIR="$SCRIPT_DIR/phi_ministral_logs"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

declare -a RUNS=(
    "gpt5mini_deepseek_empty"
    "ministral_empty"
)

run_one() {
    local name="$1"
    local logfile="$LOG_DIR/${name}_resume.log"
    echo "[$(date '+%H:%M:%S')] Resuming: $name → $logfile"

    local cmd=(
        uv run "$SCRIPT_DIR/run.py" run
        --run-id "$name"
        --labeler "$LABELER"
        --steps "$STEPS"
        --no-viz
    )
    if $DRY_RUN; then echo "  DRY RUN: ${cmd[*]}"; return 0; fi
    if "${cmd[@]}" >"$logfile" 2>&1; then
        echo "[$(date '+%H:%M:%S')] DONE: $name"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $name (exit $?) — see $logfile"
    fi
}

echo "=== Resuming unconverged empty runs ==="
echo "Adding $STEPS steps to: ${RUNS[*]}"
$DRY_RUN && echo "(DRY RUN)"
echo ""

pids=()
for name in "${RUNS[@]}"; do
    run_one "$name" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo ""
echo "=== All resumes finished ==="
