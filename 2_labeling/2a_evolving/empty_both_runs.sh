#!/usr/bin/env bash
# empty_both_runs.sh — Empty-taxonomy runs mixing normal + heretic for each model.
# Covers models where previous empty runs used heretic-only or normal-only.
# phi is omitted (no heretic variant exists, phi_empty is already normal-only).
# 100 steps, balanced datasets, 5% revisit rate, sequential (for live progress).
# Usage: bash empty_both_runs.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=100
LABELER="gpt-5.4-mini-2026-03-17"
LOG_DIR="$SCRIPT_DIR/gpt5mini_logs"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Each entry: "name|model_substr1,model_substr2,..."
# Trailing _ on normal substrings excludes heretic variants (e.g. "gemma-4-26b-a4b_" won't match "gemma-4-26b-a4b-heretic_...")
declare -a RUNS=(
    "gpt5mini_deepseek_empty_both|deepseek-r1-distill-llama-8b-self-heretic,deepseek-r1-distill-llama-8b_"
    "gpt5mini_gemma_empty_both|gemma-4-26b-a4b-heretic,gemma-4-26b-a4b_"
    "gpt5mini_gpt-oss_empty_both|gpt-oss-20b-heretic,gpt-oss-20b_"
    "gpt5mini_qwen_empty_both|qwen3.5-9B-heretic,qwen3.5-9B_"
    "ministral_empty_both|ministral-3-8B-Reasoning-2512-self-heretic,ministral-3-8B-Reasoning-2512_"
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
    # FORCE_COLOR=1 makes rich emit ANSI codes even when piped through tee.
    # PIPESTATUS[0] captures the Python exit code (not tee's).
    FORCE_COLOR=1 "${cmd[@]}" 2>&1 | tee "$logfile"
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "[$(date '+%H:%M:%S')] DONE: $name"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $name — see $logfile"
    fi
}

ask_continue() {
    local next_name="$1"
    local remaining="$2"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Next run: $next_name  ($remaining remaining including this one)"
    echo "  [y] Start next run    [a] Yes to all    [n] Stop here"
    echo -n "  > "
    read -r answer </dev/tty
    case "$answer" in
        a|A) echo "  Running all remaining runs without further prompts."; YES_TO_ALL=true ;;
        n|N) echo "  Stopping. Goodbye!"; exit 0 ;;
        *)   ;; # anything else (including y/Y/enter) continues
    esac
    echo ""
}

YES_TO_ALL=false

echo "=== Empty-taxonomy runs: normal + heretic per model ==="
echo "Steps: $STEPS  |  Labeler: $LABELER  |  Sequential (live progress)"
$DRY_RUN && echo "(DRY RUN)"
echo ""

total=${#RUNS[@]}
for i in "${!RUNS[@]}"; do
    entry="${RUNS[$i]}"
    IFS="|" read -r name model_substrs <<< "$entry"

    if [[ $i -gt 0 ]] && ! $YES_TO_ALL && ! $DRY_RUN; then
        remaining=$(( total - i ))
        ask_continue "$name" "$remaining"
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    run_one "$name" "$model_substrs"
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "=== All runs finished ==="
