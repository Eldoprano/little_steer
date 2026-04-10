#!/usr/bin/env bash
# wait_and_run.sh — wait for GPU to be free, then run the label evolution experiment
#
# GPU is considered "free" when:
#   - utilization < 10% AND
#   - free VRAM > 12 GB (enough for the 15.6 GB model at IQ4_XS)
#
# Checks every 15 minutes. Once free, loads Qwen 27B in LM Studio and runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMS="$HOME/.lmstudio/bin/lms"
VENV="$SCRIPT_DIR/.venv/bin/python"
LOG_FILE="/tmp/label_evolution_wait.log"
MODEL_KEY="gpt-oss-20b"
LITELLM_MODEL="openai/gpt-oss-20b"
LMSTUDIO_API="http://localhost:1234/v1"

CHECK_INTERVAL=900  # 15 minutes
GPU_UTIL_THRESHOLD=10  # % — below this is "free"
VRAM_FREE_THRESHOLD=10000  # MiB — at least 10 GB free needed (model is 12.1 GB, MXFP4 compressed)

# Experiment params — edit as needed
MODELS="deepseek qwen"
STEPS=5   # quick run first; increase once results look good
MAX_LABELS=20

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ── Check GPU ─────────────────────────────────────────────────────────────────
gpu_is_free() {
    read -r util mem_used mem_total <<< "$(nvidia-smi \
        --query-gpu=utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | tr ',' ' ')"
    local mem_free=$(( mem_total - mem_used ))
    log "GPU: ${util}% util | ${mem_used}/${mem_total} MiB used | ${mem_free} MiB free"
    [[ "$util" -lt "$GPU_UTIL_THRESHOLD" && "$mem_free" -gt "$VRAM_FREE_THRESHOLD" ]]
}

# ── Ensure LM Studio is running ───────────────────────────────────────────────
ensure_lmstudio() {
    if curl -s "$LMSTUDIO_API/models" &>/dev/null; then
        log "LM Studio already running"
        return
    fi
    log "Starting LM Studio..."
    nohup ~/Applications/LM-Studio-0.4.7-4-x64_3591f190e80eba627cbac05be200e31c.AppImage \
        --no-sandbox > /tmp/lmstudio.log 2>&1 &
    local waited=0
    while ! curl -s "$LMSTUDIO_API/models" &>/dev/null; do
        sleep 5; waited=$((waited + 5))
        if [[ $waited -ge 60 ]]; then
            log "ERROR: LM Studio didn't start within 60s"; exit 1
        fi
    done
    log "LM Studio ready after ${waited}s"
}

# ── Load model (skip if already available) ────────────────────────────────────
load_model() {
    # Check if model is already loaded
    if curl -s "$LMSTUDIO_API/models" | grep -q "\"$MODEL_KEY\""; then
        log "Model '$MODEL_KEY' already loaded in LM Studio, skipping load."
        return
    fi
    log "Loading $MODEL_KEY..."
    "$LMS" load "$MODEL_KEY" --gpu max 2>&1 | tee -a "$LOG_FILE" || {
        log "ERROR: Failed to load model"; exit 1
    }
    log "Model loaded. Waiting 10s for it to settle..."
    sleep 10
    local loaded
    loaded=$(curl -s "$LMSTUDIO_API/models" | grep -o '"id":"[^"]*"' | head -1)
    log "Loaded model: $loaded"
}

# ── Run experiment ────────────────────────────────────────────────────────────
run_experiment() {
    log "Starting label evolution experiment (${STEPS} steps)..."
    cd "$SCRIPT_DIR"
    OPENAI_API_BASE="$LMSTUDIO_API" \
    OPENAI_API_KEY="lmstudio" \
    "$VENV" run.py run \
        --models $MODELS \
        --labeler "$LITELLM_MODEL" \
        --steps "$STEPS" \
        --max-labels "$MAX_LABELS" \
        2>&1 | tee -a "$LOG_FILE"
    log "Experiment done. Showing results:"
    "$VENV" run.py visualize 2>&1 | tee -a "$LOG_FILE"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
log "=== wait_and_run.sh started ==="
log "Checking GPU every ${CHECK_INTERVAL}s. Thresholds: util<${GPU_UTIL_THRESHOLD}%, free VRAM>${VRAM_FREE_THRESHOLD}MiB"
log "Log file: $LOG_FILE"

while true; do
    if gpu_is_free; then
        log "GPU is FREE! Proceeding with experiment."
        curl -s -X POST https://ntfy.sh/eldoprano_master \
            -H "Title: GPU Free — Label Evolution Starting" \
            -d "GPU is free. Loading gpt-oss-20b and starting the label evolution experiment." \
            > /dev/null
        ensure_lmstudio
        load_model
        run_experiment
        log "=== All done ==="
        break
    else
        log "GPU busy. Sleeping ${CHECK_INTERVAL}s (next check: $(date -d "+${CHECK_INTERVAL} seconds" '+%H:%M:%S'))"
        sleep "$CHECK_INTERVAL"
    fi
done
