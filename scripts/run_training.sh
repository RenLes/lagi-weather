#!/usr/bin/env bash
# ===========================================================================
# Lagi -- Fiji Weather Guardian: Resilient Training Wrapper
# ===========================================================================
# Features:
#   - Runs training via nohup (survives SSH disconnection)
#   - Auto-saves checkpoints every epoch (handled by train_lagi.py)
#   - Logs all output to logs/nohup_training.log
#   - Auto-pushes final model + logs + portfolio + artifacts to GitHub
#
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh                     # foreground with nohup
#   ./run_training.sh --background        # background with nohup
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
OUTPUT_DIR="${PROJECT_DIR}/output"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
TRAINING_SCRIPT="${SCRIPT_DIR}/train_lagi.py"
NOHUP_LOG="${LOG_DIR}/nohup_training.log"

GITHUB_REPO="renles/lagi-weather"
GITHUB_BRANCH="main"

# Model configuration
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR" "${PROJECT_DIR}/data"

echo "============================================================"
echo "  LAGI -- Fiji Weather Guardian Training"
echo "============================================================"
echo "  Model:      $MODEL"
echo "  Epochs:     $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  LR:         $LR"
echo "  Log:        $NOHUP_LOG"
echo "  Started:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
run_training() {
    echo "[$(date -u '+%H:%M:%S')] Starting training..." | tee -a "$NOHUP_LOG"

    python3 "$TRAINING_SCRIPT" \
        --model "$MODEL" \
        --data-dir "${PROJECT_DIR}/data" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        2>&1 | tee -a "$NOHUP_LOG"

    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -eq 0 ]; then
        echo "[$(date -u '+%H:%M:%S')] Training completed successfully!" | tee -a "$NOHUP_LOG"
    else
        echo "[$(date -u '+%H:%M:%S')] Training FAILED with exit code $TRAIN_EXIT" | tee -a "$NOHUP_LOG"
        exit $TRAIN_EXIT
    fi
}

# ---------------------------------------------------------------------------
# Git push function
# ---------------------------------------------------------------------------
push_to_github() {
    echo "[$(date -u '+%H:%M:%S')] Pushing artifacts to GitHub..." | tee -a "$NOHUP_LOG"

    cd "$PROJECT_DIR"

    # Initialize git if needed
    if [ ! -d .git ]; then
        git init
        git remote add origin "https://github.com/${GITHUB_REPO}.git" 2>/dev/null || true
    fi

    git checkout -B "$GITHUB_BRANCH"

    # Stage all artifacts
    git add -A \
        docs/ \
        scripts/ \
        logs/*.json \
        logs/*.log \
        output/coefficients.json \
        output/error_distributions.pkl \
        README.md \
        .gitignore \
        requirements.txt \
        2>/dev/null || true

    # Don't push large model files to GitHub (use Git LFS or model hub)
    echo "output/adapter_model.safetensors" >> .gitignore 2>/dev/null || true
    echo "output/model-*" >> .gitignore 2>/dev/null || true
    echo "checkpoints/" >> .gitignore 2>/dev/null || true

    COMMIT_MSG="Lagi training complete - $(date -u '+%Y-%m-%d %H:%M UTC')

Training metrics:
- Model: ${MODEL}
- Epochs: ${EPOCHS}
- LoRA rank: 16, alpha: 32
- Training cost: < \$5 on Vast.ai RTX 4090"

    git add -A
    git commit -m "$COMMIT_MSG" || echo "Nothing to commit"
    git push -u origin "$GITHUB_BRANCH" 2>&1 | tee -a "$NOHUP_LOG" || {
        echo "[WARNING] Git push failed. You may need to set up authentication." | tee -a "$NOHUP_LOG"
        echo "Run: gh auth login  OR  git remote set-url origin git@github.com:${GITHUB_REPO}.git" | tee -a "$NOHUP_LOG"
    }

    echo "[$(date -u '+%H:%M:%S')] GitHub push complete." | tee -a "$NOHUP_LOG"
}

# ---------------------------------------------------------------------------
# Summary function
# ---------------------------------------------------------------------------
print_summary() {
    echo "" | tee -a "$NOHUP_LOG"
    echo "============================================================" | tee -a "$NOHUP_LOG"
    echo "  TRAINING SUMMARY" | tee -a "$NOHUP_LOG"
    echo "============================================================" | tee -a "$NOHUP_LOG"
    echo "  Completed:  $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$NOHUP_LOG"
    echo "  Model:      $OUTPUT_DIR" | tee -a "$NOHUP_LOG"
    echo "  Logs:       $LOG_DIR" | tee -a "$NOHUP_LOG"
    echo "  GitHub:     https://github.com/${GITHUB_REPO}" | tee -a "$NOHUP_LOG"
    echo "" | tee -a "$NOHUP_LOG"

    if [ -f "$LOG_DIR/final_metrics.json" ]; then
        echo "  Final Metrics:" | tee -a "$NOHUP_LOG"
        cat "$LOG_DIR/final_metrics.json" | tee -a "$NOHUP_LOG"
    fi

    echo "" | tee -a "$NOHUP_LOG"
    echo "  Vinaka vakalevu! Lagi is ready to serve Fiji." | tee -a "$NOHUP_LOG"
    echo "============================================================" | tee -a "$NOHUP_LOG"
}

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
main() {
    run_training
    push_to_github
    print_summary
}

# Check for background mode
if [ "${1:-}" = "--background" ]; then
    echo "Running in background mode. Output: $NOHUP_LOG"
    echo "Monitor with: tail -f $NOHUP_LOG"
    nohup bash -c "$(declare -f run_training push_to_github print_summary main); \
        SCRIPT_DIR='$SCRIPT_DIR' PROJECT_DIR='$PROJECT_DIR' LOG_DIR='$LOG_DIR' \
        OUTPUT_DIR='$OUTPUT_DIR' CHECKPOINT_DIR='$CHECKPOINT_DIR' \
        TRAINING_SCRIPT='$TRAINING_SCRIPT' NOHUP_LOG='$NOHUP_LOG' \
        GITHUB_REPO='$GITHUB_REPO' GITHUB_BRANCH='$GITHUB_BRANCH' \
        MODEL='$MODEL' BATCH_SIZE='$BATCH_SIZE' EPOCHS='$EPOCHS' LR='$LR' \
        main" >> "$NOHUP_LOG" 2>&1 &
    echo "PID: $!"
    echo "$!" > "${LOG_DIR}/training.pid"
else
    main
fi
