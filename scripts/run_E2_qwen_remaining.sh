#!/usr/bin/env bash
# Continuation driver: run only the 4 Qwen-E2 folds that didn't finish
# in the original run_E2_qwen_all_folds.sh (Br, I, N, P). Uses LLM
# timeout + retry (defaults from src/v2/llm_provider.py: 90s, 1 retry).
set -euo pipefail

ANIONS=(Br I N P)
RUN_SEED=42
LOG=experiments/E2_outer_loco/run_remaining_folds_qwen.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "ERROR: DASHSCOPE_API_KEY not set." >&2
    exit 2
fi

export LLM_PROVIDER=qwen
# Explicit timeout floor; overrides defaults if someone wants them tighter.
export LLM_TIMEOUT_S="${LLM_TIMEOUT_S:-90}"
export LLM_RETRIES="${LLM_RETRIES:-1}"

for anion in "${ANIONS[@]}"; do
    if [[ -d outputs ]]; then
        mkdir -p "experiments/E2_outer_loco/runs/pysr_scratch_qwen"
        mv outputs/* "experiments/E2_outer_loco/runs/pysr_scratch_qwen/" 2>/dev/null || true
        rmdir outputs 2>/dev/null || true
    fi

    # Skip if already finished (paranoia — lets us safely rerun the script)
    if [[ -f "experiments/E2_outer_loco/runs/fold_${anion}_seed${RUN_SEED}_qwen/fold_summary.json" ]]; then
        echo "=== fold anion=$anion already done, skipping ===" | tee -a "$LOG"
        continue
    fi

    # Remove any partial run dir (e.g. Br from the hung run)
    rm -rf "experiments/E2_outer_loco/runs/fold_${anion}_seed${RUN_SEED}_qwen" 2>/dev/null || true

    echo "=== fold anion=$anion ===" | tee -a "$LOG"
    python3 harness_runner.py \
        --experiment E2_outer_loco \
        --config "experiments/E2_outer_loco/configs/fold_${anion}.yaml" \
        --run-seed "$RUN_SEED" \
        --run-id "fold_${anion}_seed${RUN_SEED}_qwen" \
        --notes "E2 Phase 2.3.X continuation — Qwen-max, outer_anion=${anion}, timeout=${LLM_TIMEOUT_S}s" 2>&1 \
        | tee -a "$LOG" \
        | grep -E "=== fold|outer RMSE|fold anion.*complete|LLM timeout|Traceback|Error: |FAILED" \
        || true
done

if [[ -d outputs ]]; then
    mkdir -p "experiments/E2_outer_loco/runs/pysr_scratch_qwen"
    mv outputs/* "experiments/E2_outer_loco/runs/pysr_scratch_qwen/" 2>/dev/null || true
    rmdir outputs 2>/dev/null || true
fi

echo "Remaining E2-Qwen folds done. Full log: $LOG"
