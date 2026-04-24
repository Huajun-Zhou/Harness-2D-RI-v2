#!/usr/bin/env bash
# Re-run all 10 E2 outer-LOCO folds with LLM_PROVIDER=qwen (DashScope qwen-max).
# Output goes to fold_*_seed42_qwen/ so the heuristic E2 baseline is preserved.
#
# Usage: bash scripts/run_E2_qwen_all_folds.sh
# Requires: DASHSCOPE_API_KEY set in the environment or passed inline.
set -euo pipefail

ANIONS=(O S Se Te F Cl Br I N P)
RUN_SEED=42
LOG=experiments/E2_outer_loco/run_all_folds_qwen.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "ERROR: DASHSCOPE_API_KEY not set." >&2
    exit 2
fi

export LLM_PROVIDER=qwen

for anion in "${ANIONS[@]}"; do
    if [[ -d outputs ]]; then
        mkdir -p "experiments/E2_outer_loco/runs/pysr_scratch_qwen"
        mv outputs/* "experiments/E2_outer_loco/runs/pysr_scratch_qwen/" 2>/dev/null || true
        rmdir outputs 2>/dev/null || true
    fi

    echo "=== fold anion=$anion ===" | tee -a "$LOG"
    python3 harness_runner.py \
        --experiment E2_outer_loco \
        --config "experiments/E2_outer_loco/configs/fold_${anion}.yaml" \
        --run-seed "$RUN_SEED" \
        --run-id "fold_${anion}_seed${RUN_SEED}_qwen" \
        --notes "E2 Phase 2.3.X — Qwen-max driving Harness, outer_anion=${anion}" 2>&1 \
        | tee -a "$LOG" \
        | grep -E "=== fold|outer RMSE|fold anion.*complete|Traceback|Error: |FAILED" \
        || true
done

if [[ -d outputs ]]; then
    mkdir -p "experiments/E2_outer_loco/runs/pysr_scratch_qwen"
    mv outputs/* "experiments/E2_outer_loco/runs/pysr_scratch_qwen/" 2>/dev/null || true
    rmdir outputs 2>/dev/null || true
fi

echo "All 10 E2-Qwen folds done. Full log: $LOG"
