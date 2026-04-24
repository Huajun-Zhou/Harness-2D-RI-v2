#!/usr/bin/env bash
# Run all 10 E2 outer-LOCO folds sequentially.
# Usage: bash scripts/run_E2_all_folds.sh
#
# Runs ~1 min per fold (10 rounds × ~6s/round on heuristic LLM fallback).
# Total wall-clock ~10-15 minutes including PySR warmup.
set -euo pipefail

ANIONS=(O S Se Te F Cl Br I N P)
RUN_SEED=42
LOG=experiments/E2_outer_loco/run_all_folds.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

for anion in "${ANIONS[@]}"; do
    # Relocate PySR scratch from prior fold if any.
    if [[ -d outputs ]]; then
        mkdir -p "experiments/E2_outer_loco/runs/pysr_scratch_shared"
        mv outputs/* "experiments/E2_outer_loco/runs/pysr_scratch_shared/" 2>/dev/null || true
        rmdir outputs 2>/dev/null || true
    fi

    echo "=== fold anion=$anion ===" | tee -a "$LOG"
    python3 harness_runner.py \
        --experiment E2_outer_loco \
        --config "experiments/E2_outer_loco/configs/fold_${anion}.yaml" \
        --run-seed "$RUN_SEED" \
        --run-id "fold_${anion}_seed${RUN_SEED}" \
        --notes "E2 Phase 2.2.e production fold, outer_anion=${anion}" 2>&1 \
        | tee -a "$LOG" \
        | grep -E "outer RMSE|outer_anion|outer_test metrics|finished OK|Traceback|Error" \
        || true
done

# Final scratch sweep
if [[ -d outputs ]]; then
    mkdir -p "experiments/E2_outer_loco/runs/pysr_scratch_shared"
    mv outputs/* "experiments/E2_outer_loco/runs/pysr_scratch_shared/" 2>/dev/null || true
    rmdir outputs 2>/dev/null || true
fi

echo "All 10 folds done. Full log: $LOG"
