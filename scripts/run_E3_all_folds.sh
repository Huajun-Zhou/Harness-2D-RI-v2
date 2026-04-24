#!/usr/bin/env bash
# Run all 10 E3 matched-PySR folds sequentially.
# Each fold: 5 seeds × 10 rounds = 50 PySR fits.
# Expect ~5-6 min per fold → ~50-60 min total.
set -euo pipefail

ANIONS=(O S Se Te F Cl Br I N P)
RUN_SEED=42
LOG=experiments/E3_matched_pysr_baselines/run_all_folds.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

for anion in "${ANIONS[@]}"; do
    if [[ -d outputs ]]; then
        mkdir -p "experiments/E3_matched_pysr_baselines/runs/pysr_scratch_shared"
        mv outputs/* "experiments/E3_matched_pysr_baselines/runs/pysr_scratch_shared/" 2>/dev/null || true
        rmdir outputs 2>/dev/null || true
    fi

    echo "=== fold anion=$anion ===" | tee -a "$LOG"
    python3 harness_runner.py \
        --experiment E3_matched_pysr_baselines \
        --config "experiments/E3_matched_pysr_baselines/configs/fold_${anion}.yaml" \
        --run-seed "$RUN_SEED" \
        --run-id "fold_${anion}_seed${RUN_SEED}" \
        --notes "E3 Phase 2.3.d production fold, outer_anion=${anion}" 2>&1 \
        | tee -a "$LOG" \
        | grep -E "=== fold|outer RMSE|fold anion.*complete|Traceback|Error" \
        || true
done

if [[ -d outputs ]]; then
    mkdir -p "experiments/E3_matched_pysr_baselines/runs/pysr_scratch_shared"
    mv outputs/* "experiments/E3_matched_pysr_baselines/runs/pysr_scratch_shared/" 2>/dev/null || true
    rmdir outputs 2>/dev/null || true
fi

echo "All 10 E3 folds done. Full log: $LOG"
