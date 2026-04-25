"""
Re-run a single LOMO metal (default: Cd) and merge the result back into
``experiments/E5_diagnostics/runs/lomo_seed42/fold_summary.json``.

Why: in the original LOMO run, the PySR fit for Cd seed=0 round=1 hit
the 60s wall-clock cap and was skipped (Cd ended with 4/5 rounds rather
than 5/5). PySR is non-deterministic across calls — same `random_state`
but `deterministic=False` — so a retry typically completes all 5 rounds.

We reuse the same run_seed (42) so the seed derivation tree is unchanged;
the retry simply rolls the dice again on the same logical seed.

Usage:
    python3 scripts/rerun_lomo_Cd.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.runners.E2_outer_loco import E2_FEATURE_NAMES, _build_feature_matrix
from src.v2.runners.E5_diagnostics import _run_pysr_rounds, _select_v2_screen
from src.v2.expr_eval import rmse_on
from src.v2.seeds import SeedManager


METAL = "Cd"
RUN_SEED = 42
N_SEEDS = 1
N_ROUNDS = 5
DATA_CSV = "data/extended/tier2_binary.csv"
TARGET = "cauchy_A"
STABLE_RANGE = (0.0, 20.0)
PYSR_CFG = {
    "niterations": 50,
    "populations": 40,
    "population_size": 33,
    "maxsize": 25,
    "parsimony": 0.05,
    "round_timeout_s": 90,  # bump the cap a little to give the retry room
}

LOMO_SUMMARY = Path("experiments/E5_diagnostics/runs/lomo_seed42/fold_summary.json")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("rerun_Cd")

    df = pd.read_csv(DATA_CSV)
    if METAL not in set(df["metal"].unique()):
        log.error("metal %s not present in %s", METAL, DATA_CSV)
        return 2

    sm = SeedManager(run_seed=RUN_SEED)

    outer_test_df = df[df["metal"] == METAL].reset_index(drop=True)
    outer_pool_df = df[df["metal"] != METAL].reset_index(drop=True)

    X_pool = _build_feature_matrix(outer_pool_df)
    X_outer = _build_feature_matrix(outer_test_df)
    X_full = _build_feature_matrix(df)
    y_pool = outer_pool_df[TARGET].to_numpy(dtype=float)
    y_outer = outer_test_df[TARGET].to_numpy(dtype=float)

    inner_split_seed = sm.derive("e5_lomo_split", metal=METAL)
    rng = np.random.default_rng(inner_split_seed)
    perm = rng.permutation(len(outer_pool_df))
    cut = int(0.8 * len(perm))
    tr_idx, sc_idx = perm[:cut], perm[cut:]
    X_train, y_train = X_pool[tr_idx], y_pool[tr_idx]
    X_screen, y_screen = X_pool[sc_idx], y_pool[sc_idx]

    log.info("[rerun metal=%s]  outer N=%d  train N=%d  screen N=%d",
             METAL, len(outer_test_df), len(X_train), len(X_screen))

    seed_results: list[dict] = []
    for s in range(N_SEEDS):
        t0 = time.time()
        pool = _run_pysr_rounds(
            X_train, y_train, X_screen, y_screen, X_full,
            variable_names=E2_FEATURE_NAMES,
            pysr_cfg=PYSR_CFG, sm=sm,
            seed_key=f"e5_lomo_{METAL}",
            seed_idx=s, n_rounds=N_ROUNDS,
        )
        n_rounds_completed = len({c["round"] for c in pool})
        log.info("[rerun seed=%d] %d rounds completed, %d candidates pooled, %.1fs",
                 s, n_rounds_completed, len(pool), time.time() - t0)
        if n_rounds_completed < N_ROUNDS:
            log.warning("rerun did not complete all %d rounds (got %d) — abort merge",
                        N_ROUNDS, n_rounds_completed)
            return 3

        chosen = _select_v2_screen(pool)
        if chosen is None:
            log.error("no candidate selected; abort merge")
            return 4

        ev = rmse_on(chosen["expression"], E2_FEATURE_NAMES,
                     X_outer, y_outer, stable_range=STABLE_RANGE)
        seed_results.append({
            "seed_idx": s,
            "expression": chosen["expression"],
            "complexity": chosen["complexity"],
            "inner_screen_rmse": chosen["inner_screen_rmse"],
            "outer_rmse": ev["rmse"],
            "outer_mae": ev["mae"],
            "outer_r2": ev["r2"],
            "outer_n_finite": ev["n_finite"],
            "outer_n_total": ev["n_total"],
        })

    outer_rmses = [r["outer_rmse"] for r in seed_results
                    if "outer_rmse" in r and np.isfinite(r["outer_rmse"])]
    mean_r = float(np.mean(outer_rmses)) if outer_rmses else float("nan")
    std_r = float(np.std(outer_rmses, ddof=1)) if len(outer_rmses) > 1 else 0.0
    new_metal_block = {
        "metal": METAL,
        "n_outer_test": int(len(outer_test_df)),
        "seed_runs": seed_results,
        "mean_outer_rmse": mean_r,
        "std_outer_rmse": std_r,
    }
    log.info("[rerun metal=%s]  new mean RMSE = %.4f", METAL, mean_r)

    # Merge into LOMO summary.
    summary = json.loads(LOMO_SUMMARY.read_text())
    replaced = False
    for i, r in enumerate(summary["per_metal"]):
        if r["metal"] == METAL:
            log.info("[merge] old %s RMSE = %.4f  ->  new %.4f",
                     METAL, r["mean_outer_rmse"], mean_r)
            summary["per_metal"][i] = new_metal_block
            replaced = True
            break
    if not replaced:
        log.error("metal %s not found in existing summary; abort", METAL)
        return 5

    total_N = sum(r["n_outer_test"] for r in summary["per_metal"])
    weighted = sum(r["mean_outer_rmse"] * r["n_outer_test"]
                   for r in summary["per_metal"]
                   if np.isfinite(r["mean_outer_rmse"])) / total_N if total_N else float("nan")
    log.info("[merge] LOMO weighted outer RMSE: %.4f -> %.4f",
             summary["weighted_outer_rmse"], weighted)
    summary["weighted_outer_rmse"] = float(weighted)

    LOMO_SUMMARY.write_text(json.dumps(summary, indent=2))
    log.info("[merge] wrote %s", LOMO_SUMMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
