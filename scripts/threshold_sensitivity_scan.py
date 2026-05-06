"""
threshold_sensitivity_scan.py — SI §S2 numerical-stability threshold scan.

The main manuscript claims (§3) that the two numerical-stability veto
thresholds (5% of training predictions outside [-20, 20], 25% violating
A>=1) are not tuned to the dataset, and that a sensitivity scan over a
+/-50% range around each threshold leaves the outer-LOCO RMSE invariant
to within +/-0.005.

This script provides the data behind that claim, run as a vanilla-PySR
sensitivity probe on the largest outer-LOCO fold (I, N=39 outer-test).
For each of five threshold conditions:

  C1: tight    upper=2.5%   lower=12.5%
  C2: nominal  upper=5.0%   lower=25.0%   (the headline value)
  C3: loose    upper=12.5%  lower=37.5%
  C4: upper-only-tight   upper=2.5%   lower=25.0%
  C5: lower-only-tight   upper=5.0%   lower=12.5%

we re-run vanilla PySR under the E2 outer-LOCO partition (1 seed x 5
rounds), apply V2_screen selection AFTER filtering pooled candidates by
the per-condition thresholds, and report the held-out RMSE.

Output: experiments/E5_diagnostics/runs/threshold_scan_I_seed42/
  - fold_summary.json (per-condition outer RMSE, expression, stable_frac)
  - manifest.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.runners.E2_outer_loco import (
    E2_FEATURE_NAMES, _build_feature_matrix, _project_features,
)
from src.v2.expr_eval import rmse_on
from src.v2.seeds import SeedManager


ANION = "I"          # largest outer-LOCO fold
RUN_SEED = 42
N_ROUNDS = 5
DATA_CSV = "data/extended/tier2_binary.csv"
TARGET = "cauchy_A"

# +/-50% scan around the nominal (5%, 25%) thresholds.
CONDITIONS = [
    {"name": "C1_tight",        "upper_frac": 0.025, "lower_frac": 0.125},
    {"name": "C2_nominal",      "upper_frac": 0.050, "lower_frac": 0.250},
    {"name": "C3_loose",        "upper_frac": 0.125, "lower_frac": 0.375},
    {"name": "C4_upper_tight",  "upper_frac": 0.025, "lower_frac": 0.250},
    {"name": "C5_lower_tight",  "upper_frac": 0.050, "lower_frac": 0.125},
]

PYSR_CFG = {
    "niterations": 100, "populations": 40, "population_size": 33,
    "maxsize": 25, "parsimony": 0.05,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["exp", "log", "sqrt", "square", "inv", "cube", "cbrt"],
}
OUTPUT_DIR = Path(f"experiments/E5_diagnostics/runs/threshold_scan_{ANION}_seed42")


def _passes_thresholds(yhat_train: np.ndarray,
                        upper_frac: float, lower_frac: float) -> bool:
    """Replicate the agent council's numerical-stability veto:
    reject if more than upper_frac of training predictions exceed |y|>20,
    or more than lower_frac violate A>=1."""
    finite = np.isfinite(yhat_train)
    if not finite.any():
        return False
    yh = yhat_train[finite]
    upper_violation = float((np.abs(yh) > 20.0).mean())
    lower_violation = float((yh < 1.0).mean())
    return (upper_violation <= upper_frac) and (lower_violation <= lower_frac)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("threshold_scan")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from pysr import PySRRegressor

    df = pd.read_csv(DATA_CSV)
    outer_test_df = df[df["anion"] == ANION].reset_index(drop=True)
    outer_pool_df = df[df["anion"] != ANION].reset_index(drop=True)

    X_pool = _build_feature_matrix(outer_pool_df)
    y_pool = outer_pool_df[TARGET].to_numpy(dtype=float)
    X_outer = _build_feature_matrix(outer_test_df)
    y_outer = outer_test_df[TARGET].to_numpy(dtype=float)

    sm = SeedManager(run_seed=RUN_SEED)
    inner_split_seed = sm.derive("e2_inner_split", anion=ANION)
    rng = np.random.default_rng(inner_split_seed)
    perm = rng.permutation(len(outer_pool_df))
    cut = int(0.8 * len(perm))
    tr_idx, sc_idx = perm[:cut], perm[cut:]
    X_train, y_train = X_pool[tr_idx], y_pool[tr_idx]
    X_screen, y_screen = X_pool[sc_idx], y_pool[sc_idx]

    log.info("Threshold scan on anion=%s, outer N=%d, train N=%d, screen N=%d",
             ANION, len(outer_test_df), len(X_train), len(X_screen))

    # Run PySR ONCE (one pooled Pareto across n_rounds), then re-rank
    # under each threshold condition. This is the correct interpretation of
    # the sensitivity question: same search, different stability filter.
    pooled: list[dict] = []
    for rnd in range(N_ROUNDS):
        round_seed = sm.derive("threshold_scan", anion=ANION, round=rnd)
        t0 = time.time()
        model = PySRRegressor(
            **PYSR_CFG,
            random_state=int(round_seed % (2**31 - 1)),
            deterministic=False, verbosity=0, progress=False,
            temp_equation_file=True,
        )
        model.fit(X_train, y_train, variable_names=E2_FEATURE_NAMES)
        eqs = model.equations_
        if eqs is None or len(eqs) == 0:
            continue
        for ii in range(len(eqs)):
            try:
                yp_tr = model.predict(X_train, index=ii)
                yp_sc = model.predict(X_screen, index=ii)
            except Exception:
                continue
            if not np.all(np.isfinite(yp_tr)):
                continue
            row = model.equations_.iloc[ii]
            pooled.append({
                "round": rnd,
                "expression": str(row["equation"]),
                "complexity": int(row["complexity"]),
                "yp_train": yp_tr,
                "train_rmse": float(np.sqrt(((y_train - yp_tr) ** 2).mean())),
                "inner_screen_rmse": float(np.sqrt(((y_screen - yp_sc) ** 2).mean())),
            })
        log.info("  round %d: %d candidates pooled (+%.1fs)",
                 rnd, len(eqs), time.time() - t0)

    log.info("Total pooled candidates across %d rounds: %d", N_ROUNDS, len(pooled))

    # Apply each threshold condition: filter pooled by threshold, then
    # V2_screen-select on the survivors.
    results = []
    for cond in CONDITIONS:
        survivors = [c for c in pooled
                      if _passes_thresholds(c["yp_train"],
                                            cond["upper_frac"],
                                            cond["lower_frac"])]
        if not survivors:
            results.append({**cond, "n_survivors": 0,
                            "selected_expression": None,
                            "outer_rmse": None})
            continue
        chosen = min(survivors, key=lambda c: c["inner_screen_rmse"])
        ev = rmse_on(chosen["expression"], E2_FEATURE_NAMES,
                     X_outer, y_outer, stable_range=(0.0, 20.0))
        results.append({
            **cond,
            "n_survivors": len(survivors),
            "n_pooled": len(pooled),
            "selected_expression": chosen["expression"],
            "selected_complexity": chosen["complexity"],
            "selected_inner_screen_rmse": chosen["inner_screen_rmse"],
            "outer_rmse": ev["rmse"],
            "outer_n_finite": ev["n_finite"],
            "outer_n_total": ev["n_total"],
        })
        log.info("  %s: survivors=%d/%d outer_rmse=%.4f cplx=%d",
                 cond["name"], len(survivors), len(pooled),
                 ev["rmse"], chosen["complexity"])

    # Compute spread
    rmses = [r["outer_rmse"] for r in results if r["outer_rmse"] is not None]
    if rmses:
        spread = max(rmses) - min(rmses)
        log.info("RMSE spread across %d conditions: %.4f (range %.4f - %.4f)",
                 len(rmses), spread, min(rmses), max(rmses))
    else:
        spread = float("nan")

    out = {
        "anion": ANION,
        "run_seed": RUN_SEED,
        "n_outer_test": int(len(outer_test_df)),
        "n_inner_train": int(len(X_train)),
        "n_inner_screen": int(len(X_screen)),
        "n_rounds": N_ROUNDS,
        "n_total_pooled": len(pooled),
        "conditions": [{k: v for k, v in r.items() if k != "yp_train"}
                       for r in results],
        "rmse_spread": spread,
    }
    (OUTPUT_DIR / "fold_summary.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", OUTPUT_DIR / "fold_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
