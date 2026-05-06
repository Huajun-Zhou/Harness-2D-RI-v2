"""
eval_eq1_outer_loco.py — frozen evaluation of v1's precursor Eq.(1) under
outer-LOCO. Used in SI Section S5 as the counter-example to fixed-split
selection.

Eq.(1) was selected on a single random 80/20 split of Tier-2 in a v1
precursor experiment. It is a 11-node nested cube-root expression in
Z_M alone:

    A_hat = cbrt(cbrt(4.0743766 / (Z_M - 22.98636) - 0.6984615)) + 2.6530035

This script applies that frozen expression to each of the 10
outer-LOCO held-out anion slices and reports per-fold RMSE plus the
sample-weighted weighted RMSE.

Output: experiments/E5_diagnostics/runs/eq1_frozen_eval/fold_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.v2.runners.E2_outer_loco import E2_FEATURE_NAMES
from src.v2.expr_eval import compile_expr


EQ1 = "cbrt(cbrt((4.0743766 / (x0 + -22.98636)) + -0.6984615)) + 2.6530035"
ANIONS = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P")
DATA_CSV = "data/extended/tier2_binary.csv"
TARGET = "cauchy_A"
OUTPUT_DIR = Path("experiments/E5_diagnostics/runs/eq1_frozen_eval")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("eq1_eval")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    # Eq.(1) expects x0 = Z_M only. Compile with one-feature column order.
    f = compile_expr(EQ1, ["atomic_number_M"], stable_range=(0.0, 20.0))

    per_anion = []
    totN = 0; sumR = 0.0
    for a in ANIONS:
        sub = df[df["anion"] == a].reset_index(drop=True)
        X = sub["Z_M"].to_numpy(dtype=float).reshape(-1, 1)
        y = sub[TARGET].to_numpy(dtype=float)
        yhat = f(X)
        mask = np.isfinite(yhat)
        nf = int(mask.sum()); nt = int(len(y))
        if nf == 0:
            rmse = float("nan"); mae = float("nan"); r2 = float("nan")
        else:
            err = yhat[mask] - y[mask]
            rmse = float(np.sqrt((err ** 2).mean()))
            mae = float(np.abs(err).mean())
            ss_tot = float(((y[mask] - y[mask].mean()) ** 2).sum())
            ss_res = float((err ** 2).sum())
            r2 = float("nan") if ss_tot == 0 else float(1 - ss_res / ss_tot)
        per_anion.append({"anion": a, "n_outer_test": nt, "n_finite": nf,
                          "outer_rmse": rmse, "outer_mae": mae,
                          "outer_r2": r2})
        log.info("  fold %s: N=%d/%d  RMSE=%.4f  R2=%.3f",
                 a, nf, nt, rmse, r2)
        if np.isfinite(rmse):
            totN += nt; sumR += rmse * nt

    weighted = sumR / totN if totN else float("nan")
    log.info("weighted outer RMSE = %.4f on N=%d", weighted, totN)

    # Tier-2 full-set sanity check (the v1-reported held-out RMSE was on a
    # random 80/20, not the full set; the full-set RMSE is a separate number)
    X_all = df["Z_M"].to_numpy(dtype=float).reshape(-1, 1)
    y_all = df[TARGET].to_numpy(dtype=float)
    yhat_all = f(X_all)
    mask_all = np.isfinite(yhat_all)
    full_rmse = float(np.sqrt(((yhat_all[mask_all] - y_all[mask_all]) ** 2).mean()))
    log.info("Tier-2 full-set frozen RMSE = %.4f on N=%d/%d finite",
             full_rmse, int(mask_all.sum()), len(y_all))

    out = {
        "expression": EQ1,
        "expression_features": ["atomic_number_M"],
        "tier2_full_set_rmse": full_rmse,
        "tier2_full_set_n_finite": int(mask_all.sum()),
        "outer_loco_per_anion": per_anion,
        "outer_loco_weighted_rmse": weighted,
        "outer_loco_total_N": totN,
    }
    (OUTPUT_DIR / "fold_summary.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", OUTPUT_DIR / "fold_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
