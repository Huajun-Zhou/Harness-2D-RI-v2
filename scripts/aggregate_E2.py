"""
aggregate_E2.py — Phase 2.2.f E2 outer-LOCO roll-up.

Reads every fold_*_summary.json under experiments/E2_outer_loco/runs/ and
produces:

  - experiments/E2_outer_loco/e2_fold_table.csv
        one row per fold with inner-train / inner-screen / outer-test metrics
  - experiments/E2_outer_loco/e2_weighted_summary.json
        sample-weighted outer-test aggregates (matches the v1 LOCO
        weighted-average convention so the paper can compare fold-for-fold)
  - stdout: a human-readable table

Usage:
    python3 scripts/aggregate_E2.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path("experiments/E2_outer_loco/runs")


def _find_summaries() -> list[Path]:
    return sorted(ROOT.glob("fold_*/fold_summary.json"))


def _weighted_avg(values: list[float], weights: list[int]) -> float:
    if not values or sum(weights) == 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def main() -> int:
    summaries = [json.loads(p.read_text()) for p in _find_summaries()]
    if not summaries:
        print("no fold summaries found under", ROOT)
        return 1

    # Sort by the anion order used elsewhere in the paper
    anion_order = ["O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P"]
    summaries.sort(key=lambda s: anion_order.index(s["outer_anion"])
                   if s["outer_anion"] in anion_order else 99)

    # --- per-fold CSV ---
    out_csv = Path("experiments/E2_outer_loco/e2_fold_table.csv")
    fieldnames = [
        "outer_anion", "n_outer_test", "n_outer_pool",
        "inner_train_rmse", "inner_screen_rmse", "inner_screen_r2",
        "headline_complexity",
        "outer_rmse", "outer_mae", "outer_r2",
        "outer_n_finite", "outer_n_total", "outer_finite_fraction",
        "headline_expression",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            h = s["headline"]
            om = s["outer_test_metrics"]
            w.writerow({
                "outer_anion": s["outer_anion"],
                "n_outer_test": s["n_outer_test"],
                "n_outer_pool": s["n_outer_pool"],
                "inner_train_rmse": f"{h['inner_train_rmse']:.4f}",
                "inner_screen_rmse": f"{h['inner_screen_rmse']:.4f}",
                "inner_screen_r2": f"{h['inner_screen_r2']:+.4f}",
                "headline_complexity": h["complexity"],
                "outer_rmse": f"{om.get('rmse', float('nan')):.4f}",
                "outer_mae": f"{om.get('mae', float('nan')):.4f}",
                "outer_r2": f"{om.get('r2', float('nan')):+.4f}",
                "outer_n_finite": om.get("n_finite", 0),
                "outer_n_total": om.get("n_total", 0),
                "outer_finite_fraction":
                    f"{om.get('finite_fraction', float('nan')):.3f}",
                "headline_expression": h["expression"],
            })

    # --- weighted summary ---
    outer_rmses = [s["outer_test_metrics"].get("rmse") for s in summaries]
    outer_r2s = [s["outer_test_metrics"].get("r2") for s in summaries]
    outer_mae = [s["outer_test_metrics"].get("mae") for s in summaries]
    weights = [s["n_outer_test"] for s in summaries]
    weighted = {
        "n_folds": len(summaries),
        "total_outer_samples": sum(weights),
        "weighted_outer_rmse": _weighted_avg(
            [r for r in outer_rmses if r is not None and not math.isnan(r)],
            [w for r, w in zip(outer_rmses, weights)
             if r is not None and not math.isnan(r)],
        ),
        "weighted_outer_mae": _weighted_avg(
            [r for r in outer_mae if r is not None and not math.isnan(r)],
            [w for r, w in zip(outer_mae, weights)
             if r is not None and not math.isnan(r)],
        ),
        "mean_outer_r2": (
            sum(r for r in outer_r2s if r is not None and not math.isnan(r))
            / max(1, sum(1 for r in outer_r2s
                         if r is not None and not math.isnan(r)))
        ),
        "worst_fold_by_rmse": max(
            summaries,
            key=lambda s: s["outer_test_metrics"].get("rmse")
                          if not math.isnan(
                              s["outer_test_metrics"].get("rmse", float("nan"))
                          )
                          else -1,
        )["outer_anion"],
        "best_fold_by_rmse": min(
            summaries,
            key=lambda s: s["outer_test_metrics"].get("rmse")
                          if not math.isnan(
                              s["outer_test_metrics"].get("rmse", float("nan"))
                          )
                          else float("inf"),
        )["outer_anion"],
    }
    Path("experiments/E2_outer_loco/e2_weighted_summary.json").write_text(
        json.dumps(weighted, indent=2)
    )

    # --- stdout table ---
    print()
    print("E2 outer-LOCO fold table")
    print("=" * 88)
    print(f"  {'anion':<6}{'N_out':>6}{'inner_tr':>10}{'inner_sc':>10}"
          f"{'outer':>10}{'outer_R²':>10}{'cplx':>6}  headline")
    print("-" * 88)
    for s in summaries:
        h = s["headline"]; om = s["outer_test_metrics"]
        expr = h.get("expression", "")
        if len(expr) > 30:
            expr = expr[:27] + "..."
        print(f"  {s['outer_anion']:<6}{s['n_outer_test']:>6}"
              f"{h['inner_train_rmse']:>10.4f}"
              f"{h['inner_screen_rmse']:>10.4f}"
              f"{om.get('rmse', float('nan')):>10.4f}"
              f"{om.get('r2', float('nan')):>+10.3f}"
              f"{h['complexity']:>6}  {expr}")
    print("-" * 88)
    print(f"  weighted outer RMSE: {weighted['weighted_outer_rmse']:.4f}  "
          f"(N={weighted['total_outer_samples']})")
    print(f"  mean outer R²:       {weighted['mean_outer_r2']:+.4f}")
    print(f"  best fold: {weighted['best_fold_by_rmse']}   "
          f"worst fold: {weighted['worst_fold_by_rmse']}")
    print()
    print(f"CSV:  {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
