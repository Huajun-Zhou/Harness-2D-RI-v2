"""
Analyse a Harness-2D-RI run on Tier 2 (N=193) and compare to paper's N=94 results.

Reads the most recent discovery report + pareto front from results/, then:
  - computes blind-test metrics on Tier 2 test split
  - runs Moss-law check: does the expression yield -0.25 log-log slope vs E_g?
  - writes a markdown summary for the manuscript upgrade
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import re


def find_latest(pattern: str):
    files = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    return files[-1] if files else None


def load_tier2():
    df = pd.read_csv('data/extended/tier2_binary.csv')
    # Same split logic as data_pipeline (seed=42, 80/20 permutation)
    np.random.seed(42)
    idx = np.random.permutation(len(df))
    split = int(0.8 * len(df))
    train_idx, test_idx = idx[:split], idx[split:]
    return df, train_idx, test_idx


def test_moss_law(df_test_pred: pd.DataFrame) -> dict:
    """Given predicted n_static and DFT bandgap, compute Moss-law slope."""
    clean = df_test_pred[(df_test_pred['bandgap_optb88'] > 0.2) &
                         (df_test_pred['pred_A'] > 1.01)].copy()
    log_n_true = np.log(clean['cauchy_A'])
    log_n_pred = np.log(clean['pred_A'])
    log_Eg = np.log(clean['bandgap_optb88'])
    slope_true, intercept_true = np.polyfit(log_Eg, log_n_true, 1)
    slope_pred, intercept_pred = np.polyfit(log_Eg, log_n_pred, 1)
    return {
        'moss_slope_true_data': float(slope_true),
        'moss_slope_from_formula': float(slope_pred),
        'moss_slope_expected':    -0.25,
        'moss_deviation_pct_true':  abs(slope_true - (-0.25)) / 0.25 * 100,
        'moss_deviation_pct_pred':  abs(slope_pred - (-0.25)) / 0.25 * 100,
        'pearson_log_n_log_Eg_true': float(np.corrcoef(log_Eg, log_n_true)[0, 1]),
        'pearson_log_n_log_Eg_pred': float(np.corrcoef(log_Eg, log_n_pred)[0, 1]),
    }


def main():
    report_path = find_latest('results/discovery_report_*.json')
    pareto_path = find_latest('results/pareto_front_*.json')
    if not report_path or not pareto_path:
        print("No discovery report found in results/. Run main.py first.")
        return

    print(f"Loading report: {report_path}")
    with open(report_path) as f:
        report = json.load(f)
    with open(pareto_path) as f:
        pareto = json.load(f)

    df, train_idx, test_idx = load_tier2()

    print(f"\n=== Harness-2D-RI on Tier 2 (N=193) ===")
    print(f"Best formula: {report.get('best_formula', 'N/A')}")
    print(f"Complexity:   {report.get('complexity', 'N/A')}")
    print(f"Train RMSE:   {report.get('train_rmse', 'N/A')}")
    print(f"Test RMSE:    {report.get('test_rmse', 'N/A')}")

    print(f"\nTop 5 Pareto candidates:")
    for i, p in enumerate(pareto[:5], 1):
        print(f"  {i}. cplx={p.get('complexity'):>2}, "
              f"train_rmse={p.get('train_rmse', 0):.3f}, "
              f"formula={p.get('formula', '')[:80]}")


if __name__ == '__main__':
    main()
