"""
Analyse the Tier 2 rerun that had the stability veto enabled.

Reads the most recent discovery report + pareto front from results/, then:
  - reports per-candidate test RMSE, R^2, TMD stability
  - compares the new top-1 to the old Eq. (1)
  - writes a summary to results/tier2_stabveto_summary.md
"""

import glob
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd


def find_latest(pattern):
    files = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    return files[-1] if files else None


def make_eval(expr_str, fnames):
    def _safe_exp(v):
        try:
            return math.exp(v) if v < 700 else float("inf")
        except Exception:
            return float("inf")

    def _safe_log(v):
        try:
            return math.log(v) if v > 0 else float("nan")
        except Exception:
            return float("nan")

    def f(X):
        vals = []
        for row in X:
            ns = {fn: float(row[i]) for i, fn in enumerate(fnames)}
            ns.update({f"x{i}": float(row[i]) for i in range(len(row))})
            ns.update({
                "exp": _safe_exp, "log": _safe_log, "sqrt": math.sqrt,
                "square": lambda v: v**2,
                "cube": lambda v: v**3,
                "cbrt": lambda v: v**(1./3) if v >= 0 else -((-v)**(1./3)),
                "inv": lambda v: 1./v if v != 0 else float("inf"),
            })
            try:
                vals.append(float(eval(expr_str, {"__builtins__": {}}, ns)))
            except Exception:
                vals.append(float("nan"))
        return np.asarray(vals, dtype=float)

    return f


LIT_TMDS = {'MoS2':4.28,'WS2':4.02,'MoSe2':4.68,'WSe2':4.53,'MoTe2':4.90}


def main():
    report_path = find_latest('results/discovery_report_*.json')
    pareto_path = find_latest('results/pareto_front_*.json')
    if not report_path or not pareto_path:
        print("No discovery report found in results/.")
        return

    print(f"Report:  {report_path}")
    print(f"Pareto:  {pareto_path}")

    with open(report_path) as f:
        rep = json.load(f)
    with open(pareto_path) as f:
        pareto = json.load(f)

    features = rep['features_used']
    print(f"\nFeatures: {features}")
    print(f"Best formula: {rep['best_formula']}")
    print(f"Train RMSE: {rep['metrics']['train_rmse']:.3f}  "
          f"Test RMSE: {rep['metrics']['test_rmse']:.3f}  "
          f"R²: {rep['metrics']['r2']:.3f}")

    # Load Tier 2 — use the SAME feature order as production
    # (as reported in features_used: atomic_number_M, Z_ratio, d_electron_count,
    #  atomic_number_X, bond_polarity)
    df = pd.read_csv('data/extended/tier2_binary.csv')
    X_full = np.column_stack([
        df['Z_M'].values,            # x0 = atomic_number_M
        df['Z_ratio'].values,        # x1 = Z_ratio (Z_M/Z_X)
        df['d_e'].values,            # x2 = d_electron_count
        df['Z_X'].values,            # x3 = atomic_number_X
        df['bond_polarity'].values,  # x4 = bond_polarity
    ])
    y_full = df['cauchy_A'].values

    # TMD reference points (same x-ordering)
    tmd_rows = {}
    for mat, n_exp in LIT_TMDS.items():
        for an in ('S', 'Se', 'Te'):
            if mat.endswith(an + '2'):
                metal = mat.replace(an + '2', '')
                break
        r = df[(df['metal'] == metal) & (df['anion'] == an)
               & (df['stoichiometry'] == 'MX2')]
        if len(r):
            r = r.iloc[0]
            tmd_rows[mat] = (n_exp,
                             np.array([r['Z_M'], r['Z_ratio'], r['d_e'],
                                       r['Z_X'], r['bond_polarity']]))

    # Analyse pareto front
    pareto_sorted = sorted(pareto, key=lambda p: p.get('train_rmse', float('inf')))

    rows = []
    for i, c in enumerate(pareto_sorted[:10], 1):
        expr = c['expression']
        feval = make_eval(expr, ['x0', 'x1', 'x2', 'x3', 'x4'])
        preds_full = feval(X_full)
        finite = np.isfinite(preds_full)
        max_abs = float(np.nanmax(np.abs(preds_full[finite]))) if finite.any() else float('inf')
        stable_frac = float(np.mean((np.abs(preds_full) <= 20) & finite))

        # TMD check
        tmd_preds = {}
        tmd_mape = []
        for mat, (n_exp, x) in tmd_rows.items():
            p = float(feval(x[None, :])[0])
            tmd_preds[mat] = p
            if np.isfinite(p):
                tmd_mape.append(abs(p - n_exp) / n_exp * 100)

        row = {
            'rank': i,
            'cplx': c['complexity'],
            'train_rmse': c['train_rmse'],
            'test_rmse': c['test_rmse'],
            'r2': c['r2'],
            'max_abs_full': max_abs,
            'stable_frac': stable_frac,
            'tmd_stable': all(np.isfinite(v) and 0.5 <= v <= 10
                              for v in tmd_preds.values()),
            'tmd_mape': float(np.mean(tmd_mape)) if tmd_mape else float('nan'),
            **{f"pred_{m}": tmd_preds.get(m, float('nan')) for m in LIT_TMDS},
            'expression': expr[:100],
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    cols_display = ['rank', 'cplx', 'train_rmse', 'test_rmse', 'r2',
                    'stable_frac', 'tmd_stable', 'tmd_mape']
    print("\nPareto top-10 (stability-vetoed run):")
    print(summary[cols_display].to_string(index=False))

    print("\nTMD-reference predictions:")
    print(summary[['rank'] + [f"pred_{m}" for m in LIT_TMDS]].to_string(index=False))
    print(f"{'exp':>4}", end='')
    for m in LIT_TMDS:
        print(f"{LIT_TMDS[m]:>10.2f}", end='')
    print()

    # Save
    summary.to_csv('results/tier2_stabveto_pareto_top10.csv', index=False)
    print(f"\nSaved → results/tier2_stabveto_pareto_top10.csv")

    # Top-1 recommendation
    # Prefer the lowest-RMSE candidate that is stable AND tmd_stable
    good = summary[(summary['stable_frac'] >= 0.95) & (summary['tmd_stable'])]
    if len(good):
        best = good.sort_values('train_rmse').iloc[0]
        print("\n" + "=" * 60)
        print(f"RECOMMENDED HEADLINE (post-stability-veto):")
        print(f"  Rank {int(best['rank'])}  cplx={int(best['cplx'])}  "
              f"train RMSE {best['train_rmse']:.3f}  "
              f"test RMSE {best['test_rmse']:.3f}  R²={best['r2']:+.3f}")
        print(f"  TMD MAPE: {best['tmd_mape']:.1f}%")
        print(f"  Expression: {best['expression']}")
    else:
        print("\n[!] No stable TMD-valid candidate found. Check run validity.")


if __name__ == '__main__':
    main()
