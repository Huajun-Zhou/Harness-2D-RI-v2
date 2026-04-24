"""
Zero-shot evaluation of frozen Tier-2 headline Eq.(1) on Tier 3 (N~532).

We load the expression discovered on Tier 2 (binaries only) and apply it
without any retraining to the full Tier 3 dataset, which includes
ternaries and quaternaries with the (M,X) assigned as the most
electropositive / most electronegative pair.

Also runs a Moss-exponent check at the Tier-3 scale and reports
binary-vs-ternary performance splits to expose any systematic
degradation when the chemical complexity increases.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def make_eval(expr_str):
    def _safe_exp(v):
        try: return math.exp(v) if v < 700 else float('inf')
        except: return float('inf')
    def _safe_log(v):
        try: return math.log(v) if v > 0 else float('nan')
        except: return float('nan')
    def f(X):
        out = []
        for row in X:
            ns = {f'x{i}': float(row[i]) for i in range(len(row))}
            ns.update({'exp': _safe_exp, 'log': _safe_log, 'sqrt': math.sqrt,
                       'square': lambda v: v**2, 'cube': lambda v: v**3,
                       'cbrt': lambda v: v**(1/3) if v >= 0 else -((-v)**(1/3)),
                       'inv': lambda v: 1/v if v != 0 else float('inf')})
            try: out.append(float(eval(expr_str, {'__builtins__': {}}, ns)))
            except: out.append(float('nan'))
        return np.asarray(out)
    return f


def rmse(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(mean_squared_error(y[m], yhat[m]))) if m.sum() > 1 else float('nan')


def main():
    rep = json.loads(Path('results/discovery_report_20260420_165841.json').read_text())
    expr = rep['best_formula']
    print(f"Frozen Eq.(1): {expr}")

    df = pd.read_csv('data/extended/tier3_binary_plus_ternary.csv')
    print(f"\nTier 3: {df.shape}  "
          f"(binary={((df['n_elem']==2).sum())}, "
          f"ternary={((df['n_elem']==3).sum())}, "
          f"quat+={((df['n_elem']>=4).sum())})")

    X = np.column_stack([df['Z_M'].values, df['Z_ratio'].values,
                         df['d_e'].values, df['Z_X'].values,
                         df['bond_polarity'].values])
    y = df['cauchy_A'].values
    pred = make_eval(expr)(X)
    finite = np.isfinite(pred) & (pred > 0) & np.isfinite(y)
    print(f"Finite predictions: {finite.sum()}/{len(df)}")

    # Stability-veto pass on the frozen expression: simple physical range
    phys_ok = finite & (pred >= 1.0) & (pred <= 20.0)
    print(f"Physical-range pass (1≤Â≤20): {phys_ok.sum()}/{finite.sum()}")

    # Overall zero-shot metrics on the physical-range subset
    rmse_all = rmse(y[phys_ok], pred[phys_ok])
    r2_all = r2_score(y[phys_ok], pred[phys_ok]) if phys_ok.sum() > 1 else float('nan')
    mae_all = float(np.mean(np.abs(y[phys_ok] - pred[phys_ok])))
    print(f"\nZero-shot on Tier 3 (physical-range pass, N={phys_ok.sum()}):")
    print(f"  RMSE = {rmse_all:.3f}")
    print(f"  MAE  = {mae_all:.3f}")
    print(f"  R^2  = {r2_all:+.3f}")

    # Splits by n_elem
    splits = {}
    for label, m in [('binary',  phys_ok & (df['n_elem']==2).values),
                     ('ternary', phys_ok & (df['n_elem']==3).values),
                     ('quat+',   phys_ok & (df['n_elem']>=4).values)]:
        if m.sum() > 1:
            r = rmse(y[m], pred[m])
            r2 = r2_score(y[m], pred[m])
            splits[label] = {'N': int(m.sum()), 'RMSE': r, 'R2': r2,
                             'MAE': float(np.mean(np.abs(y[m]-pred[m])))}
    print(f"\nSplit by chemistry (physical-range pass):")
    for k, v in splits.items():
        print(f"  {k:8s} N={v['N']:3d}  RMSE={v['RMSE']:.3f}  MAE={v['MAE']:.3f}  R^2={v['R2']:+.3f}")

    # Tier-2-vs-Tier-3 comparison on *binaries only* (are Tier-3 binaries harder?)
    df2 = pd.read_csv('data/extended/tier2_binary.csv')
    X2 = np.column_stack([df2['Z_M'], df2['Z_ratio'], df2['d_e'], df2['Z_X'], df2['bond_polarity']])
    y2 = df2['cauchy_A'].values
    p2 = make_eval(expr)(X2)
    m2 = np.isfinite(p2) & (p2>=1.0) & (p2<=20.0) & np.isfinite(y2)
    print(f"\nSanity: Tier 2 binaries (N={m2.sum()}): "
          f"RMSE={rmse(y2[m2],p2[m2]):.3f}  R^2={r2_score(y2[m2],p2[m2]):+.3f}")

    # ── Moss-exponent check at Tier 3 ─────────────────────────────
    mask = (df['bandgap_optb88'] > 0.2) & (df['cauchy_A'] > 1.02)
    slope_data, _ = np.polyfit(np.log(df.loc[mask, 'bandgap_optb88']),
                                np.log(df.loc[mask, 'cauchy_A']), 1)
    mask_pred = mask & finite & (pred > 1.02)
    slope_pred, _ = np.polyfit(np.log(df.loc[mask_pred, 'bandgap_optb88']),
                                np.log(pred[mask_pred]), 1)
    print(f"\nMoss-exponent at Tier 3:")
    print(f"  data slope  = {slope_data:+.3f}  (N={mask.sum()})")
    print(f"  pred slope  = {slope_pred:+.3f}  (N={mask_pred.sum()})")
    print(f"  theoretical = -0.250")

    # Persist
    out = {
        'expression': expr,
        'tier3_N': int(df.shape[0]),
        'n_binary': int((df['n_elem']==2).sum()),
        'n_ternary': int((df['n_elem']==3).sum()),
        'n_quat_plus': int((df['n_elem']>=4).sum()),
        'finite_pred': int(finite.sum()),
        'physical_range_pass': int(phys_ok.sum()),
        'zero_shot_RMSE': rmse_all,
        'zero_shot_MAE':  mae_all,
        'zero_shot_R2':   r2_all,
        'splits': splits,
        'tier2_binary_RMSE_as_sanity': rmse(y2[m2], p2[m2]),
        'moss_slope_data_tier3': float(slope_data),
        'moss_slope_pred_tier3': float(slope_pred),
        'moss_slope_theory': -0.25,
    }
    Path('results/tier3_zeroshot.json').write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> results/tier3_zeroshot.json")


if __name__ == '__main__':
    main()
