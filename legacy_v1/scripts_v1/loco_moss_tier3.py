"""
Tier-3 LOCO and Moss-exponent scan with frozen Eq.(1).
Output: results/loco_tier3.csv, results/moss_loco_tier3.json.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


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


def main():
    rep = json.loads(Path('results/discovery_report_20260420_165841.json').read_text())
    expr = rep['best_formula']
    print(f"Expression: {expr}")

    df = pd.read_csv('data/extended/tier3_binary_plus_ternary.csv')
    feats = ['Z_M','Z_X','chi_M','chi_X','r_M','r_X','IE_M','EA_X','d_e',
             'd_MX','a_lat','t','bandgap_optb88',
             'abs_delta_chi','r_ratio','bond_polarity','Z_ratio']
    data = df[feats + ['cauchy_A', 'anion', 'n_elem']].dropna().copy()
    print(f"Clean Tier 3: {len(data)}")

    X_all = np.column_stack([df['Z_M'].values, df['Z_ratio'].values,
                             df['d_e'].values, df['Z_X'].values,
                             df['bond_polarity'].values])
    pred_all = make_eval(expr)(X_all)
    df2 = df.copy()
    df2['_pred'] = pred_all

    # ── Moss scan ───────────────────────────────────────────────────
    mask = (df['bandgap_optb88'] > 0.2) & (df['cauchy_A'] > 1.02)
    slope_data, _ = np.polyfit(np.log(df.loc[mask, 'bandgap_optb88']),
                                np.log(df.loc[mask, 'cauchy_A']), 1)
    mp = mask & np.isfinite(pred_all) & (pred_all > 1.02)
    slope_pred, _ = np.polyfit(np.log(df.loc[mp, 'bandgap_optb88']),
                                np.log(pred_all[mp]), 1)
    print(f"Moss data slope:  {slope_data:+.3f}  N={mask.sum()}")
    print(f"Moss pred slope:  {slope_pred:+.3f}  N={mp.sum()}")

    # Moss on binary-only slice
    mb = mask & (df['n_elem']==2).values
    slope_d_bin, _ = np.polyfit(np.log(df.loc[mb,'bandgap_optb88']),
                                 np.log(df.loc[mb,'cauchy_A']), 1)
    mpb = mb & np.isfinite(pred_all) & (pred_all > 1.02)
    slope_p_bin, _ = np.polyfit(np.log(df.loc[mpb,'bandgap_optb88']),
                                 np.log(pred_all[mpb]), 1)
    print(f"Moss data (binary only): {slope_d_bin:+.3f}  N={mb.sum()}")
    print(f"Moss pred (binary only): {slope_p_bin:+.3f}  N={mpb.sum()}")

    # ── LOCO ────────────────────────────────────────────────────────
    rows = []
    for anion in sorted(data['anion'].unique()):
        test = data[data['anion'] == anion]
        train = data[data['anion'] != anion]
        if len(test) < 5 or len(train) < 50:
            continue

        rf = RandomForestRegressor(n_estimators=500, random_state=42)
        rf.fit(train[feats].values, train['cauchy_A'].values)
        rmse_rf = float(np.sqrt(mean_squared_error(
            test['cauchy_A'].values, rf.predict(test[feats].values))))
        rd = Pipeline([('sc', StandardScaler()), ('m', Ridge(1.0))])
        rd.fit(train[feats].values, train['cauchy_A'].values)
        rmse_rd = float(np.sqrt(mean_squared_error(
            test['cauchy_A'].values, rd.predict(test[feats].values))))

        mask_a = df2['anion'] == anion
        p = df2.loc[mask_a, '_pred'].values
        yt = df2.loc[mask_a, 'cauchy_A'].values
        mv = np.isfinite(p)
        rmse_h = float(np.sqrt(mean_squared_error(yt[mv], p[mv]))) \
            if mv.sum() >= 3 else float('nan')

        rows.append({'anion': anion, 'N': int(len(test)),
                     'Harness': rmse_h, 'RF': rmse_rf, 'Ridge': rmse_rd})

    res = pd.DataFrame(rows)
    res['Delta_vs_RF'] = res['Harness'] - res['RF']
    res['win'] = res['Harness'] < res['RF']
    print("\nLOCO Tier 3:")
    print(res.to_string(index=False))

    w = res['N'].values
    wavg_h = float((res['Harness'] * w).sum() / w.sum())
    wavg_rf = float((res['RF'] * w).sum() / w.sum())
    wavg_rd = float((res['Ridge'] * w).sum() / w.sum())
    print(f"\nWeighted RMSE: Harness={wavg_h:.3f}  RF={wavg_rf:.3f}  "
          f"Ridge={wavg_rd:.3f}")
    print(f"Harness wins: {int(res['win'].sum())}/{len(res)} anions")

    Path('results/loco_tier3.csv').write_text(res.to_csv(index=False))
    out = {
        'expression': expr,
        'tier3_N': int(len(df)),
        'moss_slope_data_all':     float(slope_data),
        'moss_slope_pred_all':     float(slope_pred),
        'moss_slope_data_binary':  float(slope_d_bin),
        'moss_slope_pred_binary':  float(slope_p_bin),
        'moss_slope_theory': -0.25,
        'loco_weighted_rmse_harness': wavg_h,
        'loco_weighted_rmse_rf':      wavg_rf,
        'loco_weighted_rmse_ridge':   wavg_rd,
        'loco_wins_harness': int(res['win'].sum()),
        'loco_n_anions':     len(res),
    }
    Path('results/moss_loco_tier3.json').write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> results/loco_tier3.csv, results/moss_loco_tier3.json")


if __name__ == '__main__':
    main()
