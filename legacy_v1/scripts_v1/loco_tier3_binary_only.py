"""
LOCO on the *binary-only* slice of Tier 3 (N=222), to isolate whether the
regression of the Harness advantage is due to (a) ternaries specifically
or (b) the extra binaries that JARVIS-DFT2D has but Tier 2 filtered out.
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
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
    df = pd.read_csv('data/extended/tier3_binary_plus_ternary.csv')
    df = df[df['n_elem'] == 2].reset_index(drop=True)
    print(f"Tier-3 binary-only: {len(df)} entries")

    feats = ['Z_M','Z_X','chi_M','chi_X','r_M','r_X','IE_M','EA_X','d_e',
             'd_MX','a_lat','t','bandgap_optb88',
             'abs_delta_chi','r_ratio','bond_polarity','Z_ratio']
    data = df[feats + ['cauchy_A', 'anion']].dropna().copy()
    X_all = np.column_stack([df['Z_M'].values, df['Z_ratio'].values,
                             df['d_e'].values, df['Z_X'].values,
                             df['bond_polarity'].values])
    pred_all = make_eval(expr)(X_all)
    df2 = df.copy(); df2['_pred'] = pred_all

    rows = []
    for anion in sorted(data['anion'].unique()):
        test = data[data['anion'] == anion]
        train = data[data['anion'] != anion]
        if len(test) < 3 or len(train) < 30:
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
    print(res.to_string(index=False))
    w = res['N'].values
    wavg_h = float((res['Harness'] * w).sum() / w.sum())
    wavg_rf = float((res['RF'] * w).sum() / w.sum())
    wavg_rd = float((res['Ridge'] * w).sum() / w.sum())
    print(f"\nWeighted RMSE: Harness={wavg_h:.3f}  RF={wavg_rf:.3f}  Ridge={wavg_rd:.3f}")
    print(f"Harness wins: {int(res['win'].sum())}/{len(res)} anions")
    res.to_csv('results/loco_tier3_binary_only.csv', index=False)
    Path('results/loco_tier3_binary_only.json').write_text(json.dumps({
        'N_binary_only': int(len(df)),
        'expression': expr,
        'loco_weighted_rmse_harness': wavg_h,
        'loco_weighted_rmse_rf':      wavg_rf,
        'loco_weighted_rmse_ridge':   wavg_rd,
        'loco_wins_harness': int(res['win'].sum()),
        'loco_n_anions':     len(res),
    }, indent=2))


if __name__ == '__main__':
    main()
