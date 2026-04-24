"""
Screen Pareto-front expressions for numerical stability on TMD subspace
and overall well-behavedness. Features:
  x0 = Z_M, x1 = Z_ratio, x2 = Z_X, x3 = bond_polarity, x4 = d_e

Stability criteria:
  (a) Predictions for the full N=193 dataset must be finite and in [0, 50]
  (b) Predictions on the canonical TMD test points (MoS2, WS2, MoSe2,
      WSe2, MoTe2) must be in [1, 10]  (physical refractive index range)
  (c) Report per-anion-family RMSE stratification

Literature experimental reference (monolayer, ellipsometry):
  MoS2 4.28, WS2 4.02, MoSe2 4.68, WSe2 4.53, MoTe2 4.90
"""

import math
import numpy as np
import pandas as pd


# Literature exp. n_static for canonical TMDs (monolayer, ellipsometry/reflect.)
LIT_TMDS = {
    'MoS2':  4.28,
    'WS2':   4.02,
    'MoSe2': 4.68,
    'WSe2':  4.53,
    'MoTe2': 4.90,
}


def make_evaluator(expr_str):
    """
    Compile a PySR-style expression string into a callable f(x0,x1,x2,x3,x4).
    Supports: cube, cbrt, square, exp, log, sqrt.
    """
    py = (expr_str
          .replace('cube(', 'CUBE(').replace('cbrt(', 'CBRT(')
          .replace('square(', 'SQUARE(').replace('exp(', 'EXP(')
          .replace('log(', 'LOG(').replace('sqrt(', 'SQRT('))
    py = (py.replace('CUBE(', 'cube(').replace('CBRT(', 'cbrt(')
            .replace('SQUARE(', 'square(').replace('EXP(', 'exp(')
            .replace('LOG(', 'log(').replace('SQRT(', 'sqrt('))

    def f(x0, x1, x2, x3, x4):
        ns = {
            'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
            'cube': lambda v: v**3, 'cbrt': np.cbrt,
            'square': lambda v: v**2, 'exp': np.exp, 'log': np.log,
            'sqrt': np.sqrt,
        }
        with np.errstate(over='ignore', invalid='ignore'):
            return eval(py, {'__builtins__': {}}, ns)
    return f


def classify_anion_family(anion):
    if anion in ['O', 'S', 'Se', 'Te']:
        return 'chalcogen'
    if anion in ['F', 'Cl', 'Br', 'I']:
        return 'halogen'
    return 'pnictogen'


def main():
    df = pd.read_csv('data/extended/tier2_binary.csv')
    pareto = pd.read_csv('results/tier2_pareto_top10.csv')

    # Feature alias used by PySR search
    X = np.column_stack([
        df['Z_M'].values,           # x0
        df['Z_ratio'].values,       # x1
        df['Z_X'].values,           # x2
        df['bond_polarity'].values, # x3
        df['d_e'].values,           # x4
    ])
    y = df['cauchy_A'].values
    anion = df['anion'].values
    family = np.array([classify_anion_family(a) for a in anion])

    # Build TMD test point table
    tmd_rows = {}
    for mat, n_exp in LIT_TMDS.items():
        metal = ''.join([c for c in mat if c.isalpha()])[:2] if mat[1].islower() else mat[0]
        # Simpler: just strip trailing digit and known anion
        for an in ['S', 'Se', 'Te']:
            if mat.endswith(an + '2') or mat.endswith(an):
                metal = mat.replace('2', '').replace(an, '')
                anion_code = an
                break
        m = df[(df['metal'] == metal) & (df['anion'] == anion_code) &
               (df['stoichiometry'] == 'MX2')]
        if len(m) > 0:
            tmd_rows[mat] = {
                'n_exp': n_exp,
                'x': (m.iloc[0]['Z_M'], m.iloc[0]['Z_ratio'],
                      m.iloc[0]['Z_X'], m.iloc[0]['bond_polarity'],
                      m.iloc[0]['d_e']),
            }

    print(f"Evaluating {len(pareto)} Pareto candidates on "
          f"N={len(df)} dataset and {len(tmd_rows)} TMD reference points")
    print("=" * 100)

    results = []
    for _, row in pareto.iterrows():
        expr = row['expression']
        f = make_evaluator(expr)

        # Full-dataset prediction
        with np.errstate(all='ignore'):
            y_pred = f(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
        y_pred = np.asarray(y_pred, dtype=float)

        n_finite = np.isfinite(y_pred).sum()
        n_inrange = ((y_pred >= 0) & (y_pred <= 50) & np.isfinite(y_pred)).sum()
        if n_finite > 0:
            max_abs = np.nanmax(np.abs(y_pred[np.isfinite(y_pred)]))
        else:
            max_abs = float('inf')

        # TMD-level check
        tmd_preds = {}
        tmd_ok = True
        tmd_mape = []
        for mat, d in tmd_rows.items():
            with np.errstate(all='ignore'):
                p = float(f(*d['x']))
            tmd_preds[mat] = p
            if not (1.0 <= p <= 10.0 and np.isfinite(p)):
                tmd_ok = False
            tmd_mape.append(abs(p - d['n_exp']) / d['n_exp'] * 100)

        # Per-family RMSE
        family_rmse = {}
        for fam in ['chalcogen', 'halogen', 'pnictogen']:
            mask = (family == fam) & np.isfinite(y_pred)
            if mask.sum() > 0:
                family_rmse[fam] = float(np.sqrt(
                    ((y_pred[mask] - y[mask])**2).mean()))
            else:
                family_rmse[fam] = float('nan')

        results.append({
            'rank': row['rank'],
            'cplx': row['complexity'],
            'test_rmse': row['test_rmse'],
            'r2': row['r2'],
            'n_finite': n_finite,
            'n_inrange': n_inrange,
            'max_abs': max_abs,
            'tmd_stable': tmd_ok,
            'tmd_mape': np.mean(tmd_mape) if tmd_mape else float('nan'),
            'rmse_chalcogen': family_rmse.get('chalcogen'),
            'rmse_halogen':   family_rmse.get('halogen'),
            'rmse_pnictogen': family_rmse.get('pnictogen'),
            'tmd_preds': tmd_preds,
            'expression': expr,
        })

    # Tabulate
    display_cols = ['rank', 'cplx', 'test_rmse', 'r2',
                    'n_inrange', 'tmd_stable', 'tmd_mape',
                    'rmse_chalcogen', 'rmse_halogen', 'rmse_pnictogen']
    summary = pd.DataFrame(results)[display_cols]
    print("\nSTABILITY SCREEN:")
    print(summary.to_string(index=False))

    print("\nTMD-REFERENCE PREDICTIONS:")
    print(f"{'rank':<5}{'cplx':<5}", end='')
    for mat in LIT_TMDS:
        print(f"{mat:>10}", end='')
    print(f"  {'tmd_ok':>8}")
    print('-' * 85)
    for r in results:
        print(f"{r['rank']:<5}{r['cplx']:<5}", end='')
        for mat in LIT_TMDS:
            p = r['tmd_preds'].get(mat, float('nan'))
            if abs(p) > 1e6 or not np.isfinite(p):
                print(f"{'EXPLODE':>10}", end='')
            else:
                print(f"{p:>10.2f}", end='')
        print(f"  {'✓' if r['tmd_stable'] else '✗':>8}")
    print(f"{'exp':<5}{'':<5}", end='')
    for mat, v in LIT_TMDS.items():
        print(f"{v:>10.2f}", end='')
    print()

    # Save
    pd.DataFrame(results)[display_cols + ['expression']].to_csv(
        'results/b2_pareto_stability.csv', index=False)
    print("\nSaved → results/b2_pareto_stability.csv")

    # Recommendation
    stable = [r for r in results if r['tmd_stable']
              and r['n_inrange'] >= 0.95 * len(df)]
    if stable:
        stable.sort(key=lambda r: r['test_rmse'])
        best = stable[0]
        print("\n" + "=" * 60)
        print("RECOMMENDED HEADLINE EXPRESSION:")
        print(f"  Rank {best['rank']}  (complexity {best['cplx']}, "
              f"test RMSE {best['test_rmse']:.3f}, R²={best['r2']:+.3f})")
        print(f"  TMD MAPE vs experiment: {best['tmd_mape']:.1f}%")
        print(f"  Expression: {best['expression']}")
        print("=" * 60)
    else:
        print("\nNo Pareto candidate passes TMD-stability screen!")


if __name__ == '__main__':
    main()
