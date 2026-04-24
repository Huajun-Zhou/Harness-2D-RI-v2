"""
Budget-matched Strong-PySR baseline on Tier 3.

Mirrors scripts/strong_pysr_tier2.py but uses Tier 3 (N=532).
Reduced budget (5 seeds x 5 rounds x 40 islands x 100 iterations) so the
total wall clock is tractable — still a fair baseline since the Harness
Tier-3 headline re-training is optional.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

logging.getLogger("pysr").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_SEEDS       = 5
N_ROUNDS      = 5
NITER         = 100
NPOP          = 40
POPSIZE       = 33
MAXSIZE       = 25
PARSIMONY     = 0.05

BINARY_OPS = ["+", "-", "*", "/"]
UNARY_OPS = [
    "exp", "log", "sqrt", "square",
    "safe_inv(x::T) where T = x == zero(T) ? T(1e6) : one(T)/x",
    "safe_cbrt(x::T) where T = sign(x) * abs(x)^(T(1)/T(3))",
    "safe_cube(x::T) where T = x * x * x",
]
SYMPY_MAPPINGS = {
    "safe_inv":  lambda x: 1 / x,
    "safe_cbrt": lambda x: x ** (1 / 3),
    "safe_cube": lambda x: x ** 3,
}
PHYSICAL_RANGE_MAX = 20.0


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_tier3():
    df = pd.read_csv('data/extended/tier3_binary_plus_ternary.csv')
    X = np.column_stack([
        df['Z_M'].values, df['Z_ratio'].values,
        df['d_e'].values, df['Z_X'].values,
        df['bond_polarity'].values,
    ])
    y = df['cauchy_A'].values
    names = ['Z_M', 'Z_ratio', 'd_e', 'Z_X', 'bond_polarity']
    np.random.seed(42)
    idx = np.random.permutation(len(df))
    n_tr = int(0.8 * len(df))
    tr, te = idx[:n_tr], idx[n_tr:]
    return X[tr], y[tr], X[te], y[te], X, y, names


def detect_motifs(expr: str):
    s = expr.lower()
    s = (s.replace('x0', 'z_m').replace('x1', 'z_ratio')
           .replace('x2', 'd_e').replace('x3', 'z_x')
           .replace('x4', 'bond_polarity'))
    return {
        'has_inv':      int('safe_inv' in s or '/(' in s or '/z' in s or '/ (' in s),
        'has_cbrt':     int('safe_cbrt' in s or 'cbrt' in s),
        'has_quartic':  int('sqrt(sqrt' in s or '**0.25' in s),
        'has_zm':       int('z_m' in s),
        'has_zx':       int('z_x' in s),
        'has_zratio':   int('z_ratio' in s or ('z_m' in s and 'z_x' in s)),
        'has_de':       int('d_e' in s),
    }


def stable_on_full(model, idx, X_full):
    try:
        yp = model.predict(X_full, index=idx)
    except Exception:
        return False, 1.0
    finite = np.isfinite(yp)
    if finite.sum() == 0:
        return False, 1.0
    viol_frac = float(np.mean((~finite) | (np.abs(yp) > PHYSICAL_RANGE_MAX)))
    return viol_frac <= 0.05, viol_frac


def run_seed(seed, X_tr, y_tr, X_te, y_te, X_full, y_full, names):
    from pysr import PySRRegressor
    best = {
        'seed': seed, 'expression': None, 'complexity': None,
        'train_rmse': float('inf'), 'test_rmse': float('inf'),
        'r2': float('-inf'), 'stable': False, 'viol_frac': 1.0,
    }
    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        try:
            model = PySRRegressor(
                niterations=NITER, populations=NPOP, population_size=POPSIZE,
                maxsize=MAXSIZE, parsimony=PARSIMONY,
                binary_operators=BINARY_OPS, unary_operators=UNARY_OPS,
                extra_sympy_mappings=SYMPY_MAPPINGS,
                random_state=seed * 100 + rnd,
                deterministic=False, verbosity=0, progress=False,
                temp_equation_file=True,
            )
            model.fit(X_tr, y_tr, variable_names=names)
            eqs = model.equations_
            if eqs is None or len(eqs) == 0:
                print(f"  seed {seed} rnd {rnd}: no eqs", flush=True); continue

            best_idx, best_tr = None, float('inf')
            for ii in range(len(eqs)):
                try:
                    yp_tr = model.predict(X_tr, index=ii)
                    if not np.all(np.isfinite(yp_tr)): continue
                    tr_r = rmse(y_tr, yp_tr)
                    stable, _ = stable_on_full(model, ii, X_full)
                    if stable and tr_r < best_tr:
                        best_tr, best_idx = tr_r, ii
                except Exception:
                    continue
            if best_idx is None:
                print(f"  seed {seed} rnd {rnd}: no stable eq", flush=True); continue

            yp_te = model.predict(X_te, index=best_idx)
            te_rmse = rmse(y_te, yp_te) if np.all(np.isfinite(yp_te)) else float('inf')
            r2      = r2_score(y_te, yp_te) if np.all(np.isfinite(yp_te)) else float('-inf')
            expr    = str(eqs.iloc[best_idx]['sympy_format'])
            cplx    = int(eqs.iloc[best_idx]['complexity'])
            stable, viol = stable_on_full(model, best_idx, X_full)
            print(f"  seed {seed} rnd {rnd}: tr={best_tr:.3f} te={te_rmse:.3f} "
                  f"cplx={cplx} stable={stable} ({time.time()-t0:.1f}s)", flush=True)
            if best_tr < best['train_rmse']:
                best.update({'expression': expr, 'complexity': cplx,
                             'train_rmse': best_tr, 'test_rmse': te_rmse,
                             'r2': r2, 'stable': stable, 'viol_frac': viol})
        except Exception as e:
            print(f"  seed {seed} rnd {rnd}: ERROR {e}", flush=True); continue
    best.update(detect_motifs(best['expression'] or ''))
    return best


def main():
    X_tr, y_tr, X_te, y_te, X_full, y_full, names = load_tier3()
    print(f"Tier 3 split: train={len(y_tr)}  test={len(y_te)}  features={names}\n")
    t_all = time.time()
    records = []
    for seed in range(N_SEEDS):
        print(f"-- Seed {seed} " + "-" * 40)
        records.append(run_seed(seed, X_tr, y_tr, X_te, y_te, X_full, y_full, names))

    df = pd.DataFrame(records)
    out_dir = Path('results')
    df.to_csv(out_dir / 'strong_pysr_tier3_runs.csv', index=False)
    agg = pd.DataFrame([{
        'n_seeds':         len(df),
        'train_rmse_mean': df['train_rmse'].mean(),
        'train_rmse_std':  df['train_rmse'].std(ddof=1),
        'test_rmse_mean':  df['test_rmse'].mean(),
        'test_rmse_std':   df['test_rmse'].std(ddof=1),
        'r2_mean':         df['r2'].mean(),
        'r2_std':          df['r2'].std(ddof=1),
        'stable_frac':     df['stable'].mean(),
        'cbrt_frac':       df['has_cbrt'].mean(),
        'inv_frac':        df['has_inv'].mean(),
        'zratio_frac':     df['has_zratio'].mean(),
        'de_frac':         df['has_de'].mean(),
        'quartic_frac':    df['has_quartic'].mean(),
    }])
    agg.to_csv(out_dir / 'strong_pysr_tier3_summary.csv', index=False)
    print("\n", df[['seed','complexity','train_rmse','test_rmse','r2','stable',
                     'has_cbrt','has_inv','has_zratio']].to_string(index=False))
    print(f"\nTotal wall time: {(time.time()-t_all)/60:.1f} min\n")
    print("Aggregate:\n", agg.to_string(index=False))

    # Zero-shot Harness on same test split for head-to-head on Tier 3
    import math
    def _safe_exp(v):
        try: return math.exp(v) if v < 700 else float('inf')
        except: return float('inf')
    def _safe_log(v):
        try: return math.log(v) if v > 0 else float('nan')
        except: return float('nan')
    expr_h = json.loads(Path('results/discovery_report_20260420_165841.json').read_text())['best_formula']
    def feval(X):
        out = []
        for row in X:
            ns = {f'x{i}': float(row[i]) for i in range(len(row))}
            ns.update({'exp': _safe_exp, 'log': _safe_log, 'sqrt': math.sqrt,
                       'square': lambda v: v**2, 'cube': lambda v: v**3,
                       'cbrt': lambda v: v**(1/3) if v>=0 else -((-v)**(1/3)),
                       'inv': lambda v: 1/v if v != 0 else float('inf')})
            try: out.append(float(eval(expr_h, {'__builtins__': {}}, ns)))
            except: out.append(float('nan'))
        return np.asarray(out)
    yp_h = feval(X_te)
    mh = np.isfinite(yp_h)
    h_te_rmse = rmse(y_te[mh], yp_h[mh])
    h_r2 = r2_score(y_te[mh], yp_h[mh])

    md = ["# Tier-3 Strong-PySR baseline vs frozen Harness Eq.(1)\n\n",
          f"Budget per seed: {N_ROUNDS} rounds x {NPOP} islands x {NITER} iterations. "
          f"Seeds: {N_SEEDS}. Split: 80/20 (seed=42).\n\n",
          "## Head-to-head on Tier-3 test split\n\n",
          "| method | test RMSE | R² | stable | notes |\n",
          "|---|---|---|---|---|\n",
          f"| Frozen Harness Eq.(1) (zero-shot) | {h_te_rmse:.3f} | {h_r2:+.3f} | ✓ | no retraining |\n",
          f"| Strong-PySR mean ± std ({N_SEEDS} seeds) | "
          f"{agg['test_rmse_mean'].iloc[0]:.3f} ± {agg['test_rmse_std'].iloc[0]:.3f} | "
          f"{agg['r2_mean'].iloc[0]:+.3f} ± {agg['r2_std'].iloc[0]:.3f} | "
          f"{agg['stable_frac'].iloc[0]*100:.0f}% | retrained on Tier 3 |\n\n",
          "## Per-seed best expressions\n\n"]
    for _, r in df.iterrows():
        md.append(f"- seed {int(r['seed'])}: `{r['expression']}`  \n"
                  f"  cplx={r['complexity']}  tr={r['train_rmse']:.3f}  "
                  f"te={r['test_rmse']:.3f}  stable={bool(r['stable'])}\n")
    Path('results/strong_pysr_tier3_report.md').write_text(''.join(md))
    print("\nSaved -> results/strong_pysr_tier3_runs.csv / summary.csv / report.md")


if __name__ == '__main__':
    main()
