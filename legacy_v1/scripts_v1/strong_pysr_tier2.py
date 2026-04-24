"""
A4 — Budget-matched Strong-PySR baseline on Tier 2.

Runs vanilla PySR (no harness, no veto, no agents, no adaptive env, no memory)
with the SAME total PySR budget as the Harness Tier 2 headline run:
  10 rounds × 40 islands × 100 iterations  (≈ 40 000 island-iterations).

Repeats over 5 PySR seeds for statistical honesty. Same 80/20 split
(np.random.seed(42)) as the Harness run, same feature set in production
order, same operator vocabulary.

Measures (a) test RMSE / R², (b) Moss-motif recovery (does the expression
contain a Z_M, Z_X, or inverse/quartic-root structure consistent with
n^4 ∝ 1/E_g?), (c) stability on the full Tier 2 set (same criterion as
the stability veto: ≤ 5% of predictions outside [−20, 20]).

Results:
  results/strong_pysr_tier2_runs.csv      per-seed metrics + best expr
  results/strong_pysr_tier2_summary.csv   aggregate mean ± std
  results/strong_pysr_tier2_report.md     markdown summary
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

# ── Config ────────────────────────────────────────────────────────────
N_SEEDS       = 5
N_ROUNDS      = 10
NITER         = 100
NPOP          = 40
POPSIZE       = 33
MAXSIZE       = 25
PARSIMONY     = 0.05

BINARY_OPS = ["+", "-", "*", "/"]
# Mirror the production unary set (Julia-native to keep parity with Harness)
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

# Stability threshold (mirror ValidatorAgent)
PHYSICAL_RANGE_MAX = 20.0


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_tier2():
    """Production feature order: x0=Z_M, x1=Z_ratio, x2=d_e, x3=Z_X, x4=bond_polarity."""
    df = pd.read_csv('data/extended/tier2_binary.csv')
    X = np.column_stack([
        df['Z_M'].values,
        df['Z_ratio'].values,
        df['d_e'].values,
        df['Z_X'].values,
        df['bond_polarity'].values,
    ])
    y = df['cauchy_A'].values
    names = ['Z_M', 'Z_ratio', 'd_e', 'Z_X', 'bond_polarity']

    # Same 80/20 split as Harness production
    np.random.seed(42)
    idx = np.random.permutation(len(df))
    n_tr = int(0.8 * len(df))
    tr, te = idx[:n_tr], idx[n_tr:]
    return X[tr], y[tr], X[te], y[te], X, y, names


def detect_motifs(expr: str):
    """Cheap heuristic checks for physically-grounded structure.

    Accepts both x-indexed (x0..x4) and named (Z_M, Z_ratio, d_e, Z_X,
    bond_polarity) forms — production Harness runs use x-indexed.
    """
    s = expr.lower()
    # Normalise x-indexed names to match the named convention for motif-checking
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
    """Stability on full Tier 2 (same criterion as the training veto)."""
    try:
        yp = model.predict(X_full, index=idx)
    except Exception:
        return False, 1.0
    finite = np.isfinite(yp)
    if finite.sum() == 0:
        return False, 1.0
    viol_frac = float(np.mean((~finite) | (np.abs(yp) > PHYSICAL_RANGE_MAX)))
    return viol_frac <= 0.05, viol_frac


def run_seed(seed: int, X_tr, y_tr, X_te, y_te, X_full, y_full, names):
    """One seed: 10 iterative PySR rounds, keep best-by-train candidate."""
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
                print(f"  seed {seed} round {rnd}: no equations", flush=True)
                continue

            # Select the lowest-train-loss equation that is stable on full data
            best_idx, best_tr_rmse = None, float('inf')
            for ii in range(len(eqs)):
                try:
                    yp_tr = model.predict(X_tr, index=ii)
                    if not np.all(np.isfinite(yp_tr)):
                        continue
                    tr_r = rmse(y_tr, yp_tr)
                    stable, _ = stable_on_full(model, ii, X_full)
                    if stable and tr_r < best_tr_rmse:
                        best_tr_rmse = tr_r
                        best_idx = ii
                except Exception:
                    continue

            if best_idx is None:
                print(f"  seed {seed} round {rnd}: no stable equation", flush=True)
                continue

            yp_te = model.predict(X_te, index=best_idx)
            te_rmse = rmse(y_te, yp_te) if np.all(np.isfinite(yp_te)) else float('inf')
            r2      = r2_score(y_te, yp_te) if np.all(np.isfinite(yp_te)) else float('-inf')
            expr    = str(eqs.iloc[best_idx]['sympy_format'])
            cplx    = int(eqs.iloc[best_idx]['complexity'])
            stable, viol = stable_on_full(model, best_idx, X_full)

            print(f"  seed {seed} round {rnd}: "
                  f"tr={best_tr_rmse:.3f} te={te_rmse:.3f} "
                  f"cplx={cplx} stable={stable} ({time.time()-t0:.1f}s)",
                  flush=True)

            if best_tr_rmse < best['train_rmse']:
                best.update({
                    'expression': expr, 'complexity': cplx,
                    'train_rmse': best_tr_rmse, 'test_rmse': te_rmse,
                    'r2': r2, 'stable': stable, 'viol_frac': viol,
                })
        except Exception as e:
            print(f"  seed {seed} round {rnd}: ERROR {e}", flush=True)
            continue

    best.update(detect_motifs(best['expression'] or ''))
    return best


def main():
    X_tr, y_tr, X_te, y_te, X_full, y_full, names = load_tier2()
    print(f"Tier 2 split: train={len(y_tr)}  test={len(y_te)}  "
          f"features={names}\n")

    out_dir = Path('results')
    t_all = time.time()
    records = []
    for seed in range(N_SEEDS):
        print(f"── Seed {seed} " + "─" * 40)
        records.append(run_seed(seed, X_tr, y_tr, X_te, y_te, X_full, y_full, names))

    df = pd.DataFrame(records)
    df.to_csv(out_dir / 'strong_pysr_tier2_runs.csv', index=False)

    # Aggregate
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
    agg.to_csv(out_dir / 'strong_pysr_tier2_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("Strong-PySR Tier 2 baseline (5 seeds × 10 rounds × 40 × 100):")
    print(df[['seed', 'complexity', 'train_rmse', 'test_rmse', 'r2',
              'stable', 'has_cbrt', 'has_inv', 'has_zratio']]
          .to_string(index=False))
    print(f"\nTotal wall time: {(time.time() - t_all) / 60:.1f} min\n")
    print("Aggregate:")
    print(agg.to_string(index=False))

    # ── Markdown report ──────────────────────────────────────────────────
    hstats = json.loads(Path('results/discovery_report_20260420_165841.json').read_text())
    h_train = hstats['metrics']['train_rmse']
    h_test  = hstats['metrics']['test_rmse']
    h_r2    = hstats['metrics']['r2']

    md = ["# A4 — Budget-matched Strong-PySR baseline on Tier 2\n\n"]
    md.append(f"Budget per seed: {N_ROUNDS} rounds × {NPOP} islands × "
              f"{NITER} iterations (same as Harness run).  Seeds: {N_SEEDS}.\n\n")

    md.append("## Head-to-head on the 20% test split\n\n")
    md.append("| method | train RMSE | test RMSE | R² | stable on full set |\n")
    md.append("|---|---|---|---|---|\n")
    md.append(f"| Harness-2D-RI (post-veto headline) | {h_train:.3f} | "
              f"{h_test:.3f} | {h_r2:+.3f} | ✓ |\n")
    md.append(f"| Strong-PySR (mean ± std, {N_SEEDS} seeds) | "
              f"{agg['train_rmse_mean'].iloc[0]:.3f} ± {agg['train_rmse_std'].iloc[0]:.3f} | "
              f"{agg['test_rmse_mean'].iloc[0]:.3f} ± {agg['test_rmse_std'].iloc[0]:.3f} | "
              f"{agg['r2_mean'].iloc[0]:+.3f} ± {agg['r2_std'].iloc[0]:.3f} | "
              f"{agg['stable_frac'].iloc[0] * 100:.0f}% of seeds |\n\n")

    md.append("## Per-seed best expressions\n\n")
    for _, r in df.iterrows():
        md.append(f"- **seed {int(r['seed'])}** (cplx {r['complexity']}, "
                  f"test RMSE {r['test_rmse']:.3f}, R² {r['r2']:+.3f}, "
                  f"{'stable' if r['stable'] else 'UNSTABLE'}):  \n"
                  f"  `{r['expression']}`\n")

    md.append("\n## Motif-recovery rates across seeds\n\n")
    md.append("| motif | Strong-PySR (5 seeds) | Harness-2D-RI |\n|---|---|---|\n")
    # Check whether the Harness headline has each motif
    h_expr = hstats['best_formula']
    h_motifs = detect_motifs(h_expr)
    for tag, label in [
        ('cbrt_frac',    'cbrt (cube-root)'),
        ('inv_frac',     'inverse (1/x)'),
        ('zratio_frac',  'Z_M/Z_X ratio'),
        ('de_frac',      'd-electron count'),
        ('quartic_frac', 'quartic root (Moss-law)'),
    ]:
        strong_frac = agg[tag].iloc[0]
        short = tag.replace('_frac', '')
        h_has = h_motifs.get(f'has_{short}', 0)
        md.append(f"| {label} | {strong_frac * 100:.0f}% ({int(strong_frac * N_SEEDS)}/{N_SEEDS}) | "
                  f"{'✓' if h_has else '✗'} |\n")

    md.append("\n## Interpretation\n\n")
    diff = agg['test_rmse_mean'].iloc[0] - h_test
    if diff > 0.05:
        verdict = (f"Harness beats Strong-PySR by Δ_test_RMSE = {diff:+.3f} "
                   f"({diff / h_test * 100:.1f}% relative).")
    elif diff < -0.05:
        verdict = (f"Strong-PySR beats Harness by Δ_test_RMSE = {-diff:+.3f} "
                   f"— the governance layer is costing accuracy here.")
    else:
        verdict = (f"The two methods are within noise (Δ = {diff:+.3f}, "
                   f"σ ≈ {agg['test_rmse_std'].iloc[0]:.3f}). Harness's value must "
                   f"come from stability or motif-recovery, not raw accuracy.")
    md.append(verdict + "\n\n")
    md.append(
        f"Seed-to-seed reproducibility (Strong-PySR test RMSE std = "
        f"{agg['test_rmse_std'].iloc[0]:.3f}) quantifies how much the vanilla "
        f"search varies without governance; Harness's deterministic agent voting "
        f"eliminates this variance by construction.\n"
    )

    Path('results/strong_pysr_tier2_report.md').write_text(''.join(md))
    print("\nSaved → results/strong_pysr_tier2_{runs,summary}.csv, "
          "strong_pysr_tier2_report.md")


if __name__ == '__main__':
    main()
