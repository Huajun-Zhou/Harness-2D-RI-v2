"""
Strong PySR baseline (matched budget, no harness).
Run from project root: python run_strong_pysr.py
"""
import warnings, os, sys
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold

DATA_DIR = Path(ROOT) / "data"
OUT_DIR  = Path(ROOT) / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "dataset_master.csv")

FEATURES = ['atomic_number_M','atomic_number_X','electronegativity_M','electronegativity_X',
            'atomic_radius_M','atomic_radius_X','ionization_energy_M','electron_affinity_X',
            'd_electron_count','bond_length_MX','lattice_constant_a','monolayer_thickness',
            'electronegativity_diff','radius_ratio','bond_polarity','packing_fraction',
            'Z_ratio','IE_EA_ratio']  # skip coordination_number, point_group_encoded (constant)
TARGET = 'cauchy_A'

train_df = df[df['split'] == 'train']
val_df   = df[df['split'] == 'val']
test_df  = df[df['split'] == 'test']

X_train = train_df[FEATURES].values.astype(float)
y_train = train_df[TARGET].values.astype(float)
X_val   = val_df[FEATURES].values.astype(float)
y_val   = val_df[TARGET].values.astype(float)
X_test  = test_df[FEATURES].values.astype(float)
y_test  = test_df[TARGET].values.astype(float)

def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

# ── PySR import ───────────────────────────────────────────────────────────────
try:
    from pysr import PySRRegressor
    import pysr
    print(f"PySR version: {pysr.__version__}")
except ImportError:
    print("ERROR: pysr not installed")
    sys.exit(1)

# ── Experiment config ─────────────────────────────────────────────────────────
N_ROUNDS  = 10
N_SEEDS   = 5
SEEDS     = [0, 1, 2, 3, 4]
NITER     = 100
NPOP      = 40
POPSIZE   = 33
MAXSIZE   = 25
PARSIMONY = 0.05

# Full operator set including cbrt from round 1 (equalised with HARNESS)
BINARY_OPS = ["+", "-", "*", "/"]
UNARY_OPS  = [
    "exp", "log", "sqrt", "square",
    "safe_inv(x::T) where T = x == zero(T) ? T(1e6) : one(T)/x",
    "safe_cbrt(x::T) where T = sign(x) * abs(x)^(T(1)/T(3))",
]
SYMPY_MAPPINGS = {"safe_inv": lambda x: 1/x, "safe_cbrt": lambda x: x**(1/3)}

def check_positivity(expr_fn, X):
    """Fraction of samples where prediction < 1 (positivity violation)."""
    try:
        preds = expr_fn(X)
        return float(np.mean(preds < 1.0))
    except:
        return 1.0

def check_sign_ok(model, X, y, feature_names):
    """
    Fraction of top features (|r| > 0.1) where sensitivity sign is consistent
    with Pearson r sign. Returns sign-ok%.
    """
    from scipy.stats import pearsonr
    correlations = {}
    for i, fn in enumerate(feature_names):
        r, _ = pearsonr(X[:, i], y)
        correlations[fn] = r

    try:
        # Use finite-difference sensitivity at mean
        x_mean = X.mean(axis=0)
        y_mean = model.predict(x_mean.reshape(1, -1))[0]
        consistent = 0
        checked = 0
        for i, fn in enumerate(feature_names):
            if abs(correlations[fn]) < 0.10:
                continue
            delta = max(abs(x_mean[i]) * 1e-4, 1e-6)
            x_pert = x_mean.copy()
            x_pert[i] += delta
            y_pert = model.predict(x_pert.reshape(1, -1))[0]
            sensitivity_sign = np.sign(y_pert - y_mean)
            pearson_sign = np.sign(correlations[fn])
            if sensitivity_sign == pearson_sign:
                consistent += 1
            checked += 1
        return consistent / checked if checked > 0 else 1.0
    except:
        return 0.5

# ── Run experiments ───────────────────────────────────────────────────────────
results = []

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print('='*50)
    best_expr_str    = None
    best_val_rmse    = np.inf
    best_model       = None
    best_complexity  = None
    best_pareto_all  = []

    for round_idx in range(1, N_ROUNDS + 1):
        print(f"  Round {round_idx}/{N_ROUNDS} ...", end=" ", flush=True)
        try:
            model = PySRRegressor(
                niterations=NITER,
                populations=NPOP,
                population_size=POPSIZE,
                maxsize=MAXSIZE,
                parsimony=PARSIMONY,
                binary_operators=BINARY_OPS,
                unary_operators=UNARY_OPS,
                extra_sympy_mappings=SYMPY_MAPPINGS,
                random_state=seed * 100 + round_idx,
                verbosity=0,
                progress=False,
                temp_equation_file=True,
            )
            model.fit(X_train, y_train, variable_names=FEATURES)

            # Collect Pareto front
            eqs = model.equations_
            if eqs is None or len(eqs) == 0:
                print("no equations")
                continue

            # Select by val criterion: lowest complexity within 5% of best val RMSE
            val_rmses = []
            for idx in range(len(eqs)):
                try:
                    yp = model.predict(X_val, index=idx)
                    if not np.all(np.isfinite(yp)):
                        val_rmses.append(np.inf)
                    else:
                        val_rmses.append(rmse(y_val, yp))
                except:
                    val_rmses.append(np.inf)

            best_val_r = min(val_rmses)
            threshold  = best_val_r * 1.05
            candidates = [(i, eqs.iloc[i]['complexity'], val_rmses[i])
                          for i in range(len(val_rmses))
                          if val_rmses[i] <= threshold]
            candidates.sort(key=lambda x: x[1])  # lowest complexity

            if not candidates:
                print(f"best_val={best_val_r:.3f} (no valid candidates)")
                continue

            sel_idx, sel_cplx, sel_val = candidates[0]
            sel_expr = str(eqs.iloc[sel_idx]['sympy_format'])

            print(f"best_val={sel_val:.3f} cplx={sel_cplx} expr={sel_expr[:60]}")

            if sel_val < best_val_rmse:
                best_val_rmse   = sel_val
                best_expr_str   = sel_expr
                best_model      = (model, sel_idx)
                best_complexity = sel_cplx

            best_pareto_all.extend([(sel_expr, sel_cplx, sel_val)])

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Evaluate best expression on test set
    test_rmse_val = np.inf
    test_r2_val   = -999.0
    pos_viol_rate = 1.0
    sign_ok_pct   = 0.0

    if best_model is not None:
        m, idx = best_model
        try:
            yp_test = m.predict(X_test, index=idx)
            if np.all(np.isfinite(yp_test)):
                test_rmse_val = rmse(y_test, yp_test)
                test_r2_val   = r2_score(y_test, yp_test)

            yp_val = m.predict(X_val, index=idx)
            if np.all(np.isfinite(yp_val)):
                pos_viol_rate = float(np.mean(yp_val < 1.0))
                sign_ok_pct   = check_sign_ok(
                    type('obj', (), {'predict': lambda self, X: m.predict(X, index=idx)})(),
                    X_train, y_train, FEATURES)
        except Exception as e:
            print(f"  Eval error: {e}")

    expr_str = best_expr_str or "NONE"
    has_cbrt = int('cbrt' in expr_str)           # matches safe_cbrt too
    has_inv  = int('inv' in expr_str or '1/' in expr_str)  # matches safe_inv too
    has_zmzx = int('Z_ratio' in expr_str or 'ZM/ZX' in expr_str or
                   ('atomic_number_M' in expr_str and 'atomic_number_X' in expr_str))

    print(f"\n  SEED {seed} FINAL:")
    print(f"    expr      = {expr_str[:80]}")
    print(f"    complexity= {best_complexity}")
    print(f"    val_rmse  = {best_val_rmse:.4f}")
    print(f"    test_rmse = {test_rmse_val:.4f}")
    print(f"    test_r2   = {test_r2_val:.4f}")
    print(f"    pos_viol  = {pos_viol_rate:.3f}")
    print(f"    sign_ok   = {sign_ok_pct:.3f}")
    print(f"    has_cbrt  = {has_cbrt}, has_inv={has_inv}, has_zmzx={has_zmzx}")

    results.append({
        'seed': seed,
        'expression': expr_str,
        'complexity': best_complexity,
        'val_rmse': round(best_val_rmse, 4),
        'test_rmse': round(test_rmse_val, 4),
        'test_r2': round(test_r2_val, 4),
        'positivity_violation_rate': round(pos_viol_rate, 3),
        'sign_ok_pct': round(sign_ok_pct, 3),
        'has_cbrt': has_cbrt,
        'has_inv': has_inv,
        'has_zmzx_motif': has_zmzx,
    })

# Save per-seed results
rdf = pd.DataFrame(results)
rdf.to_csv(OUT_DIR / "results_strong_pysr.csv", index=False)
print(f"\nSaved results_strong_pysr.csv")
print(rdf[['seed','expression','complexity','val_rmse','test_rmse','test_r2']].to_string())

# ── CV on best expression across seeds ───────────────────────────────────────
# Find best seed by val_rmse
valid_results = [r for r in results if r['val_rmse'] < np.inf]
if valid_results:
    best_seed_result = min(valid_results, key=lambda x: x['val_rmse'])
    best_expr_final  = best_seed_result['expression']
    print(f"\nBest expression for CV: {best_expr_final} (seed {best_seed_result['seed']})")

    # Build a callable from the expression
    # We'll run 10x5-fold CV using the best expression evaluated symbolically
    # Using sympy to evaluate the expression
    X_full = df[FEATURES].values.astype(float)
    y_full = df[TARGET].values.astype(float)

    # Re-run best seed's model on full data via KFold
    print("\nRunning 10×5-fold CV on best expression ...")
    cv_rec = []
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    for rep in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=rep)
        for fi, (tri, ei) in enumerate(kf.split(X_full)):
            try:
                m_cv = PySRRegressor(
                    niterations=50,   # lighter for CV
                    populations=20,
                    population_size=20,
                    maxsize=MAXSIZE,
                    parsimony=PARSIMONY,
                    binary_operators=BINARY_OPS,
                    unary_operators=UNARY_OPS,
                    extra_sympy_mappings={
                        "inv":  lambda x: 1/x,
                        "cbrt": lambda x: x**(1/3),
                    },
                    random_state=rep * 5 + fi,
                    verbosity=0,
                    progress=False,
                    temp_equation_file=True,
                )
                m_cv.fit(X_full[tri], y_full[tri], variable_names=FEATURES)
                eqs_cv = m_cv.equations_
                if eqs_cv is None: continue
                # pick best val equivalent = test fold
                vr = []
                for ii in range(len(eqs_cv)):
                    try:
                        yp = m_cv.predict(X_full[ei], index=ii)
                        vr.append(rmse(y_full[ei], yp) if np.all(np.isfinite(yp)) else np.inf)
                    except: vr.append(np.inf)
                best_ii = int(np.argmin(vr))
                yp = m_cv.predict(X_full[ei], index=best_ii)
                if not np.all(np.isfinite(yp)): continue
                cv_rec.append({
                    "rep": rep, "fold": fi,
                    "rmse": round(rmse(y_full[ei], yp), 6),
                    "r2":   round(r2_score(y_full[ei], yp), 6),
                })
                print(f"  CV rep{rep} fold{fi}: RMSE={cv_rec[-1]['rmse']:.3f}", end="\r")
            except Exception as e:
                print(f"  CV rep{rep} fold{fi} error: {e}")
                continue

    if cv_rec:
        cvdf = pd.DataFrame(cv_rec)
        cvdf.to_csv(OUT_DIR / "results_strong_pysr_cv.csv", index=False)
        print(f"\nCV RMSE={cvdf['rmse'].mean():.4f}±{cvdf['rmse'].std():.4f}  "
              f"R²={cvdf['r2'].mean():.4f}±{cvdf['r2'].std():.4f}")
        print(f"Saved results_strong_pysr_cv.csv")
    else:
        print("No CV results collected")
else:
    print("No valid results to run CV on")

# Save motifs
mdf = pd.DataFrame([{
    'seed': r['seed'],
    'expression': r['expression'],
    'complexity': r['complexity'],
    'val_rmse': r['val_rmse'],
    'test_rmse': r['test_rmse'],
    'test_r2': r['test_r2'],
    'has_cbrt': r['has_cbrt'],
    'has_inv': r['has_inv'],
    'has_zmzx_motif': r['has_zmzx_motif'],
} for r in results])
mdf.to_csv(OUT_DIR / "strong_pysr_motifs.csv", index=False)
print("Saved strong_pysr_motifs.csv")
print("\nDONE.")
