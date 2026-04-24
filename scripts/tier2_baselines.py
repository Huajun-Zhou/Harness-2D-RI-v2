"""
Baseline CV table for Tier 2 (N=193) — mirrors Table 1 of the manuscript.

Runs 10-repeat 5-fold CV on:
  - Mean predictor
  - Lasso (α=0.1)
  - Ridge (α=1.0)
  - Z_X linear
  - Random Forest (100 trees)
  - bandgap-only linear (new Moss-law-implied baseline)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def cv_rmse_r2(model, X, y, rkf):
    rmse = -cross_val_score(model, X, y, cv=rkf, scoring='neg_root_mean_squared_error')
    r2 = cross_val_score(model, X, y, cv=rkf, scoring='r2')
    return rmse.mean(), rmse.std(), r2.mean(), r2.std()


def main():
    df = pd.read_csv('data/extended/tier2_binary.csv')
    cols = ['Z_M','Z_X','chi_M','chi_X','r_M','r_X','IE_M','EA_X','d_e',
            'd_MX','a_lat','t','bandgap_optb88',
            'abs_delta_chi','r_ratio','bond_polarity','Z_ratio','IE_EA_ratio']
    data = df[cols + ['cauchy_A']].dropna()
    X = data[cols].values
    y = data['cauchy_A'].values

    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    results = []

    def run(name, model, X_use):
        m, s, r2m, r2s = cv_rmse_r2(model, X_use, y, rkf)
        results.append({
            'Method': name,
            'RMSE': f"{m:.3f}±{s:.3f}",
            'R²':   f"{r2m:+.3f}±{r2s:.3f}",
            '#Var': X_use.shape[1],
        })
        print(f"  {name:30s}: RMSE={m:.3f}±{s:.3f}, R²={r2m:+.3f}±{r2s:.3f}")

    print(f"Tier 2 baselines on N={len(data)}:\n")

    run("Mean predictor", DummyRegressor(strategy='mean'), X)
    run("Lasso (α=0.1)",
        Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=0.1, max_iter=10000))]), X)
    run("Ridge (α=1.0)",
        Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]), X)
    run("Z_X linear (1 var)",
        LinearRegression(), data[['Z_X']].values)
    run("bandgap linear (1 var)",
        LinearRegression(), data[['bandgap_optb88']].values)
    run("Z_X + bandgap linear (2 var)",
        LinearRegression(), data[['Z_X', 'bandgap_optb88']].values)
    run("Random Forest (100)",
        RandomForestRegressor(n_estimators=100, random_state=42), X)
    run("Random Forest (500)",
        RandomForestRegressor(n_estimators=500, random_state=42), X)

    df_out = pd.DataFrame(results)
    df_out.to_csv('results/tier2_baselines_cv.csv', index=False)
    print(f"\nSaved → results/tier2_baselines_cv.csv")
    print(df_out.to_string(index=False))


if __name__ == '__main__':
    main()
