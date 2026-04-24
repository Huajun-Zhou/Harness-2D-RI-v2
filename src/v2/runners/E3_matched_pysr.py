"""
E3 — Matched PySR baselines on the E2 outer-LOCO folds.

This experiment answers the key reviewer question: when Harness and vanilla
PySR are evaluated under the SAME true outer-LOCO protocol, does Harness
still win?

Protocol per (outer_anion, seed):
  1. Reuse the exact E2 partitioning (outer_test + outer_pool + inner split).
  2. Run PySR directly — no agent council, no causal diagnostics, no
     knowledge memory, no adaptive scheduler. Config matches Harness's
     per-round PySR config exactly (40 populations × 33 pop_size × 100
     iterations × 10 rounds, same operators, same maxsize, same parsimony).
  3. Each of 10 rounds is one PySRRegressor.fit() on the inner train slice.
     Accumulate all candidates across rounds into a pooled Pareto front.
  4. From that pooled Pareto, derive four selected expressions per seed:
       V1 train-selected       : min train_rmse (v1's protocol)
       V2 screening-selected   : min inner_screen_rmse (fair vs Harness)
       V3 complexity-capped    : V2 restricted to complexity <= 11
       V4 stability-only       : V2 restricted to stable_frac >= 0.95
  5. Evaluate each selection on the held-out outer_test.

The output is a per-fold JSON that mirrors E2's fold_summary but with four
variants × n_seeds expressions/metrics instead of one Harness headline.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


log = logging.getLogger("v2.runners.E3")


def _install_file_log(run_dir: Path) -> logging.Handler:
    fh = logging.FileHandler(run_dir / "e3_run.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger("v2").addHandler(fh)
    return fh


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_pred)
    if not mask.any():
        return float("inf")
    return float(np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean()))


def _score_candidate(
    model, idx: int, X_tr, y_tr, X_sc, y_sc, X_full,
) -> dict[str, Any] | None:
    """Extract the ``idx``th candidate from a PySR model and score it on
    train / inner-screen / full-set stability."""
    try:
        yp_tr = model.predict(X_tr, index=idx)
        yp_sc = model.predict(X_sc, index=idx)
        yp_full = model.predict(X_full, index=idx)
    except Exception:
        return None
    if not np.all(np.isfinite(yp_tr)):
        return None
    # Stability = finite AND within physical range (same rule as Harness).
    finite = np.isfinite(yp_full)
    stable = finite & (np.abs(yp_full) <= 20.0)
    stable_frac = float(stable.mean())

    row = model.equations_.iloc[idx]
    return {
        "expression": str(row["equation"]),
        "complexity": int(row["complexity"]),
        "train_rmse": _rmse(y_tr, yp_tr),
        "inner_screen_rmse": _rmse(y_sc, yp_sc),
        "stable_frac": stable_frac,
    }


def _select(candidates: list[dict], *, policy: str) -> dict | None:
    """Pick one expression from a pool according to the policy."""
    pool = [c for c in candidates if c["train_rmse"] < float("inf")]
    if not pool:
        return None
    if policy == "V1_train":
        return min(pool, key=lambda c: c["train_rmse"])
    if policy == "V2_screen":
        return min(pool, key=lambda c: c["inner_screen_rmse"])
    if policy == "V3_cplx_capped":
        sub = [c for c in pool if c["complexity"] <= 11]
        if not sub:
            return None
        return min(sub, key=lambda c: c["inner_screen_rmse"])
    if policy == "V4_stability_only":
        sub = [c for c in pool if c["stable_frac"] >= 0.95]
        if not sub:
            return None
        return min(sub, key=lambda c: c["inner_screen_rmse"])
    raise ValueError(f"unknown policy {policy!r}")


# -- feature layout ---------------------------------------------------------

# Reuse E2's feature layout so the expressions can be evaluated on the same
# outer_test matrices without rebuilding the column order.
from src.v2.runners.E2_outer_loco import (
    E2_FEATURE_NAMES, _LONG_TO_SHORT, _build_feature_matrix, _project_features,
)
from src.v2.expr_eval import rmse_on
from src.v2.seeds import SeedManager


# -- main entry --------------------------------------------------------------


def run(config_path: Path, run_seed: int, run_dir: Path) -> dict[str, Any]:
    """
    Run one fold of E3: the same outer_anion partitioning as E2, but
    replace the Harness with plain PySR and derive four selection policies.

    The config carries:
      outer_anion: e.g. "O"
      n_seeds:     default 5
      n_rounds:    default 10
      pysr:        PySR configuration (same knobs E2 uses)
      stability_range: [lo, hi] — default [0, 20] to match Harness veto
    """
    file_handler = _install_file_log(run_dir)

    from pysr import PySRRegressor

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    outer_anion = cfg["outer_anion"]
    n_seeds = int(cfg.get("n_seeds", 5))
    n_rounds = int(cfg.get("n_rounds", 10))
    data_csv = cfg.get("data_csv", "data/extended/tier2_binary.csv")
    target = cfg.get("target", "cauchy_A")
    stable_lo, stable_hi = cfg.get("stability_range", [0.0, 20.0])
    pysr_cfg = cfg.get("pysr", {})

    sm = SeedManager(run_seed=run_seed)

    log.info("E3 fold outer_anion=%s  n_seeds=%d  n_rounds=%d",
             outer_anion, n_seeds, n_rounds)

    # -- 1. load + partition (identical to E2) --
    df = pd.read_csv(data_csv)
    if outer_anion not in set(df["anion"].unique()):
        raise ValueError(f"anion {outer_anion!r} not present in {data_csv}")
    outer_test_df = df[df["anion"] == outer_anion].reset_index(drop=True)
    outer_pool_df = df[df["anion"] != outer_anion].reset_index(drop=True)
    log.info("  outer_test N=%d  outer_pool N=%d",
             len(outer_test_df), len(outer_pool_df))

    # Build feature matrices. E3 gives vanilla PySR ALL 20 features —
    # there is no council to prune them. This is the fair comparison;
    # the council's feature-selection is itself one of the governance
    # features that would otherwise be invisible in the head-to-head.
    X_pool = _build_feature_matrix(outer_pool_df)
    y_pool = outer_pool_df[target].to_numpy(dtype=float)
    X_outer = _build_feature_matrix(outer_test_df)
    y_outer = outer_test_df[target].to_numpy(dtype=float)
    X_full = _build_feature_matrix(df)

    # SAME inner split seed as E2 — critical for fair comparison.
    inner_split_seed = sm.derive("e2_inner_split", anion=outer_anion)
    rng = np.random.default_rng(inner_split_seed)
    perm = rng.permutation(len(outer_pool_df))
    cut = int(0.8 * len(perm))
    tr_idx, sc_idx = perm[:cut], perm[cut:]
    X_train, y_train = X_pool[tr_idx], y_pool[tr_idx]
    X_screen, y_screen = X_pool[sc_idx], y_pool[sc_idx]
    log.info("  inner_train N=%d  inner_screen N=%d",
             len(X_train), len(X_screen))

    # -- 2. per-seed PySR search (n_seeds × n_rounds invocations) --
    seed_runs: list[dict] = []
    for s in range(n_seeds):
        pysr_base_seed = sm.derive("e3_pysr", anion=outer_anion, seed_idx=s)
        log.info("  seed %d  base_seed=%d", s, pysr_base_seed)

        pooled_candidates: list[dict] = []
        for rnd in range(n_rounds):
            round_seed = sm.derive(
                "e3_pysr_round", anion=outer_anion, seed_idx=s, round=rnd
            )
            t0 = time.time()
            try:
                model = PySRRegressor(
                    niterations=pysr_cfg.get("niterations", 100),
                    populations=pysr_cfg.get("populations", 40),
                    population_size=pysr_cfg.get("population_size", 33),
                    binary_operators=pysr_cfg.get(
                        "binary_operators", ["+", "-", "*", "/"],
                    ),
                    unary_operators=pysr_cfg.get(
                        "unary_operators",
                        ["exp", "log", "sqrt", "square", "inv", "cube", "cbrt"],
                    ),
                    maxsize=pysr_cfg.get("maxsize", 25),
                    parsimony=pysr_cfg.get("parsimony", 0.05),
                    random_state=int(round_seed % (2**31 - 1)),
                    deterministic=False,
                    verbosity=0,
                    progress=False,
                    temp_equation_file=True,
                )
                model.fit(X_train, y_train, variable_names=E2_FEATURE_NAMES)
            except Exception as e:
                log.warning("    seed=%d round=%d: PySR fit failed: %s",
                            s, rnd, e)
                continue
            eqs = model.equations_
            if eqs is None or len(eqs) == 0:
                continue

            # Score every candidate in this round's Pareto
            for ii in range(len(eqs)):
                c = _score_candidate(model, ii, X_train, y_train,
                                     X_screen, y_screen, X_full)
                if c is not None:
                    c["round"] = rnd
                    pooled_candidates.append(c)
            log.info("    round %d: %d candidates pooled (+%.1fs)",
                     rnd, len(eqs), time.time() - t0)

        # Derive the four selection policies from pooled Pareto.
        seed_result: dict[str, Any] = {
            "seed_idx": s,
            "n_candidates_pooled": len(pooled_candidates),
        }
        for policy in ("V1_train", "V2_screen", "V3_cplx_capped",
                       "V4_stability_only"):
            chosen = _select(pooled_candidates, policy=policy)
            if chosen is None:
                seed_result[policy] = {"error": "no candidate passed policy"}
                continue
            # Evaluate on outer_test using the same stability veto.
            X_outer_proj = _project_features(outer_test_df, E2_FEATURE_NAMES)
            ev = rmse_on(
                chosen["expression"], E2_FEATURE_NAMES,
                X_outer_proj, y_outer,
                stable_range=(stable_lo, stable_hi),
            )
            seed_result[policy] = {
                "expression": chosen["expression"],
                "complexity": chosen["complexity"],
                "train_rmse": chosen["train_rmse"],
                "inner_screen_rmse": chosen["inner_screen_rmse"],
                "stable_frac_full": chosen["stable_frac"],
                "outer_rmse": ev["rmse"],
                "outer_mae": ev["mae"],
                "outer_r2": ev["r2"],
                "outer_n_finite": ev["n_finite"],
                "outer_n_total": ev["n_total"],
            }
        seed_runs.append(seed_result)

    # -- 3. aggregate --
    def _mean_std(values: list[float]) -> tuple[float, float]:
        vals = [v for v in values if v is not None and np.isfinite(v)]
        if not vals:
            return float("nan"), float("nan")
        a = np.asarray(vals, dtype=float)
        return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

    policy_summary: dict[str, Any] = {}
    for policy in ("V1_train", "V2_screen", "V3_cplx_capped",
                   "V4_stability_only"):
        outer_rmses = [sr[policy].get("outer_rmse")
                       for sr in seed_runs
                       if isinstance(sr[policy], dict)
                       and "outer_rmse" in sr[policy]]
        cplxs = [sr[policy].get("complexity")
                 for sr in seed_runs
                 if isinstance(sr[policy], dict)
                 and "complexity" in sr[policy]]
        n_success = len(outer_rmses)
        m_rmse, s_rmse = _mean_std(outer_rmses)
        m_cplx, s_cplx = _mean_std(cplxs)
        policy_summary[policy] = {
            "n_seeds_success": n_success,
            "mean_outer_rmse": m_rmse,
            "std_outer_rmse": s_rmse,
            "mean_complexity": m_cplx,
            "std_complexity": s_cplx,
            "per_seed_outer_rmse": [float(v) if np.isfinite(v) else None
                                    for v in outer_rmses],
        }

    fold_summary = {
        "outer_anion": outer_anion,
        "run_seed": run_seed,
        "inner_split_seed": int(inner_split_seed),
        "n_outer_test": int(len(outer_test_df)),
        "n_inner_train": int(len(X_train)),
        "n_inner_screen": int(len(X_screen)),
        "n_seeds": n_seeds,
        "n_rounds_per_seed": n_rounds,
        "seed_runs": seed_runs,
        "policy_summary": policy_summary,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2))

    logging.getLogger("v2").removeHandler(file_handler)
    file_handler.close()

    log.info("E3 fold anion=%s complete.", outer_anion)
    for policy in ("V1_train", "V2_screen", "V3_cplx_capped",
                   "V4_stability_only"):
        ps = policy_summary[policy]
        log.info("  %-20s  outer RMSE = %.4f ± %.4f  (cplx %.1f ± %.1f, n=%d)",
                 policy,
                 ps["mean_outer_rmse"], ps["std_outer_rmse"],
                 ps["mean_complexity"], ps["std_complexity"],
                 ps["n_seeds_success"])
    return fold_summary
