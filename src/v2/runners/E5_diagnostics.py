"""
E5 — Diagnostic experiments defending the v2 claims.

Three diagnostics, all dispatched by the ``mode`` key in the config:

  1. ``zm_only``       — Reviewer critique: "Z_M is a periodic-table lookup."
                         Re-run the outer-LOCO protocol with vanilla PySR
                         restricted to a single input feature (default Z_M).
                         If it cannot match the full 20-feature baseline,
                         Z_M alone is insufficient.

  2. ``lomo``          — Leave-one-metal-out. Outer-LOCO controls chemistry
                         at the anion level; LOMO controls at the metal
                         level. Vanilla PySR on all 20 features, held-out
                         metal as outer_test. Only metals with N >= cutoff
                         (default 5) are evaluated.

  3. ``target_permute`` — Null test. Shuffle the target across samples,
                          re-run vanilla PySR on one fold. If the model
                          still beats std(y), something leaks.

All three share the feature layout and inner-split seed derivation
conventions used by E2/E3 so the numbers are directly comparable.
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

from src.v2.runners.E2_outer_loco import (
    E2_FEATURE_NAMES, _build_feature_matrix, _project_features,
)
from src.v2.expr_eval import rmse_on
from src.v2.seeds import SeedManager


log = logging.getLogger("v2.runners.E5")


# -- shared helpers ----------------------------------------------------------


def _install_file_log(run_dir: Path, tag: str) -> logging.Handler:
    fh = logging.FileHandler(run_dir / f"e5_{tag}.log", mode="w")
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


def _score_candidate_on(
    model, idx: int,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_sc: np.ndarray, y_sc: np.ndarray,
    X_full: np.ndarray,
) -> dict[str, Any] | None:
    """Same contract as E3._score_candidate, kept local so E5 does not
    quietly inherit a signature change from E3."""
    try:
        yp_tr = model.predict(X_tr, index=idx)
        yp_sc = model.predict(X_sc, index=idx)
        yp_full = model.predict(X_full, index=idx)
    except Exception:
        return None
    if not np.all(np.isfinite(yp_tr)):
        return None
    stable = np.isfinite(yp_full) & (np.abs(yp_full) <= 20.0)
    row = model.equations_.iloc[idx]
    return {
        "expression": str(row["equation"]),
        "complexity": int(row["complexity"]),
        "train_rmse": _rmse(y_tr, yp_tr),
        "inner_screen_rmse": _rmse(y_sc, yp_sc),
        "stable_frac": float(stable.mean()),
    }


def _run_pysr_rounds(
    X_train: np.ndarray, y_train: np.ndarray,
    X_screen: np.ndarray, y_screen: np.ndarray,
    X_full: np.ndarray,
    *,
    variable_names: list[str],
    pysr_cfg: dict,
    sm: SeedManager,
    seed_key: str,
    seed_idx: int,
    n_rounds: int,
) -> list[dict]:
    """
    Run ``n_rounds`` PySR invocations, pool their Pareto candidates, score
    each on train / screen / full-set stability, and return the pooled list.
    Mirrors E3's round loop but accepts an explicit variable-name list so
    Z_M-only and full-feature runs share the code.
    """
    import signal

    from pysr import PySRRegressor

    # Per-round wall-clock cap. PySR can deadlock on degenerate input
    # (e.g. single integer-valued column) — we hit this on Z_M-only Te
    # seed 2 round 2 in the first attempt. SIGALRM is sufficient on macOS
    # for a single-process bound; the alarm fires inside the C/Julia
    # extension and Python re-enters the handler at the next checkpoint.
    round_timeout_s = float(pysr_cfg.get("round_timeout_s", 60.0))

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"PySR fit exceeded {round_timeout_s}s")

    pooled: list[dict] = []
    for rnd in range(n_rounds):
        round_seed = sm.derive(seed_key, seed_idx=seed_idx, round=rnd)
        t0 = time.time()
        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(round_timeout_s))
        try:
            model = PySRRegressor(
                niterations=pysr_cfg.get("niterations", 50),
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
            model.fit(X_train, y_train, variable_names=variable_names)
        except TimeoutError as e:
            log.warning("    seed=%d round=%d: %s — skipping round",
                        seed_idx, rnd, e)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)
            continue
        except Exception as e:
            log.warning("    seed=%d round=%d: PySR fit failed: %s",
                        seed_idx, rnd, e)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)
            continue
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)
        eqs = model.equations_
        if eqs is None or len(eqs) == 0:
            continue
        for ii in range(len(eqs)):
            c = _score_candidate_on(model, ii, X_train, y_train,
                                    X_screen, y_screen, X_full)
            if c is not None:
                c["round"] = rnd
                pooled.append(c)
        log.info("    round %d: %d candidates pooled (+%.1fs)",
                 rnd, len(eqs), time.time() - t0)
    return pooled


def _select_v2_screen(pool: list[dict]) -> dict | None:
    valid = [c for c in pool if c["train_rmse"] < float("inf")]
    if not valid:
        return None
    return min(valid, key=lambda c: c["inner_screen_rmse"])


# -- E5.1  Z_M-only PySR -----------------------------------------------------


def _run_zm_only(cfg: dict, run_seed: int, run_dir: Path) -> dict:
    """One pass over all outer-anion folds restricted to a single input
    feature (default Z_M). Mirrors E3's outer-LOCO partitioning so the
    numbers plug into the E3 comparison table."""
    anions = cfg.get("anions", ["O", "S", "Se", "Te", "F", "Cl", "Br",
                                 "I", "N", "P"])
    feature = cfg.get("feature", "atomic_number_M")
    n_seeds = int(cfg.get("n_seeds", 3))
    n_rounds = int(cfg.get("n_rounds", 5))
    data_csv = cfg.get("data_csv", "data/extended/tier2_binary.csv")
    target = cfg.get("target", "cauchy_A")
    stable_range = tuple(cfg.get("stability_range", [0.0, 20.0]))
    pysr_cfg = cfg.get("pysr", {})

    if feature not in E2_FEATURE_NAMES:
        raise ValueError(f"feature {feature!r} not in E2_FEATURE_NAMES")
    feature_idx = E2_FEATURE_NAMES.index(feature)
    var_names = [feature]
    short_name = "Z_M" if feature == "atomic_number_M" else feature

    df = pd.read_csv(data_csv)
    sm = SeedManager(run_seed=run_seed)

    per_anion: list[dict] = []
    for anion in anions:
        if anion not in set(df["anion"].unique()):
            log.warning("anion %s not present; skipping", anion)
            continue
        outer_test_df = df[df["anion"] == anion].reset_index(drop=True)
        outer_pool_df = df[df["anion"] != anion].reset_index(drop=True)

        X_pool_full = _build_feature_matrix(outer_pool_df)
        X_outer_full = _build_feature_matrix(outer_test_df)
        X_full_full = _build_feature_matrix(df)
        y_pool = outer_pool_df[target].to_numpy(dtype=float)
        y_outer = outer_test_df[target].to_numpy(dtype=float)

        X_pool = X_pool_full[:, [feature_idx]]
        X_outer = X_outer_full[:, [feature_idx]]
        X_full = X_full_full[:, [feature_idx]]

        inner_split_seed = sm.derive("e2_inner_split", anion=anion)
        rng = np.random.default_rng(inner_split_seed)
        perm = rng.permutation(len(outer_pool_df))
        cut = int(0.8 * len(perm))
        tr_idx, sc_idx = perm[:cut], perm[cut:]
        X_train, y_train = X_pool[tr_idx], y_pool[tr_idx]
        X_screen, y_screen = X_pool[sc_idx], y_pool[sc_idx]

        log.info("[zm_only anion=%s]  outer N=%d  train N=%d  screen N=%d",
                 anion, len(outer_test_df), len(X_train), len(X_screen))

        seed_results: list[dict] = []
        for s in range(n_seeds):
            pool = _run_pysr_rounds(
                X_train, y_train, X_screen, y_screen, X_full,
                variable_names=var_names,
                pysr_cfg=pysr_cfg, sm=sm,
                seed_key=f"e5_zm_only_{anion}",
                seed_idx=s, n_rounds=n_rounds,
            )
            chosen = _select_v2_screen(pool)
            if chosen is None:
                seed_results.append({"seed_idx": s, "error": "no candidate"})
                continue
            ev = rmse_on(chosen["expression"], var_names,
                         X_outer, y_outer, stable_range=stable_range)
            seed_results.append({
                "seed_idx": s,
                "expression": chosen["expression"],
                "complexity": chosen["complexity"],
                "inner_screen_rmse": chosen["inner_screen_rmse"],
                "outer_rmse": ev["rmse"],
                "outer_mae": ev["mae"],
                "outer_r2": ev["r2"],
                "outer_n_finite": ev["n_finite"],
                "outer_n_total": ev["n_total"],
            })

        outer_rmses = [r["outer_rmse"] for r in seed_results
                        if isinstance(r, dict) and "outer_rmse" in r
                        and np.isfinite(r["outer_rmse"])]
        mean_r = float(np.mean(outer_rmses)) if outer_rmses else float("nan")
        std_r = float(np.std(outer_rmses, ddof=1)) if len(outer_rmses) > 1 else 0.0
        per_anion.append({
            "anion": anion,
            "n_outer_test": int(len(outer_test_df)),
            "seed_runs": seed_results,
            "mean_outer_rmse": mean_r,
            "std_outer_rmse": std_r,
        })
        log.info("[zm_only anion=%s]  mean outer RMSE = %.4f ± %.4f  (%d seeds)",
                 anion, mean_r, std_r, len(outer_rmses))

    total_N = sum(r["n_outer_test"] for r in per_anion)
    weighted = sum(r["mean_outer_rmse"] * r["n_outer_test"]
                   for r in per_anion
                   if np.isfinite(r["mean_outer_rmse"])) / total_N if total_N else float("nan")

    out = {
        "mode": "zm_only",
        "feature": feature,
        "feature_short": short_name,
        "n_seeds": n_seeds,
        "n_rounds": n_rounds,
        "per_anion": per_anion,
        "weighted_outer_rmse": float(weighted),
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(out, indent=2))
    log.info("E5 zm_only done. weighted outer RMSE = %.4f on N=%d",
             weighted, total_N)
    return out


# -- E5.2  leave-one-metal-out ----------------------------------------------


def _run_lomo(cfg: dict, run_seed: int, run_dir: Path) -> dict:
    min_n = int(cfg.get("min_metal_count", 5))
    n_seeds = int(cfg.get("n_seeds", 1))
    n_rounds = int(cfg.get("n_rounds", 5))
    data_csv = cfg.get("data_csv", "data/extended/tier2_binary.csv")
    target = cfg.get("target", "cauchy_A")
    stable_range = tuple(cfg.get("stability_range", [0.0, 20.0]))
    pysr_cfg = cfg.get("pysr", {})

    df = pd.read_csv(data_csv)
    counts = df["metal"].value_counts()
    metals = sorted([m for m, c in counts.items() if c >= min_n])
    log.info("LOMO metals (N >= %d): %d of %d",
             min_n, len(metals), len(counts))

    sm = SeedManager(run_seed=run_seed)

    per_metal: list[dict] = []
    for metal in metals:
        outer_test_df = df[df["metal"] == metal].reset_index(drop=True)
        outer_pool_df = df[df["metal"] != metal].reset_index(drop=True)

        X_pool = _build_feature_matrix(outer_pool_df)
        X_outer = _build_feature_matrix(outer_test_df)
        X_full = _build_feature_matrix(df)
        y_pool = outer_pool_df[target].to_numpy(dtype=float)
        y_outer = outer_test_df[target].to_numpy(dtype=float)

        inner_split_seed = sm.derive("e5_lomo_split", metal=metal)
        rng = np.random.default_rng(inner_split_seed)
        perm = rng.permutation(len(outer_pool_df))
        cut = int(0.8 * len(perm))
        tr_idx, sc_idx = perm[:cut], perm[cut:]
        X_train, y_train = X_pool[tr_idx], y_pool[tr_idx]
        X_screen, y_screen = X_pool[sc_idx], y_pool[sc_idx]

        log.info("[lomo metal=%s]  outer N=%d  train N=%d  screen N=%d",
                 metal, len(outer_test_df), len(X_train), len(X_screen))

        seed_results: list[dict] = []
        for s in range(n_seeds):
            pool = _run_pysr_rounds(
                X_train, y_train, X_screen, y_screen, X_full,
                variable_names=E2_FEATURE_NAMES,
                pysr_cfg=pysr_cfg, sm=sm,
                seed_key=f"e5_lomo_{metal}",
                seed_idx=s, n_rounds=n_rounds,
            )
            chosen = _select_v2_screen(pool)
            if chosen is None:
                seed_results.append({"seed_idx": s, "error": "no candidate"})
                continue
            ev = rmse_on(chosen["expression"], E2_FEATURE_NAMES,
                         X_outer, y_outer, stable_range=stable_range)
            seed_results.append({
                "seed_idx": s,
                "expression": chosen["expression"],
                "complexity": chosen["complexity"],
                "inner_screen_rmse": chosen["inner_screen_rmse"],
                "outer_rmse": ev["rmse"],
                "outer_mae": ev["mae"],
                "outer_r2": ev["r2"],
                "outer_n_finite": ev["n_finite"],
                "outer_n_total": ev["n_total"],
            })

        outer_rmses = [r["outer_rmse"] for r in seed_results
                        if isinstance(r, dict) and "outer_rmse" in r
                        and np.isfinite(r["outer_rmse"])]
        mean_r = float(np.mean(outer_rmses)) if outer_rmses else float("nan")
        std_r = float(np.std(outer_rmses, ddof=1)) if len(outer_rmses) > 1 else 0.0
        per_metal.append({
            "metal": metal,
            "n_outer_test": int(len(outer_test_df)),
            "seed_runs": seed_results,
            "mean_outer_rmse": mean_r,
            "std_outer_rmse": std_r,
        })
        log.info("[lomo metal=%s]  mean outer RMSE = %.4f ± %.4f  (%d seeds)",
                 metal, mean_r, std_r, len(outer_rmses))

    total_N = sum(r["n_outer_test"] for r in per_metal)
    weighted = sum(r["mean_outer_rmse"] * r["n_outer_test"]
                   for r in per_metal
                   if np.isfinite(r["mean_outer_rmse"])) / total_N if total_N else float("nan")

    out = {
        "mode": "lomo",
        "min_metal_count": min_n,
        "n_metals_evaluated": len(metals),
        "n_seeds": n_seeds,
        "n_rounds": n_rounds,
        "per_metal": per_metal,
        "weighted_outer_rmse": float(weighted),
        "total_outer_N": int(total_N),
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(out, indent=2))
    log.info("E5 lomo done. weighted outer RMSE = %.4f on N=%d across %d metals",
             weighted, total_N, len(metals))
    return out


# -- E5.3  target-permutation null test --------------------------------------


def _run_target_permute(cfg: dict, run_seed: int, run_dir: Path) -> dict:
    """Shuffle the target across the outer_pool + outer_test union, then
    fit vanilla PySR under the usual outer-LOCO protocol. Expected outcome:
    outer RMSE ≈ std(y_shuffled); a significantly lower value indicates
    leakage."""
    anion = cfg.get("outer_anion", "O")
    n_seeds = int(cfg.get("n_seeds", 5))
    n_rounds = int(cfg.get("n_rounds", 3))
    data_csv = cfg.get("data_csv", "data/extended/tier2_binary.csv")
    target = cfg.get("target", "cauchy_A")
    stable_range = tuple(cfg.get("stability_range", [0.0, 20.0]))
    pysr_cfg = cfg.get("pysr", {})

    df = pd.read_csv(data_csv)
    sm = SeedManager(run_seed=run_seed)

    outer_test_df = df[df["anion"] == anion].reset_index(drop=True)
    outer_pool_df = df[df["anion"] != anion].reset_index(drop=True)

    X_pool_full = _build_feature_matrix(outer_pool_df)
    X_outer_full = _build_feature_matrix(outer_test_df)
    X_full = _build_feature_matrix(df)
    y_orig_global = df[target].to_numpy(dtype=float)
    std_y_global = float(np.std(y_orig_global))

    inner_split_seed = sm.derive("e2_inner_split", anion=anion)
    rng_split = np.random.default_rng(inner_split_seed)
    perm_split = rng_split.permutation(len(outer_pool_df))
    cut = int(0.8 * len(perm_split))
    tr_idx, sc_idx = perm_split[:cut], perm_split[cut:]

    seed_results: list[dict] = []
    for s in range(n_seeds):
        shuffle_seed = sm.derive("e5_permute_y", seed_idx=s)
        rng_y = np.random.default_rng(shuffle_seed)
        # Shuffle target across the ENTIRE dataset so train/screen/outer
        # all receive independently permuted labels.
        perm_y = rng_y.permutation(len(df))
        y_shuffled_all = y_orig_global[perm_y]
        y_pool_sh = y_shuffled_all[df["anion"].to_numpy() != anion]
        y_outer_sh = y_shuffled_all[df["anion"].to_numpy() == anion]
        y_train = y_pool_sh[tr_idx]
        y_screen = y_pool_sh[sc_idx]

        std_y_outer = float(np.std(y_outer_sh))
        log.info("[permute seed=%d] shuffle_seed=%d  std(y_outer)=%.4f",
                 s, int(shuffle_seed), std_y_outer)

        pool = _run_pysr_rounds(
            X_pool_full[tr_idx], y_train,
            X_pool_full[sc_idx], y_screen,
            X_full,
            variable_names=E2_FEATURE_NAMES,
            pysr_cfg=pysr_cfg, sm=sm,
            seed_key=f"e5_permute_{anion}",
            seed_idx=s, n_rounds=n_rounds,
        )
        chosen = _select_v2_screen(pool)
        if chosen is None:
            seed_results.append({
                "seed_idx": s, "std_y_outer": std_y_outer,
                "error": "no candidate",
            })
            continue
        ev = rmse_on(chosen["expression"], E2_FEATURE_NAMES,
                     X_outer_full, y_outer_sh, stable_range=stable_range)
        seed_results.append({
            "seed_idx": s,
            "shuffle_seed": int(shuffle_seed),
            "std_y_outer": std_y_outer,
            "expression": chosen["expression"],
            "complexity": chosen["complexity"],
            "inner_screen_rmse": chosen["inner_screen_rmse"],
            "outer_rmse": ev["rmse"],
            "outer_r2": ev["r2"],
            "outer_n_finite": ev["n_finite"],
            "outer_n_total": ev["n_total"],
        })

    outer_rmses = [r["outer_rmse"] for r in seed_results
                    if isinstance(r, dict) and "outer_rmse" in r
                    and np.isfinite(r["outer_rmse"])]
    std_outers = [r["std_y_outer"] for r in seed_results
                   if isinstance(r, dict) and "std_y_outer" in r]
    mean_r = float(np.mean(outer_rmses)) if outer_rmses else float("nan")
    mean_std_y = float(np.mean(std_outers)) if std_outers else float("nan")
    ratio = mean_r / mean_std_y if mean_std_y else float("nan")

    out = {
        "mode": "target_permute",
        "outer_anion": anion,
        "n_seeds": n_seeds,
        "n_rounds": n_rounds,
        "n_outer_test": int(len(outer_test_df)),
        "std_y_global": std_y_global,
        "mean_std_y_outer_shuffled": mean_std_y,
        "mean_outer_rmse_shuffled": mean_r,
        "rmse_over_std_ratio": float(ratio),
        "seed_runs": seed_results,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(out, indent=2))
    log.info("E5 target_permute done. mean RMSE=%.4f  mean std(y_outer)=%.4f  ratio=%.2f",
             mean_r, mean_std_y, ratio)
    return out


# -- dispatch ----------------------------------------------------------------


def run(config_path: Path, run_seed: int, run_dir: Path) -> dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mode = cfg.get("mode")
    if mode not in {"zm_only", "lomo", "target_permute"}:
        raise ValueError(f"E5 config missing/invalid 'mode' (got {mode!r})")

    fh = _install_file_log(run_dir, mode)
    try:
        if mode == "zm_only":
            return _run_zm_only(cfg, run_seed, run_dir)
        if mode == "lomo":
            return _run_lomo(cfg, run_seed, run_dir)
        return _run_target_permute(cfg, run_seed, run_dir)
    finally:
        logging.getLogger("v2").removeHandler(fh)
        fh.close()
