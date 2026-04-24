"""
v2.expr_eval — safe evaluation of PySR-style expression strings on feature matrices.

PySR returns expressions as Python-valid strings using either generic
`x0, x1, ...` or the actual feature names (when `variable_names` was passed
to PySRRegressor). This module compiles such strings into a callable
``f(X) → np.ndarray`` using a restricted namespace.

It intentionally does NOT use sympy: the strings come from our own process,
the eval scope is locked to a whitelist, and we prefer the numpy-speed
path for evaluating 100-500 samples per call.

The four things it does differently from v1's ad-hoc `make_eval`:

1. Accepts BOTH `x0..xN` style and named-feature style in the same string.
2. Vectorises when possible (math operators are numpy-compatible) instead
   of the per-row Python loop.
3. Returns a sentinel `np.nan` rather than an inf that quietly pollutes
   RMSE computations for unstable expressions.
4. Accepts a kw-only `stable_range=(lo, hi)`. Any prediction outside is
   returned as NaN so downstream code can count unstable predictions
   without arithmetic fireworks.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np


# --- safe operators --------------------------------------------------------


def _safe_exp(v):
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.exp(np.clip(v, a_min=None, a_max=700.0))
    return np.where(np.isfinite(out), out, np.nan)


def _safe_log(v):
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(v)
    return np.where((v > 0) & np.isfinite(out), out, np.nan)


def _safe_sqrt(v):
    v = np.asarray(v, dtype=float)
    with np.errstate(invalid="ignore"):
        out = np.sqrt(v)
    return np.where(v >= 0, out, np.nan)


def _safe_inv(v):
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(1.0, v, where=v != 0)
    return np.where(v != 0, out, np.nan)


def _safe_cbrt(v):
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.abs(v) ** (1.0 / 3)


_SAFE_NS: dict = {
    "exp":    _safe_exp,
    "log":    _safe_log,
    "sqrt":   _safe_sqrt,
    "square": lambda v: np.asarray(v, dtype=float) ** 2,
    "cube":   lambda v: np.asarray(v, dtype=float) ** 3,
    "cbrt":   _safe_cbrt,
    "inv":    _safe_inv,
}


# --- main API --------------------------------------------------------------


def compile_expr(
    expr_str: str,
    feature_names: Sequence[str],
    *,
    stable_range: tuple[float, float] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable ``f(X) → np.ndarray`` that evaluates `expr_str` on
    each row of X. X is assumed to have columns in the order given by
    `feature_names`; the same columns are also bound to `x0..xN` for
    compatibility with PySR's generic output.

    Unstable predictions (outside ``stable_range`` if provided, or
    non-finite) are returned as NaN.
    """
    feature_names = list(feature_names)

    def f(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"expression was compiled for {len(feature_names)} features "
                f"({feature_names}) but X has {X.shape[1]}"
            )

        # Try a single vectorised evaluation first — this handles expressions
        # with numpy-compatible operations (the common case).
        ns: dict = dict(_SAFE_NS)
        ns.update({fn: X[:, i] for i, fn in enumerate(feature_names)})
        ns.update({f"x{i}": X[:, i] for i in range(X.shape[1])})
        ns["__builtins__"] = {}
        try:
            out = eval(expr_str, ns, {})
            out = np.asarray(out, dtype=float)
            if out.ndim == 0:
                out = np.full(X.shape[0], float(out))
        except Exception:
            # Fall back to row-by-row; handles cases where the expression
            # contains scalar-only constructs.
            out = np.empty(X.shape[0], dtype=float)
            for k in range(X.shape[0]):
                row_ns: dict = dict(_SAFE_NS)
                row_ns.update({fn: float(X[k, i]) for i, fn in enumerate(feature_names)})
                row_ns.update({f"x{i}": float(X[k, i]) for i in range(X.shape[1])})
                row_ns["__builtins__"] = {}
                try:
                    out[k] = float(eval(expr_str, row_ns, {}))
                except Exception:
                    out[k] = np.nan

        out = np.where(np.isfinite(out), out, np.nan)
        if stable_range is not None:
            lo, hi = stable_range
            out = np.where((out >= lo) & (out <= hi), out, np.nan)
        return out

    return f


def rmse_on(
    expr_str: str,
    feature_names: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    *,
    stable_range: tuple[float, float] | None = None,
) -> dict:
    """
    One-call evaluator returning {rmse, mae, r2, n_finite, n_total,
    finite_fraction}. Rows where prediction is NaN are dropped from the
    metrics (and their count reported separately) so a partially-unstable
    expression can still be scored on its reliable region.
    """
    f = compile_expr(expr_str, feature_names, stable_range=stable_range)
    y = np.asarray(y, dtype=float)
    pred = f(X)
    mask = np.isfinite(pred)
    if not mask.any():
        return {
            "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"),
            "n_finite": 0, "n_total": int(len(y)), "finite_fraction": 0.0,
        }
    err = pred[mask] - y[mask]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y[mask] - y[mask].mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt(ss_res / mask.sum())),
        "mae":  float(np.abs(err).mean()),
        "r2":   float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot),
        "n_finite": int(mask.sum()),
        "n_total":  int(len(y)),
        "finite_fraction": float(mask.mean()),
    }
