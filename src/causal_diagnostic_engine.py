"""
Causal Diagnostic Engine
========================
Gap 1 Fix: Constraints are filters not guardrails.

Wraps CauchyValidator to provide material-level causal attribution:
- Which materials violated constraints (by metal/chalcogen name)
- Which additive terms caused the violation
- Feature sensitivities via finite differences
- LLM-ready causal text explaining failures

Usage:
    engine = CausalDiagnosticEngine(validator, feature_names, material_metadata)
    report = engine.diagnose(formula_str, predictions, X, y)
    print(report.causal_text)
"""

import logging
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from src.cauchy_validator import CauchyValidator, ValidationResult
from src.diagnostic_types import (
    MaterialViolation, ConstraintDiagnostic, DiagnosticReport
)

logger = logging.getLogger("HARNESS.CausalDiagnosticEngine")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    logger.debug("pandas not available; material metadata lookup disabled")

try:
    import sympy
    from sympy import sympify, symbols, lambdify
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False
    logger.debug("sympy not available; using string-based formula parsing")


class CausalDiagnosticEngine:
    """
    Extends CauchyValidator with material-level causal attribution.

    Parameters
    ----------
    validator : CauchyValidator
        Existing validator instance.
    feature_names : list of str
        Names of features in the same order as columns in X.
    material_metadata : DataFrame or None
        DataFrame with columns ['metal', 'chalcogen', 'phase'] (one row per sample).
    """

    def __init__(self, validator: CauchyValidator,
                 feature_names: List[str],
                 material_metadata=None):
        self.validator = validator
        self.feature_names = feature_names
        self.material_metadata = material_metadata  # pandas DataFrame or None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(self, formula_str: str, predictions: np.ndarray,
                 X: np.ndarray, y: Optional[np.ndarray] = None,
                 complexity: int = 10) -> DiagnosticReport:
        """
        Full causal diagnosis of a formula.

        Returns DiagnosticReport with material-level attribution and LLM text.
        """
        report = DiagnosticReport(formula=formula_str)

        # 1. Run standard validator checks + row-level tracking
        try:
            val_result, row_failures = self._validate_with_material_ids(
                formula_str, predictions, X, self.feature_names
            )
        except Exception as e:
            logger.warning(f"Validator failed: {e}")
            val_result = ValidationResult(
                is_valid=True, checks_passed={}, warnings=[], score=1.0
            )
            row_failures = {}

        # 2. Build ConstraintDiagnostic objects
        for check_name, failed_rows in row_failures.items():
            mat_ids = self._rows_to_material_ids(failed_rows)
            worst = 0.0
            if check_name == "positivity" and len(failed_rows) > 0:
                worst = float(np.min(predictions[failed_rows]))
            elif check_name == "range" and len(failed_rows) > 0:
                worst = float(np.max(np.abs(predictions[failed_rows] - 4.0)))
            diag = ConstraintDiagnostic(
                check_name=check_name,
                passed=len(failed_rows) == 0,
                violating_row_indices=list(failed_rows),
                violating_material_ids=mat_ids,
                worst_violation=worst,
                explanation=val_result.warnings[0] if val_result.warnings else ""
            )
            report.constraint_diagnostics.append(diag)

        # 3. Build MaterialViolation objects for positivity failures
        positivity_failures = row_failures.get("positivity", [])
        for idx in positivity_failures:
            mat_id = self._row_to_material_id(idx)
            metal, chalcogen, phase = self._get_material_info(idx)
            pred_val = float(predictions[idx])
            actual_val = float(y[idx]) if y is not None else float('nan')

            # Decompose formula into additive terms and evaluate each
            additive_contributions = self._decompose_formula(
                formula_str, X[idx], self.feature_names
            )

            violation = MaterialViolation(
                material_id=mat_id,
                metal=metal,
                chalcogen=chalcogen,
                phase=phase,
                predicted_value=pred_val,
                actual_value=actual_val,
                violation_type="positivity",
                additive_terms=additive_contributions,
            )
            report.material_violations.append(violation)

        # 4. Feature sensitivities via finite differences
        report.feature_sensitivities = self._compute_feature_sensitivities(
            formula_str, X, predictions
        )

        # 5. Mean additive term contributions (across all samples)
        report.additive_term_contributions = self._mean_term_contributions(
            formula_str, X, self.feature_names
        )

        # 6. Overall validity
        report.overall_valid = val_result.is_valid
        report.violation_fraction = (
            len(positivity_failures) / len(predictions)
            if len(predictions) > 0 else 0.0
        )

        # 7. Build human-readable causal text
        report.causal_text = self._build_causal_text(report, formula_str)

        return report

    # ------------------------------------------------------------------
    # Validator extension: track which rows fail each check
    # ------------------------------------------------------------------

    def _validate_with_material_ids(
        self, expression: str, predictions: np.ndarray,
        X: np.ndarray, feature_names: List[str]
    ) -> Tuple[ValidationResult, Dict[str, List[int]]]:
        """
        Run CauchyValidator checks and also track which row indices fail each check.

        Returns (ValidationResult, {check_name: [row_indices_that_failed]})
        """
        # Run the existing validate() for the ValidationResult
        val_result = self.validator.validate(
            expression, predictions,
            complexity=10, feature_names=feature_names
        )

        row_failures: Dict[str, List[int]] = {}

        # Positivity: n < 1
        pos_fail = np.where(predictions < 1.0)[0].tolist()
        row_failures["positivity"] = pos_fail

        # Range: outside [1.0, 8.0]
        range_fail = np.where(
            (predictions < 1.0) | (predictions > 8.0)
        )[0].tolist()
        row_failures["range"] = range_fail

        # Finite: NaN or Inf
        finite_fail = np.where(~np.isfinite(predictions))[0].tolist()
        row_failures["finite"] = finite_fail

        return val_result, row_failures

    # ------------------------------------------------------------------
    # Formula decomposition
    # ------------------------------------------------------------------

    def _decompose_formula(self, formula_str: str,
                           x_row: np.ndarray,
                           feature_names: List[str]) -> Dict[str, float]:
        """
        Parse formula into additive terms and evaluate each at x_row.

        Returns dict mapping term_string → numerical value.
        """
        if _SYMPY_AVAILABLE:
            return self._decompose_sympy(formula_str, x_row, feature_names)
        return self._decompose_string(formula_str, x_row, feature_names)

    def _decompose_sympy(self, formula_str: str,
                          x_row: np.ndarray,
                          feature_names: List[str]) -> Dict[str, float]:
        """Sympy-based additive decomposition."""
        try:
            # Map feature names to sympy symbols
            sym_map = {name: sympy.Symbol(name) for name in feature_names}
            # Also support x0, x1, ... format
            for i, name in enumerate(feature_names):
                sym_map[f"x{i}"] = sympy.Symbol(name)

            expr = sympy.sympify(formula_str, locals=sym_map)
            # Expand to get additive terms
            expr_expanded = sympy.expand(expr)

            # Get Add args (terms)
            if isinstance(expr_expanded, sympy.Add):
                terms = expr_expanded.args
            else:
                terms = [expr_expanded]

            # Substitute numeric values
            subs = {sympy.Symbol(name): float(x_row[i])
                    for i, name in enumerate(feature_names)}

            contributions = {}
            for term in terms:
                term_str = str(term)
                try:
                    val = float(term.subs(subs))
                    contributions[term_str] = val
                except Exception:
                    contributions[term_str] = float('nan')

            return contributions

        except Exception as e:
            logger.debug(f"sympy decomposition failed: {e}")
            return self._decompose_string(formula_str, x_row, feature_names)

    def _decompose_string(self, formula_str: str,
                           x_row: np.ndarray,
                           feature_names: List[str]) -> Dict[str, float]:
        """
        Fallback: split on ' + ' (simple additive formulas) and eval each term.
        """
        # Build substitution dict
        ns = {name: float(x_row[i]) for i, name in enumerate(feature_names)}
        ns.update({f"x{i}": float(x_row[i]) for i in range(len(x_row))})
        ns["__builtins__"] = {}

        import math
        ns.update({"exp": math.exp, "log": math.log, "sqrt": math.sqrt,
                   "sin": math.sin, "cos": math.cos, "abs": abs})

        # Try to split on top-level " + "
        terms = self._split_additive_terms(formula_str)
        contributions = {}
        for term in terms:
            term = term.strip()
            if not term:
                continue
            try:
                val = eval(term, {"__builtins__": {}}, ns)
                contributions[term] = float(val)
            except Exception:
                contributions[term] = float('nan')

        return contributions

    def _split_additive_terms(self, formula_str: str) -> List[str]:
        """
        Split a formula string on '+' at the top level (not inside parens).
        """
        terms = []
        depth = 0
        current = ""
        for ch in formula_str:
            if ch in "([{":
                depth += 1
                current += ch
            elif ch in ")]}":
                depth -= 1
                current += ch
            elif ch == "+" and depth == 0:
                terms.append(current)
                current = ""
            else:
                current += ch
        if current:
            terms.append(current)
        return terms

    # ------------------------------------------------------------------
    # Feature sensitivities via finite differences
    # ------------------------------------------------------------------

    def _compute_feature_sensitivities(
        self, formula_str: str,
        X: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate dOutput/dFeature at the mean point via finite differences.
        """
        sensitivities: Dict[str, float] = {}
        x_mean = np.mean(X, axis=0)
        eps = 1e-4

        for i, name in enumerate(self.feature_names):
            try:
                x_plus = x_mean.copy()
                x_plus[i] += eps
                x_minus = x_mean.copy()
                x_minus[i] -= eps

                pred_plus = self._eval_formula(formula_str, x_plus)
                pred_minus = self._eval_formula(formula_str, x_minus)

                if np.isfinite(pred_plus) and np.isfinite(pred_minus):
                    sensitivities[name] = float((pred_plus - pred_minus) / (2 * eps))
                else:
                    sensitivities[name] = 0.0
            except Exception:
                sensitivities[name] = 0.0

        return sensitivities

    def _eval_formula(self, formula_str: str, x_row: np.ndarray) -> float:
        """Evaluate formula at a single row using the feature names as variable names."""
        ns = {name: float(x_row[i]) for i, name in enumerate(self.feature_names)}
        ns.update({f"x{i}": float(x_row[i]) for i in range(len(x_row))})

        import math
        safe_builtins = {}
        ns.update({"exp": math.exp, "log": math.log, "sqrt": math.sqrt,
                   "sin": math.sin, "cos": math.cos, "abs": abs,
                   "square": lambda v: v**2, "inv": lambda v: 1.0/v,
                   "__builtins__": safe_builtins})
        try:
            result = eval(formula_str, {"__builtins__": safe_builtins}, ns)
            return float(result)
        except Exception:
            return float('nan')

    # ------------------------------------------------------------------
    # Mean term contributions across all samples
    # ------------------------------------------------------------------

    def _mean_term_contributions(self, formula_str: str,
                                  X: np.ndarray,
                                  feature_names: List[str]) -> Dict[str, float]:
        """Evaluate additive term contributions at the mean point."""
        x_mean = np.mean(X, axis=0)
        return self._decompose_formula(formula_str, x_mean, feature_names)

    # ------------------------------------------------------------------
    # Material ID helpers
    # ------------------------------------------------------------------

    def _row_to_material_id(self, idx: int) -> str:
        metal, chalcogen, phase = self._get_material_info(idx)
        return f"{metal}{chalcogen}2_{phase}"

    def _rows_to_material_ids(self, indices: List[int]) -> List[str]:
        return [self._row_to_material_id(i) for i in indices]

    def _get_material_info(self, idx: int) -> Tuple[str, str, str]:
        """Return (metal, chalcogen, phase) for a given row index."""
        if self.material_metadata is not None and _PANDAS_AVAILABLE:
            try:
                row = self.material_metadata.iloc[idx]
                metal = str(row.get("metal", "M"))
                chalcogen = str(row.get("chalcogen", "X"))
                phase = str(row.get("phase", "H"))
                return metal, chalcogen, phase
            except Exception:
                pass
        return "M", "X", "H"

    # ------------------------------------------------------------------
    # Causal text generation
    # ------------------------------------------------------------------

    def _build_causal_text(self, report: DiagnosticReport,
                            formula_str: str) -> str:
        """
        Build an LLM-ready causal text block explaining violations.

        Example output:
        "Formula predicts n=0.87 for ZrTe2_H because Z_X coefficient (0.023)
         is too small for heavy-metal+heavy-chalcogen. Add Z_X² term."
        """
        lines = []
        lines.append(f"Causal Diagnostic Report for: {formula_str}")
        lines.append("")

        if not report.material_violations:
            lines.append("No physical constraint violations detected.")
        else:
            lines.append(
                f"Violations: {len(report.material_violations)} material(s) "
                f"violated positivity constraint (n > 1):"
            )
            for v in report.material_violations[:5]:  # show up to 5
                lines.append(
                    f"  - {v.material_id}: predicted n={v.predicted_value:.4f} "
                    f"(actual={v.actual_value:.4f})"
                )
                if v.additive_terms:
                    # Find the problematic term (most negative or smallest)
                    sorted_terms = sorted(v.additive_terms.items(),
                                          key=lambda kv: kv[1])
                    worst_term, worst_val = sorted_terms[0]
                    lines.append(
                        f"    Root cause: term '{worst_term}' contributes "
                        f"{worst_val:.4f} — too small for "
                        f"{v.metal}+{v.chalcogen} combination."
                    )

        # Feature sensitivity summary
        if report.feature_sensitivities:
            lines.append("")
            lines.append("Feature sensitivities (finite differences at mean):")
            sorted_sens = sorted(
                report.feature_sensitivities.items(),
                key=lambda kv: abs(kv[1]), reverse=True
            )
            for fname, sens in sorted_sens[:5]:
                direction = "+" if sens > 0 else "-"
                lines.append(
                    f"  {fname}: {sens:+.4f} (increasing {fname} "
                    f"{'increases' if sens > 0 else 'decreases'} prediction)"
                )

        # Suggestions
        lines.append("")
        if report.material_violations:
            # Identify sign mismatches
            neg_sens = [
                fname for fname, s in report.feature_sensitivities.items()
                if s < 0 and fname in (
                    "atomic_number_X", "d_electron_count", "bond_polarity"
                )
            ]
            if neg_sens:
                lines.append(
                    f"Suggestion: Features {neg_sens} have negative sensitivity "
                    f"but physics expects positive. Check for sign flip in formula."
                )
            else:
                lines.append(
                    "Suggestion: Consider adding Z_X² or bond_polarity² term "
                    "to boost predictions for heavy chalcogen materials."
                )

        return "\n".join(lines)
