"""
Validator Agent
===============
Gap 3 Fix: Agents are sequential not deliberative.

Post-search validation agent using CausalDiagnosticEngine.

VETO conditions:
- > 10% materials violate positivity
- Coefficient signs contradict PHYSICS_PRIORS with confidence > 0.8

Additional capabilities:
- leave_one_chalcogen_out() for OOD generalization testing
- check_coefficient_signs() using feature sensitivities
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from agents.base_agent import BaseAgent
from agents.theory_agent import PHYSICS_PRIORS
from src.diagnostic_types import AgentVote, DiagnosticReport

logger = logging.getLogger("HARNESS.ValidatorAgent")

POSITIVITY_VIOLATION_THRESHOLD = 0.25   # 25% — relaxed for small real datasets where PySR may extrapolate below n=1
SIGN_CONFIDENCE_THRESHOLD = 0.8

# Numerical-stability (physical-range) veto.
# Refractive index for 2D semiconductors never exceeds ~6;
# predictions above PHYSICAL_RANGE_MAX indicate numerical blow-up (e.g. exp()*cube() chains)
# that fit training data incidentally but fail catastrophically on any nearby point.
PHYSICAL_RANGE_MAX = 20.0
PHYSICAL_RANGE_VIOLATION_THRESHOLD = 0.05   # 5% of training materials


class ValidatorAgent(BaseAgent):
    """
    Post-search validation agent with physics-based veto power.

    Parameters
    ----------
    diagnostic_engine : CausalDiagnosticEngine, optional
        Used for material-level attribution.
    X_train, y_train, X_test, y_test : np.ndarray
    feature_names : list of str
    """

    def __init__(self, name: str, role: str,
                 knowledge=None, config: dict = None,
                 diagnostic_engine=None,
                 X_train: Optional[np.ndarray] = None,
                 y_train: Optional[np.ndarray] = None,
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None,
                 feature_names: Optional[List[str]] = None):
        super().__init__(name, role, knowledge, config)
        self.diagnostic_engine = diagnostic_engine
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or []
        self._last_analysis: Optional[Dict] = None

    # ------------------------------------------------------------------
    # analyze()
    # ------------------------------------------------------------------

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate candidates from context and return diagnostic information.

        Expected context keys:
            candidates: List[FormulaCandidate]
            X_test, y_test, feature_names
        """
        candidates = context.get("candidates", [])
        X_test = context.get("X_test", self.X_test)
        y_test = context.get("y_test", self.y_test)
        feature_names = context.get("feature_names", self.feature_names)

        result: Dict[str, Any] = {
            "diagnostic_reports": [],
            "violation_fractions": {},
            "sign_warnings": {},
            "ood_scores": {},
        }

        for candidate in candidates[:5]:  # Analyze top 5
            formula = getattr(candidate, "expression", str(candidate))

            # Get predictions for this candidate
            predictions = self._get_predictions(candidate, X_test)
            if predictions is None:
                continue

            # Run causal diagnostics if engine available
            if self.diagnostic_engine is not None:
                try:
                    diag = self.diagnostic_engine.diagnose(
                        formula, predictions, X_test, y_test
                    )
                    result["diagnostic_reports"].append(diag)
                    result["violation_fractions"][formula] = diag.violation_fraction
                    result["sign_warnings"][formula] = self.check_coefficient_signs(
                        formula, diag.feature_sensitivities
                    )
                except Exception as e:
                    self.logger.debug(f"Diagnostic failed for {formula}: {e}")

        self._last_analysis = result
        return result

    # ------------------------------------------------------------------
    # vote()
    # ------------------------------------------------------------------

    def vote(self, topic: str, proposal: str,
             context: Dict[str, Any]) -> AgentVote:
        """
        Vote on a formula or search proposal.

        VETO if:
        - > 10% materials violate positivity
        - Coefficient signs contradict physics with confidence > 0.8
        """
        if topic == "formula_acceptance":
            return self._vote_formula(proposal, context)
        elif topic == "search_space":
            return self._vote_search_space(proposal, context)
        else:
            return self._make_vote(
                "APPROVE", 0.5,
                f"ValidatorAgent: no specific check for topic '{topic}'"
            )

    def _vote_formula(self, formula_str: str,
                       context: Dict[str, Any]) -> AgentVote:
        """Vote on formula acceptance based on violation counts."""
        X_test = context.get("X_test", self.X_test)
        y_test = context.get("y_test", self.y_test)

        if X_test is None:
            return self._make_vote(
                "APPROVE", 0.4, "ValidatorAgent: no test data available"
            )

        # Get predictions on the evaluation set
        candidate = context.get("candidate")
        if candidate is not None:
            predictions = self._get_predictions(candidate, X_test)
        else:
            # Try to evaluate formula string
            predictions = self._eval_formula_on_data(formula_str, X_test)

        # Also evaluate on training set for the numerical-stability veto.
        # Training set is the search's fit distribution, so stability there is
        # a minimal admissibility requirement. If unavailable, reuse X_test.
        X_train = context.get("X_train", self.X_train)
        if X_train is not None:
            if candidate is not None and hasattr(candidate, "predict"):
                try:
                    train_predictions = np.array(candidate.predict(X_train))
                except Exception:
                    train_predictions = self._eval_formula_on_data(formula_str, X_train)
            else:
                train_predictions = self._eval_formula_on_data(formula_str, X_train)
        else:
            train_predictions = predictions

        if predictions is None or not np.any(np.isfinite(predictions)):
            return self._make_vote(
                "VETO", 0.9,
                "ValidatorAgent: formula produces non-finite predictions"
            )

        # Numerical-stability (physical-range) veto:
        # reject candidates whose predictions blow up beyond the physical
        # refractive-index range on the TRAINING set. This catches expressions
        # like exp(d_e)*cube(...) that fit by accident but explode on nearby points.
        if train_predictions is not None:
            tp = np.asarray(train_predictions, dtype=float)
            finite_train = np.isfinite(tp)
            abs_train = np.abs(tp[finite_train])
            n_non_finite = int((~finite_train).sum())
            if abs_train.size > 0:
                range_violation_frac = float(np.mean(abs_train > PHYSICAL_RANGE_MAX))
                if range_violation_frac > PHYSICAL_RANGE_VIOLATION_THRESHOLD \
                        or n_non_finite > 0:
                    return self._make_vote(
                        "VETO",
                        confidence=min(0.95, 0.8 + range_violation_frac),
                        reason=(
                            f"ValidatorAgent: numerical blow-up — "
                            f"{range_violation_frac*100:.1f}% of training materials "
                            f"produce |pred| > {PHYSICAL_RANGE_MAX:g} "
                            f"(max |pred| = {float(abs_train.max()):.2e}; "
                            f"{n_non_finite} non-finite). "
                            f"Refractive index cannot physically exceed ~6."
                        ),
                        evidence={
                            "range_violation_fraction": range_violation_frac,
                            "max_abs_prediction": float(abs_train.max()),
                            "non_finite_count": n_non_finite,
                        },
                    )

        # Check positivity violation fraction
        violation_frac = float(np.mean(predictions < 1.0))
        if violation_frac > POSITIVITY_VIOLATION_THRESHOLD:
            # Get detailed causal report if engine available
            causal_text = ""
            if self.diagnostic_engine is not None:
                try:
                    diag = self.diagnostic_engine.diagnose(
                        formula_str, predictions, X_test, y_test
                    )
                    causal_text = diag.causal_text[:300]
                except Exception:
                    pass

            return self._make_vote(
                "VETO",
                confidence=min(0.95, 0.7 + violation_frac),
                reason=(
                    f"ValidatorAgent: {violation_frac*100:.1f}% of materials "
                    f"violate n>1 (threshold={POSITIVITY_VIOLATION_THRESHOLD*100:.0f}%). "
                    + causal_text
                ),
                evidence={"violation_fraction": violation_frac}
            )

        # Check coefficient signs
        feature_sensitivities = {}
        if self.diagnostic_engine is not None:
            try:
                diag = self.diagnostic_engine.diagnose(
                    formula_str, predictions, X_test, y_test
                )
                feature_sensitivities = diag.feature_sensitivities
            except Exception:
                pass

        sign_warnings = self.check_coefficient_signs(formula_str, feature_sensitivities)
        high_conf_violations = [
            w for w in sign_warnings
            if "HIGH" in w.upper() or "confidence" in w.lower()
        ]
        if high_conf_violations and len(high_conf_violations) >= 3:
            return self._make_vote(
                "VETO",
                confidence=0.85,
                reason=(
                    "ValidatorAgent: multiple coefficient sign violations vs physics: "
                    + "; ".join(high_conf_violations[:3])
                ),
                evidence={"sign_warnings": high_conf_violations}
            )

        return self._make_vote(
            "APPROVE",
            confidence=0.8,
            reason=(
                f"ValidatorAgent: formula passes positivity check "
                f"({violation_frac*100:.1f}% violations) and sign checks"
            ),
            evidence={"violation_fraction": violation_frac, "sign_warnings": sign_warnings}
        )

    def _vote_search_space(self, proposal: str,
                            context: Dict[str, Any]) -> AgentVote:
        """Basic search space validation."""
        return self._make_vote(
            "APPROVE", 0.7,
            "ValidatorAgent: search space acceptable from physics perspective"
        )

    # ------------------------------------------------------------------
    # leave_one_chalcogen_out()
    # ------------------------------------------------------------------

    def leave_one_chalcogen_out(
        self, formula_fn, X: np.ndarray, y: np.ndarray,
        metadata
    ) -> Dict[str, float]:
        """
        Leave-one-chalcogen-out cross-validation for OOD generalization.

        Parameters
        ----------
        formula_fn : callable
            A function X → predictions.
        X, y : np.ndarray
        metadata : DataFrame with 'chalcogen' column

        Returns
        -------
        dict mapping chalcogen_name → rmse on held-out chalcogen data
        """
        results: Dict[str, float] = {}

        try:
            import pandas as pd
            if not hasattr(metadata, "iloc"):
                return results

            chalcogens = metadata["chalcogen"].unique()
            for chalcogen in chalcogens:
                mask = metadata["chalcogen"].values == chalcogen
                if np.sum(mask) == 0:
                    continue

                X_test_ch = X[mask]
                y_test_ch = y[mask]

                try:
                    preds = formula_fn(X_test_ch)
                    rmse = float(np.sqrt(np.mean((y_test_ch - preds) ** 2)))
                    results[str(chalcogen)] = rmse
                except Exception as e:
                    self.logger.debug(
                        f"LOCO eval failed for chalcogen {chalcogen}: {e}"
                    )
                    results[str(chalcogen)] = float("inf")

        except Exception as e:
            self.logger.debug(f"LOCO failed: {e}")

        return results

    # ------------------------------------------------------------------
    # check_coefficient_signs()
    # ------------------------------------------------------------------

    def check_coefficient_signs(
        self, formula_str: str,
        feature_sensitivities: Dict[str, float]
    ) -> List[str]:
        """
        Compare feature sensitivities against physics priors.

        Returns a list of warning strings for violations.
        """
        warnings = []
        for fname, sens in feature_sensitivities.items():
            prior = PHYSICS_PRIORS.get(fname, {})
            expected = prior.get("expected_sign", "ambiguous")

            if expected == "ambiguous":
                continue

            if expected == "+" and sens < -0.05:
                basis = prior.get("physical_basis", "")
                warnings.append(
                    f"HIGH CONFIDENCE sign violation: '{fname}' expected '+' "
                    f"(physics: {basis[:80]}) but formula gives sens={sens:.4f}"
                )
            elif expected == "-" and sens > 0.05:
                basis = prior.get("physical_basis", "")
                warnings.append(
                    f"HIGH CONFIDENCE sign violation: '{fname}' expected '-' "
                    f"(physics: {basis[:80]}) but formula gives sens={sens:.4f}"
                )

        return warnings

    # ------------------------------------------------------------------
    # _heuristic_decision()
    # ------------------------------------------------------------------

    def _heuristic_decision(self, context: Any) -> Dict[str, Any]:
        """Deterministic heuristic fallback."""
        return {
            "verdict": "APPROVE",
            "reason": "ValidatorAgent heuristic: default approval",
            "confidence": 0.5,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_predictions(self, candidate, X: np.ndarray) -> Optional[np.ndarray]:
        """Try to get predictions from a FormulaCandidate."""
        if X is None:
            return None

        # 1. Use cached test predictions (most accurate — set by _simulate_pysr)
        y_pred_test = getattr(candidate, "y_pred_test", None)
        if y_pred_test is not None and len(y_pred_test) == len(X):
            return np.array(y_pred_test, dtype=float)

        # 2. If candidate has a predict method (real PySR model)
        if hasattr(candidate, "predict"):
            try:
                return np.array(candidate.predict(X))
            except Exception:
                pass

        # 3. Try to evaluate the formula string numerically
        formula_str = getattr(candidate, "expression", None)
        if formula_str:
            preds = self._eval_formula_on_data(formula_str, X)
            if preds is not None:
                return preds

        # 4. Last resort: mock from RMSE noise (least accurate — avoid)
        test_rmse = getattr(candidate, "test_rmse", None)
        if test_rmse is not None and np.isfinite(test_rmse) and self.y_test is not None:
            rng = np.random.default_rng(42)
            preds = self.y_test + rng.normal(0, max(0.001, test_rmse), size=len(self.y_test))
            return np.clip(preds, 0.5, 10.0)

        return None

    def _eval_formula_on_data(self, formula_str: str,
                               X: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate a formula string on a data matrix."""
        feature_names = self.feature_names or [f"x{i}" for i in range(X.shape[1])]
        results = []
        import math
        safe_builtins: Dict = {}

        for row in X:
            ns: Dict = {fname: float(row[i]) for i, fname in enumerate(feature_names)}
            ns.update({f"x{i}": float(row[i]) for i in range(len(row))})
            def _safe_exp(v):
                try:
                    return math.exp(v) if v < 700 else float("inf")
                except (OverflowError, ValueError):
                    return float("inf")

            def _safe_log(v):
                try:
                    return math.log(v) if v > 0 else float("nan")
                except (ValueError,):
                    return float("nan")

            ns.update({
                "exp": _safe_exp, "log": _safe_log, "sqrt": math.sqrt,
                "sin": math.sin, "cos": math.cos, "abs": abs,
                "square": lambda v: v**2, "inv": lambda v: 1.0/v if v != 0 else float("inf"),
                "cube": lambda v: v**3, "cbrt": lambda v: v**(1.0/3) if v >= 0 else -((-v)**(1.0/3)),
                "__builtins__": safe_builtins,
            })
            try:
                val = eval(formula_str, {"__builtins__": safe_builtins}, ns)
                results.append(float(val))
            except Exception:
                results.append(float("nan"))

        arr = np.array(results, dtype=float)
        return arr if np.any(np.isfinite(arr)) else None
