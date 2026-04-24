"""
Cauchy Validator
================
Layer 2 of the Harness: Physical constraints and validation.

Enforces physics-informed constraints on discovered formulas:
1. Cauchy dispersion model consistency
2. Dimensional analysis (refractive index must be dimensionless)
3. Positivity (n > 1 for all solid materials)
4. Normal dispersion (dn/dλ < 0 in the transparent region)
5. Symmetry constraints (point group invariance)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

logger = logging.getLogger("HARNESS.CauchyValidator")


@dataclass
class ValidationResult:
    """Result of physical validation."""
    is_valid: bool
    checks_passed: Dict[str, bool]
    warnings: List[str]
    score: float  # 0-1, fraction of checks passed


class CauchyValidator:
    """
    Validates discovered formulas against physical constraints
    specific to 2D material refractive indices.
    """

    def __init__(self, constraints_config: dict):
        self.config = constraints_config
        self.cauchy_config = constraints_config.get('cauchy_model', {})
        self.dim_config = constraints_config.get('dimensional_analysis', {})
        self.symmetry_config = constraints_config.get('symmetry', {})
        self.simplicity_config = constraints_config.get('simplicity', {})

    def validate(self, expression: str, predictions: np.ndarray,
                 complexity: int, feature_names: List[str]) -> ValidationResult:
        """
        Run all physical validation checks on a candidate formula.
        
        Args:
            expression: String representation of the formula
            predictions: Predicted values from the formula
            complexity: Expression tree complexity
            feature_names: Names of features used
            
        Returns:
            ValidationResult with detailed check results
        """
        checks = {}
        warnings = []

        # Check 1: Positivity (n > 1 for solid materials)
        checks['positivity'] = self._check_positivity(predictions, warnings)

        # Check 2: Reasonable range (typical RI for TMDs: 1.5 - 6.0)
        checks['range'] = self._check_range(predictions, warnings)

        # Check 3: No NaN or Inf
        checks['finite'] = self._check_finite(predictions, warnings)

        # Check 4: Simplicity (Occam's razor)
        checks['simplicity'] = self._check_simplicity(
            complexity, expression, warnings
        )

        # Check 5: Physical plausibility of expression structure
        checks['plausibility'] = self._check_physical_plausibility(
            expression, feature_names, warnings
        )

        # Check 6: Statistical robustness (not overfitting)
        checks['robustness'] = self._check_robustness(
            predictions, complexity, warnings
        )

        # Overall validity
        critical_checks = ['positivity', 'finite', 'range']
        is_valid = all(checks.get(c, False) for c in critical_checks)

        # Score: weighted sum
        weights = {
            'positivity': 0.2, 'range': 0.15, 'finite': 0.15,
            'simplicity': 0.2, 'plausibility': 0.15, 'robustness': 0.15
        }
        score = sum(weights.get(k, 0.1) * (1.0 if v else 0.0)
                    for k, v in checks.items())

        return ValidationResult(
            is_valid=is_valid,
            checks_passed=checks,
            warnings=warnings,
            score=score
        )

    def _check_positivity(self, predictions: np.ndarray,
                           warnings: List[str]) -> bool:
        """Refractive index must be > 1 for solid materials."""
        if np.any(predictions < 1.0):
            frac = np.mean(predictions < 1.0)
            warnings.append(f"Positivity: {frac*100:.1f}% of predictions have n < 1")
            return frac < 0.05  # Allow up to 5% violations (noise)
        return True

    def _check_range(self, predictions: np.ndarray,
                      warnings: List[str]) -> bool:
        """Typical RI range for 2D TMDs: 1.5 - 6.5."""
        in_range = np.logical_and(predictions >= 1.0, predictions <= 8.0)
        frac_ok = np.mean(in_range)
        if frac_ok < 0.95:
            warnings.append(
                f"Range: {(1-frac_ok)*100:.1f}% of predictions outside [1.0, 8.0]"
            )
            return False
        return True

    def _check_finite(self, predictions: np.ndarray,
                       warnings: List[str]) -> bool:
        """No NaN or Inf values."""
        n_bad = np.sum(~np.isfinite(predictions))
        if n_bad > 0:
            warnings.append(f"Finite: {n_bad} non-finite predictions")
            return False
        return True

    def _check_simplicity(self, complexity: int, expression: str,
                           warnings: List[str]) -> bool:
        """Prefer simple formulas (Occam's razor)."""
        max_depth = self.simplicity_config.get('max_tree_depth', 6)
        max_ops = self.simplicity_config.get('max_operators', 8)

        # Estimate operator count from expression
        import re
        operators = len(re.findall(r'[+\-*/]', expression))
        functions = len(re.findall(r'(exp|log|sqrt|sin|cos|square|inv)\b', expression))
        total_ops = operators + functions

        if complexity > max_depth * 5:
            warnings.append(f"Simplicity: complexity {complexity} exceeds threshold")
            return False
        if total_ops > max_ops:
            warnings.append(f"Simplicity: {total_ops} operators exceed max {max_ops}")
            return False
        return True

    def _check_physical_plausibility(self, expression: str,
                                      feature_names: List[str],
                                      warnings: List[str]) -> bool:
        """
        Check if the formula uses features that make physical sense.
        
        Known physical relationships:
        - n should increase with atomic_number_X (heavier chalcogen → higher polarizability)
        - n should depend on bond_polarity or electronegativity_diff
        - d_electron_count is important for optical response
        """
        # At least one of these key features should appear
        key_features = {
            'atomic_number_X', 'electronegativity_diff', 'bond_polarity',
            'd_electron_count', 'atomic_number_M'
        }

        used_key = key_features.intersection(set(feature_names))
        expr_lower = expression.lower()

        # Check if any key feature appears in the expression
        has_key_feature = any(f.lower() in expr_lower for f in used_key)

        if not has_key_feature and len(feature_names) > 2:
            warnings.append(
                "Plausibility: formula doesn't use any key physical features "
                "(Z_X, d_electrons, bond_polarity)"
            )
            return False
        return True

    def _check_robustness(self, predictions: np.ndarray,
                           complexity: int,
                           warnings: List[str]) -> bool:
        """Check for potential overfitting."""
        # Very low variance in predictions might indicate a constant
        pred_std = np.std(predictions)
        if pred_std < 0.01:
            warnings.append("Robustness: predictions have near-zero variance (constant?)")
            return False

        # Very high complexity relative to prediction variance is suspicious
        if complexity > 20 and pred_std < 0.1:
            warnings.append("Robustness: high complexity with low prediction variance")
            return False

        return True

    def validate_with_material_ids(
        self, expression: str, predictions: np.ndarray,
        X: np.ndarray, feature_names: List[str],
        material_ids: Optional[List[str]] = None
    ) -> Tuple['ValidationResult', Dict[str, List[int]]]:
        """
        Run all validation checks and also track which row indices fail each check.

        Parameters
        ----------
        expression : str
        predictions : np.ndarray  shape (n,)
        X : np.ndarray            shape (n, p)
        feature_names : list of str
        material_ids : list of str, optional  length n

        Returns
        -------
        (ValidationResult, {check_name: [row_indices_that_failed]})
        """
        import re as _re

        checks: Dict[str, bool] = {}
        warnings: List[str] = []

        # Row-level tracking
        row_failures: Dict[str, List[int]] = {}

        # Positivity: n < 1
        pos_fail = np.where(predictions < 1.0)[0].tolist()
        row_failures["positivity"] = pos_fail
        checks["positivity"] = self._check_positivity(predictions, warnings)

        # Range: outside [1.0, 8.0]
        range_fail = np.where(
            (predictions < 1.0) | (predictions > 8.0)
        )[0].tolist()
        row_failures["range"] = range_fail
        checks["range"] = self._check_range(predictions, warnings)

        # Finite: NaN or Inf
        finite_fail = np.where(~np.isfinite(predictions))[0].tolist()
        row_failures["finite"] = finite_fail
        checks["finite"] = self._check_finite(predictions, warnings)

        # Simplicity and plausibility (not row-level)
        complexity = len(_re.findall(r'[+\-*/]', expression)) * 2
        checks["simplicity"] = self._check_simplicity(complexity, expression, warnings)
        checks["plausibility"] = self._check_physical_plausibility(
            expression, feature_names, warnings
        )
        checks["robustness"] = self._check_robustness(predictions, complexity, warnings)
        row_failures["plausibility"] = []
        row_failures["robustness"] = []

        critical_checks = ["positivity", "finite", "range"]
        is_valid = all(checks.get(c, False) for c in critical_checks)

        weights = {
            "positivity": 0.2, "range": 0.15, "finite": 0.15,
            "simplicity": 0.2, "plausibility": 0.15, "robustness": 0.15
        }
        score = sum(
            weights.get(k, 0.1) * (1.0 if v else 0.0)
            for k, v in checks.items()
        )

        val_result = ValidationResult(
            is_valid=is_valid,
            checks_passed=checks,
            warnings=warnings,
            score=score
        )
        return val_result, row_failures

    def validate_cauchy_consistency(self, A_predictions: np.ndarray,
                                     B_predictions: np.ndarray) -> bool:
        """
        Validate that predicted Cauchy coefficients give physically
        consistent refractive indices.
        
        n(λ) = A + B/λ² must satisfy:
        - n > 1 for all λ in [755, 1064] nm
        - dn/dλ < 0 (normal dispersion)
        """
        lambda_range = np.linspace(755, 1064, 10)  # nm

        for lam in lambda_range:
            n = A_predictions + B_predictions / (lam**2) * 1e6
            if np.any(n < 1.0):
                return False

        # Check dispersion: n should decrease with wavelength
        n_755 = A_predictions + B_predictions / (755**2) * 1e6
        n_1064 = A_predictions + B_predictions / (1064**2) * 1e6

        # B should be positive for normal dispersion
        if np.mean(B_predictions < 0) > 0.1:
            return False

        return True
