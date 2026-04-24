"""
Formula Evaluator
=================
Evaluates and ranks discovered formulas using a multi-objective scoring
function that balances accuracy, simplicity, and physical validity.
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from src.cauchy_validator import CauchyValidator, ValidationResult

logger = logging.getLogger("HARNESS.Evaluator")


class FormulaEvaluator:
    """Multi-objective formula evaluation and ranking."""

    def __init__(self, constraints_config: dict,
                 X_test: np.ndarray, y_test: np.ndarray):
        self.validator = CauchyValidator(constraints_config)
        self.X_test = X_test
        self.y_test = y_test

        # Scoring weights
        self.weights = {
            'accuracy': 0.35,      # RMSE-based score
            'generalization': 0.25, # Train-test gap
            'simplicity': 0.20,    # Inverse complexity
            'physics': 0.20,       # Physical validation score
        }

    def score_formula(self, expression: str, predictions: np.ndarray,
                      train_rmse: float, test_rmse: float,
                      complexity: int, feature_names: List[str]) -> Dict:
        """
        Compute comprehensive score for a formula.
        
        Returns dict with individual and overall scores.
        """
        # Accuracy score: 1 / (1 + RMSE)
        accuracy_score = 1.0 / (1.0 + test_rmse)

        # Generalization score: penalize large train-test gap
        if train_rmse > 0:
            gen_ratio = test_rmse / max(train_rmse, 1e-8)
            gen_score = 1.0 / (1.0 + max(0, gen_ratio - 1.0))
        else:
            gen_score = 0.5

        # Simplicity score: exponential decay with complexity
        max_complexity = 30
        simplicity_score = np.exp(-0.1 * complexity)

        # Physics validation score
        validation = self.validator.validate(
            expression, predictions, complexity, feature_names
        )
        physics_score = validation.score

        # Overall weighted score
        overall = (
            self.weights['accuracy'] * accuracy_score +
            self.weights['generalization'] * gen_score +
            self.weights['simplicity'] * simplicity_score +
            self.weights['physics'] * physics_score
        )

        return {
            'overall': overall,
            'accuracy': accuracy_score,
            'generalization': gen_score,
            'simplicity': simplicity_score,
            'physics': physics_score,
            'validation': validation,
        }

    def rank_formulas(self, candidates) -> List:
        """Rank a list of FormulaCandidate objects by overall score."""
        scored = []
        for c in candidates:
            try:
                # Generate predictions placeholder (would use actual model in production)
                score_dict = {
                    'overall': 1.0 / (1.0 + c.test_rmse) * (1.0 if c.physics_valid else 0.5),
                    'test_rmse': c.test_rmse,
                    'r2': c.r2,
                    'complexity': c.complexity,
                }
                scored.append((c, score_dict))
            except Exception as e:
                logger.debug(f"Scoring failed for {c.expression}: {e}")

        scored.sort(key=lambda x: x[1]['overall'], reverse=True)
        return scored
