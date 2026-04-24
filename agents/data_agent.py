"""
Data Agent
==========
Gap 3 Fix: Agents are sequential not deliberative.

DataAgent performs statistical analysis of features and can VETO
proposals that violate data quality thresholds.

VETO conditions:
- Feature variance < 0.01 (constant feature)
- Mutual information < 0.005 with target
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

from agents.base_agent import BaseAgent
from src.diagnostic_types import AgentVote

logger = logging.getLogger("HARNESS.DataAgent")

# Thresholds (configurable via config dict)
DEFAULT_VARIANCE_THRESHOLD = 0.01
DEFAULT_MI_THRESHOLD = 0.005
DEFAULT_CORR_THRESHOLD = 0.9   # For multicollinearity veto


class DataAgent(BaseAgent):
    """
    Statistical data analysis agent with veto power on feature quality.

    Performs:
    - Pearson correlation with target
    - Mutual information (sklearn if available, else Spearman fallback)
    - Low-variance detection
    - OOD risk (train vs test distribution shift)
    """

    def __init__(self, name: str, role: str,
                 knowledge=None, config: dict = None):
        super().__init__(name, role, knowledge, config)

        self.variance_threshold = self.config.get(
            "feature_veto_variance_threshold", DEFAULT_VARIANCE_THRESHOLD
        )
        self.mi_threshold = self.config.get(
            "feature_veto_mi_threshold", DEFAULT_MI_THRESHOLD
        )

        # Cache last analysis result for use in vote()
        self._last_analysis: Optional[Dict] = None

    # ------------------------------------------------------------------
    # analyze()
    # ------------------------------------------------------------------

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze features in context.

        Expected context keys:
            X_train, y_train, X_test, y_test, feature_names

        Returns:
            dict with pearson_corr, mutual_info, low_variance_features,
            ood_risk, top_features
        """
        X_train = context.get("X_train")
        y_train = context.get("y_train")
        X_test = context.get("X_test")
        y_test = context.get("y_test")
        feature_names = context.get("feature_names", [])

        if X_train is None or y_train is None:
            self.logger.warning("DataAgent: No data in context; skipping analysis")
            return {}

        result: Dict[str, Any] = {}

        # 1. Pearson correlation
        pearson = {}
        for i, fname in enumerate(feature_names):
            col = X_train[:, i]
            if np.std(col) > 1e-10:
                r = np.corrcoef(col, y_train)[0, 1]
                pearson[fname] = float(r) if np.isfinite(r) else 0.0
            else:
                pearson[fname] = 0.0
        result["pearson_corr"] = pearson

        # 2. Mutual information
        mi = self._compute_mutual_info(X_train, y_train, feature_names)
        result["mutual_info"] = mi

        # 3. Low-variance features
        low_var = []
        variances = {}
        for i, fname in enumerate(feature_names):
            v = float(np.var(X_train[:, i]))
            variances[fname] = v
            if v < self.variance_threshold:
                low_var.append(fname)
        result["low_variance_features"] = low_var
        result["variances"] = variances

        # 4. OOD risk: compare train vs test distributions
        ood_risk = {}
        if X_test is not None:
            for i, fname in enumerate(feature_names):
                train_mean = float(np.mean(X_train[:, i]))
                test_mean = float(np.mean(X_test[:, i]))
                train_std = float(np.std(X_train[:, i]))
                shift = abs(train_mean - test_mean) / (train_std + 1e-10)
                ood_risk[fname] = float(shift)
        result["ood_risk"] = ood_risk

        # 5. Top features by |Pearson|
        top_features = sorted(
            pearson.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        result["top_features"] = [f for f, _ in top_features[:8]]

        # Log summary
        self.logger.info(
            f"DataAgent analysis: {len(low_var)} low-variance features, "
            f"top feature = {top_features[0][0] if top_features else 'N/A'}"
        )

        self._last_analysis = result
        return result

    # ------------------------------------------------------------------
    # vote()
    # ------------------------------------------------------------------

    def vote(self, topic: str, proposal: str,
             context: Dict[str, Any]) -> AgentVote:
        """
        Vote on a proposal. VETO if data quality thresholds are violated.

        Topics handled:
        - "feature_selection": veto low-variance or low-MI features
        - "search_space": approve/modify based on feature statistics
        """
        analysis = self._last_analysis or self.analyze(context)
        feature_names = context.get("feature_names", [])

        if topic == "feature_selection":
            return self._vote_feature_selection(
                proposal, analysis, feature_names, context
            )
        elif topic == "search_space":
            return self._vote_search_space(proposal, analysis, feature_names)
        else:
            # Default: approve with low confidence
            return self._make_vote(
                "APPROVE", 0.5,
                f"DataAgent: no specific analysis for topic '{topic}'"
            )

    def _vote_feature_selection(self, proposal: str,
                                 analysis: Dict, feature_names: List[str],
                                 context: Dict[str, Any]) -> AgentVote:
        """Check proposed features against variance and MI thresholds."""
        low_var = analysis.get("low_variance_features", [])
        mi_scores = analysis.get("mutual_info", {})
        pearson = analysis.get("pearson_corr", {})

        # Parse proposed features from proposal string
        proposed = self._parse_features_from_proposal(proposal, feature_names)

        # Find problematic features in proposal
        veto_reasons = []
        for fname in proposed:
            if fname in low_var:
                veto_reasons.append(
                    f"'{fname}' has variance < {self.variance_threshold} "
                    f"(constant-like feature)"
                )
            mi = mi_scores.get(fname, 1.0)
            if mi < self.mi_threshold:
                veto_reasons.append(
                    f"'{fname}' has MI={mi:.5f} < threshold={self.mi_threshold}"
                )

        if veto_reasons:
            return self._make_vote(
                "VETO",
                confidence=0.85,
                reason="Data quality thresholds violated: " + "; ".join(veto_reasons),
                evidence={
                    "low_variance_features": low_var,
                    "mi_scores": {f: mi_scores.get(f, 0) for f in proposed},
                    "veto_reasons": veto_reasons,
                },
                modification=(
                    f"Remove {veto_reasons} from feature set. "
                    f"Use top features: {analysis.get('top_features', [])[:5]}"
                )
            )

        # Check multicollinearity among proposed features
        multicoll_pairs = self._check_multicollinearity(
            proposed, context.get("X_train"), feature_names
        )
        if multicoll_pairs:
            pair_str = ", ".join(f"({a},{b})" for a, b, _ in multicoll_pairs)
            return self._make_vote(
                "MODIFY",
                confidence=0.75,
                reason=f"Multicollinear feature pairs detected: {pair_str}",
                evidence={"multicollinear_pairs": multicoll_pairs},
                modification=(
                    f"Remove one feature from each correlated pair: {pair_str}"
                )
            )

        return self._make_vote(
            "APPROVE", 0.8,
            f"DataAgent: features {proposed} pass variance and MI thresholds",
            evidence={
                "pearson": {f: pearson.get(f, 0) for f in proposed},
                "mi": {f: mi_scores.get(f, 0) for f in proposed},
            }
        )

    def _vote_search_space(self, proposal: str,
                            analysis: Dict, feature_names: List[str]) -> AgentVote:
        """Approve/modify search space based on feature quality."""
        top = analysis.get("top_features", [])
        low_var = analysis.get("low_variance_features", [])

        if low_var:
            return self._make_vote(
                "MODIFY", 0.7,
                f"Remove {len(low_var)} low-variance features from search space",
                evidence={"low_variance": low_var},
                modification=f"Exclude {low_var} from feature set"
            )

        return self._make_vote(
            "APPROVE", 0.8,
            f"DataAgent: search space acceptable. Recommend top features: {top[:5]}"
        )

    # ------------------------------------------------------------------
    # _heuristic_decision() — no LLM needed
    # ------------------------------------------------------------------

    def _heuristic_decision(self, context: Any) -> Dict[str, Any]:
        """Deterministic heuristic fallback (no LLM required)."""
        return {
            "features": ["atomic_number_X", "bond_polarity", "d_electron_count"],
            "reason": "DataAgent heuristic: physics-informed default features",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_mutual_info(self, X: np.ndarray, y: np.ndarray,
                              feature_names: List[str]) -> Dict[str, float]:
        """Compute mutual information (sklearn if available, else Spearman)."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            mi_vals = mutual_info_regression(X, y, random_state=42)
            return {fname: float(mi_vals[i])
                    for i, fname in enumerate(feature_names)}
        except ImportError:
            pass

        # Fallback: Spearman rank correlation as MI proxy
        mi = {}
        for i, fname in enumerate(feature_names):
            col = X[:, i]
            ranks_x = np.argsort(np.argsort(col)).astype(float)
            ranks_y = np.argsort(np.argsort(y)).astype(float)
            n = len(ranks_x)
            if n > 1:
                d2 = np.sum((ranks_x - ranks_y) ** 2)
                rho = 1 - 6 * d2 / (n * (n**2 - 1))
                # Map Spearman to a rough MI proxy
                mi[fname] = float(max(0, rho**2 * 0.5))
            else:
                mi[fname] = 0.0
        return mi

    def _check_multicollinearity(self, proposed: List[str],
                                  X: Optional[np.ndarray],
                                  feature_names: List[str],
                                  threshold: float = DEFAULT_CORR_THRESHOLD
                                  ) -> List[tuple]:
        """Find feature pairs with |correlation| > threshold."""
        pairs = []
        if X is None or len(proposed) < 2:
            return pairs

        idx_map = {name: i for i, name in enumerate(feature_names)}
        cols = [idx_map[f] for f in proposed if f in idx_map]
        if len(cols) < 2:
            return pairs

        X_sub = X[:, cols]
        corr = np.corrcoef(X_sub.T)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if abs(corr[i, j]) > threshold:
                    pairs.append((proposed[i], proposed[j], float(corr[i, j])))
        return pairs

    def _parse_features_from_proposal(self, proposal: str,
                                       feature_names: List[str]) -> List[str]:
        """Extract feature names mentioned in the proposal string."""
        found = [f for f in feature_names if f in proposal]
        return found if found else feature_names[:5]
