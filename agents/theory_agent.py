"""
Theory Agent
============
Gap 3 Fix: Agents are sequential not deliberative.

TheoryAgent uses physics priors for all 20 TMD features to:
- Recommend operator restrictions based on KnowledgeMemory exclusion rules
- Propose hypothesis skeletons based on success patterns
- VETO multicollinear feature pairs and intractable search spaces
- VETO formulas with sign mismatches vs physics priors

All votes work WITHOUT LLM (heuristic fallback uses PHYSICS_PRIORS dict directly).
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

from agents.base_agent import BaseAgent
from src.diagnostic_types import AgentVote

logger = logging.getLogger("HARNESS.TheoryAgent")

# Full physics priors for all 20 TMD features
PHYSICS_PRIORS = {
    "atomic_number_X": {
        "expected_sign": "+",
        "physical_basis": (
            "Heavier chalcogen → higher polarizability → higher n. "
            "Te (Z=52) gives ~2x higher n than S (Z=16)."
        ),
        "expected_magnitude": "O(0.01-0.03)",
        "nonlinearity": "linear, possible saturation for Te",
    },
    "bond_polarity": {
        "expected_sign": "+",
        "physical_basis": "Ionic character enhances dielectric response",
        "expected_magnitude": "O(0.3-0.8)",
        "nonlinearity": "linear in typical range",
    },
    "d_electron_count": {
        "expected_sign": "+",
        "physical_basis": "d-orbitals contribute to optical transitions",
        "expected_magnitude": "O(0.01-0.02)",
        "nonlinearity": "approximately linear",
    },
    "bond_length_MX": {
        "expected_sign": "-",
        "physical_basis": "Shorter bonds → denser electron cloud → higher n",
        "expected_magnitude": "O(0.05-0.15)",
        "nonlinearity": "possibly inverse",
    },
    "electronegativity_diff": {
        "expected_sign": "+",
        "physical_basis": "Higher polarity → stronger dielectric response",
        "expected_magnitude": "O(0.1-0.4)",
        "nonlinearity": "linear",
    },
    "atomic_number_M": {
        "expected_sign": "ambiguous",
        "physical_basis": (
            "Heavier metal → more electrons → slight polarizability increase, "
            "but Z_M is highly correlated with IE_M and d_electron_count. "
            "In nonlinear SR formulas the net sensitivity sign is expression-dependent."
        ),
        "expected_magnitude": "O(0.001-0.005)",
        "nonlinearity": "weak, sign ambiguous in complex nonlinear expressions",
    },
    "electronegativity_M": {
        "expected_sign": "ambiguous",
        "physical_basis": (
            "Metal electronegativity has competing effects in nonlinear formulas: "
            "lower χ_M → more charge donation → higher n, but χ_M also correlates "
            "with d-band position. Net sign in complex nonlinear expressions is "
            "data-dependent; treat as ambiguous for nonlinear SR."
        ),
        "expected_magnitude": "O(0.05-0.2)",
        "nonlinearity": "nonlinear, sign ambiguous in complex expressions",
    },
    "electronegativity_X": {
        "expected_sign": "ambiguous",
        "physical_basis": (
            "Chalcogen electronegativity correlates with atomic number Z_X which "
            "strongly drives higher n. In nonlinear SR formulas, χ_X and Z_X are "
            "collinear; net sensitivity sign is expression-dependent. Treat as "
            "ambiguous to avoid false veto of valid nonlinear formulas."
        ),
        "expected_magnitude": "O(0.05-0.2)",
        "nonlinearity": "nonlinear, sign ambiguous in complex expressions",
    },
    "atomic_radius_M": {
        "expected_sign": "+",
        "physical_basis": "Larger metal → more diffuse orbitals → higher polarizability",
        "expected_magnitude": "O(0.1-0.5)",
        "nonlinearity": "linear",
    },
    "atomic_radius_X": {
        "expected_sign": "+",
        "physical_basis": "Larger chalcogen → more polarizable electron cloud",
        "expected_magnitude": "O(0.2-0.8)",
        "nonlinearity": "linear, possible saturation",
    },
    "ionization_energy_M": {
        "expected_sign": "-",
        "physical_basis": (
            "Higher IE → tighter electron binding → lower polarizability → lower n"
        ),
        "expected_magnitude": "O(0.05-0.2)",
        "nonlinearity": "weak linear",
    },
    "electron_affinity_X": {
        "expected_sign": "+",
        "physical_basis": (
            "Higher EA → stronger charge acceptance → enhanced polarization"
        ),
        "expected_magnitude": "O(0.05-0.2)",
        "nonlinearity": "weak linear",
    },
    "lattice_constant_a": {
        "expected_sign": "-",
        "physical_basis": "Larger lattice → lower packing density → lower n",
        "expected_magnitude": "O(0.02-0.1)",
        "nonlinearity": "possibly inverse",
    },
    "monolayer_thickness": {
        "expected_sign": "+",
        "physical_basis": "Thicker monolayer → more material → higher effective n",
        "expected_magnitude": "O(0.05-0.2)",
        "nonlinearity": "linear",
    },
    "coordination_number": {
        "expected_sign": "+",
        "physical_basis": (
            "Higher coordination → more bonds → denser electron cloud"
        ),
        "expected_magnitude": "O(0.1-0.3)",
        "nonlinearity": "discrete (typically 6 or 4 for TMDs)",
    },
    "point_group_encoded": {
        "expected_sign": "ambiguous",
        "physical_basis": "Symmetry affects optical activity; sign depends on encoding",
        "expected_magnitude": "O(0.01-0.1)",
        "nonlinearity": "discrete categorical",
    },
    "electronegativity_diff": {
        "expected_sign": "+",
        "physical_basis": "|χ_M - χ_X| drives bond ionicity → dielectric response",
        "expected_magnitude": "O(0.1-0.4)",
        "nonlinearity": "linear",
    },
    "radius_ratio": {
        "expected_sign": "+",
        "physical_basis": (
            "r_M/r_X relates to geometric packing; higher ratio → denser packing"
        ),
        "expected_magnitude": "O(0.1-0.5)",
        "nonlinearity": "possibly nonlinear at extremes",
    },
    "packing_fraction": {
        "expected_sign": "+",
        "physical_basis": "Higher packing → more electron density → higher n",
        "expected_magnitude": "O(0.5-2.0)",
        "nonlinearity": "linear to quadratic",
    },
    # Z_ratio and IE_EA_ratio as derived features
    "Z_ratio": {
        "expected_sign": "+",
        "physical_basis": "Z_X/Z_M ratio captures relative chalcogen/metal contribution",
        "expected_magnitude": "O(0.1-0.5)",
        "nonlinearity": "linear",
    },
    "IE_EA_ratio": {
        "expected_sign": "-",
        "physical_basis": (
            "IE_M/EA_X: high ratio → poor charge transfer → lower ionic polarization"
        ),
        "expected_magnitude": "O(0.05-0.2)",
        "nonlinearity": "linear",
    },
}


class TheoryAgent(BaseAgent):
    """
    Physics-informed theory agent with VETO power on search space design.

    Reads KnowledgeMemory to:
    - Recommend operator restrictions from exclusion_rules
    - Propose hypothesis skeletons from success_patterns

    VETO conditions:
    - Multicollinear feature pairs (|r| > 0.9)
    - Intractable search spaces (n_features * 2^depth > 1000)
    """

    def __init__(self, name: str, role: str,
                 knowledge=None, config: dict = None):
        super().__init__(name, role, knowledge, config)
        self.physics_priors = PHYSICS_PRIORS
        self._last_analysis: Optional[Dict] = None

    # ------------------------------------------------------------------
    # analyze()
    # ------------------------------------------------------------------

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate physics-informed analysis.

        Returns:
            dict with:
                operator_restrictions: list of operators to avoid
                hypothesis_skeletons: list of proposed formula templates
                feature_prior_signs: dict of expected signs per feature
                recommended_features: list of top recommended features
        """
        feature_names = context.get("feature_names", [])
        X_train = context.get("X_train")
        y_train = context.get("y_train")

        result: Dict[str, Any] = {}

        # 1. Operator restrictions from KnowledgeMemory
        op_restrictions = self._get_operator_restrictions()
        result["operator_restrictions"] = op_restrictions

        # 2. Success pattern hypotheses
        skeletons = self._get_hypothesis_skeletons()
        result["hypothesis_skeletons"] = skeletons

        # 3. Expected signs from physics priors
        prior_signs = {
            fname: self.physics_priors.get(fname, {}).get("expected_sign", "ambiguous")
            for fname in feature_names
        }
        result["feature_prior_signs"] = prior_signs

        # 4. Physics-motivated feature recommendations
        recommended = self._recommend_features(feature_names, X_train, y_train)
        result["recommended_features"] = recommended

        # 5. Log analysis
        self.logger.info(
            f"TheoryAgent: {len(op_restrictions)} operator restrictions, "
            f"{len(skeletons)} hypothesis skeletons"
        )

        self._last_analysis = result
        return result

    # ------------------------------------------------------------------
    # vote()
    # ------------------------------------------------------------------

    def vote(self, topic: str, proposal: str,
             context: Dict[str, Any]) -> AgentVote:
        """
        Physics-informed vote on a proposal.

        VETO:
        - Multicollinear feature pairs (|r| > 0.9)
        - Intractable search spaces (n_features * 2^depth > 1000)
        """
        analysis = self._last_analysis or self.analyze(context)
        feature_names = context.get("feature_names", [])
        X_train = context.get("X_train")

        if topic == "feature_selection":
            return self._vote_feature_selection(
                proposal, analysis, feature_names, X_train
            )
        elif topic == "search_space":
            return self._vote_search_space(proposal, context)
        elif topic == "formula_acceptance":
            return self._vote_formula(proposal, context)
        else:
            return self._make_vote(
                "APPROVE", 0.5,
                f"TheoryAgent: no physics constraint for topic '{topic}'"
            )

    def _vote_feature_selection(self, proposal: str, analysis: Dict,
                                 feature_names: List[str],
                                 X_train) -> AgentVote:
        """Veto multicollinear feature pairs."""
        proposed = [f for f in feature_names if f in proposal]
        if not proposed:
            proposed = feature_names

        # Check multicollinearity
        if X_train is not None and len(proposed) >= 2:
            multicoll = self._check_multicollinearity(
                proposed, X_train, feature_names, threshold=0.9
            )
            if multicoll:
                pair_strs = [f"({a},{b}: r={r:.2f})" for a, b, r in multicoll]
                return self._make_vote(
                    "VETO",
                    confidence=0.9,
                    reason=(
                        "TheoryAgent: multicollinear feature pairs violate "
                        "identifiability: " + ", ".join(pair_strs)
                    ),
                    evidence={"multicollinear_pairs": multicoll},
                    modification=(
                        "Remove one feature from each pair; "
                        f"recommended: {analysis.get('recommended_features', [])[:5]}"
                    )
                )

        return self._make_vote(
            "APPROVE", 0.75,
            "TheoryAgent: features are physically motivated and non-collinear"
        )

    def _vote_search_space(self, proposal: str,
                            context: Dict[str, Any]) -> AgentVote:
        """Veto intractable search spaces."""
        feature_names = context.get("feature_names", [])
        n_features = len(feature_names)

        # Estimate max_depth from proposal
        import re
        depth_match = re.search(r"depth[=:\s]+(\d+)", proposal, re.IGNORECASE)
        max_depth = int(depth_match.group(1)) if depth_match else 5

        # Intractability check: n_features * 2^depth > 1000
        complexity_estimate = n_features * (2 ** max_depth)
        if complexity_estimate > 1000:
            return self._make_vote(
                "VETO",
                confidence=0.85,
                reason=(
                    f"TheoryAgent: search space too large "
                    f"({n_features} features × 2^{max_depth} = {complexity_estimate} > 1000). "
                    f"Risk of overfitting and slow convergence."
                ),
                evidence={"n_features": n_features, "max_depth": max_depth,
                           "complexity_estimate": complexity_estimate},
                modification=f"Reduce to depth={max(3, max_depth-2)} or features≤{min(n_features, 5)}"
            )

        return self._make_vote(
            "APPROVE", 0.8,
            f"TheoryAgent: search space size {complexity_estimate} is tractable"
        )

    def _vote_formula(self, formula_str: str,
                       context: Dict[str, Any]) -> AgentVote:
        """Check if formula coefficient signs match physics priors."""
        feature_sensitivities = context.get("feature_sensitivities", {})
        if not feature_sensitivities:
            return self._make_vote(
                "APPROVE", 0.5, "TheoryAgent: no sensitivity data available for sign check"
            )

        sign_violations = []
        for fname, sens in feature_sensitivities.items():
            prior = self.physics_priors.get(fname, {})
            expected = prior.get("expected_sign", "ambiguous")
            if expected == "+":
                if sens < -0.01:
                    sign_violations.append(
                        f"'{fname}': physics expects + but formula gives - (sens={sens:.4f})"
                    )
            elif expected == "-":
                if sens > 0.01:
                    sign_violations.append(
                        f"'{fname}': physics expects - but formula gives + (sens={sens:.4f})"
                    )

        if len(sign_violations) >= 2:
            return self._make_vote(
                "VETO",
                confidence=0.85,
                reason=(
                    "TheoryAgent: coefficient signs contradict physics priors "
                    "with high confidence: " + "; ".join(sign_violations[:3])
                ),
                evidence={"sign_violations": sign_violations}
            )
        elif sign_violations:
            return self._make_vote(
                "MODIFY",
                confidence=0.7,
                reason="TheoryAgent: some coefficient signs may contradict physics",
                evidence={"sign_violations": sign_violations},
                modification="Check formula for sign conventions; consider flipping"
            )

        return self._make_vote(
            "APPROVE", 0.85,
            "TheoryAgent: coefficient signs consistent with physics priors"
        )

    # ------------------------------------------------------------------
    # _heuristic_decision()
    # ------------------------------------------------------------------

    def _heuristic_decision(self, context: Any) -> Dict[str, Any]:
        """Deterministic physics-based heuristic (no LLM needed)."""
        return {
            "hypothesis": (
                "n = α*Z_X + β*bond_polarity + γ*d_electrons + const "
                "(linear model motivated by polarizability theory)"
            ),
            "skeleton": "linear_ZX_polarity_d",
            "operator_restrictions": ["log", "exp"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_operator_restrictions(self) -> List[str]:
        """Get operator restrictions from KnowledgeMemory exclusion rules."""
        restrictions = []
        if self.knowledge is None:
            return restrictions

        try:
            for entry in self.knowledge.exclusion_rules:
                if entry.confidence >= 0.5 and entry.pattern in (
                    "log", "exp", "sqrt", "sin", "cos", "inv"
                ):
                    restrictions.append(entry.pattern)
        except Exception as e:
            self.logger.debug(f"Knowledge query failed: {e}")

        return restrictions

    def _get_hypothesis_skeletons(self) -> List[str]:
        """Get hypothesis skeletons from KnowledgeMemory success patterns."""
        skeletons = []
        if self.knowledge is None:
            # Default physics-motivated skeletons
            return [
                "α*atomic_number_X + β*bond_polarity + γ*d_electron_count + const",
                "α*atomic_number_X + β*electronegativity_diff + const",
            ]

        try:
            for entry in sorted(
                self.knowledge.success_patterns,
                key=lambda e: e.metadata.get("best_r2", 0), reverse=True
            )[:3]:
                if entry.formulas:
                    skeletons.append(entry.formulas[-1])
        except Exception as e:
            self.logger.debug(f"Knowledge query failed: {e}")

        # Add default if empty
        if not skeletons:
            skeletons = [
                "α*atomic_number_X + β*bond_polarity + γ*d_electron_count + const"
            ]

        return skeletons

    def _recommend_features(self, feature_names: List[str],
                             X: Optional[np.ndarray],
                             y: Optional[np.ndarray]) -> List[str]:
        """
        Recommend features based on physics priors + data correlation.
        """
        # Physics-based priority order
        physics_priority = [
            "atomic_number_X", "bond_polarity", "d_electron_count",
            "electronegativity_diff", "atomic_number_M", "bond_length_MX",
            "atomic_radius_X", "monolayer_thickness", "radius_ratio",
            "packing_fraction",
        ]

        # Start with physics priority
        recommended = [f for f in physics_priority if f in feature_names]

        # Add remaining features not in priority list
        remaining = [f for f in feature_names if f not in recommended]
        recommended.extend(remaining)

        return recommended[:8]

    def _check_multicollinearity(self, proposed: List[str],
                                  X: np.ndarray,
                                  feature_names: List[str],
                                  threshold: float = 0.9) -> List[tuple]:
        """Check for multicollinearity in proposed feature set."""
        pairs = []
        idx_map = {name: i for i, name in enumerate(feature_names)}
        cols = [idx_map[f] for f in proposed if f in idx_map]
        if len(cols) < 2:
            return pairs

        X_sub = X[:, cols]
        try:
            corr = np.corrcoef(X_sub.T)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if abs(corr[i, j]) > threshold:
                        pairs.append((proposed[i], proposed[j], float(corr[i, j])))
        except Exception:
            pass

        return pairs
