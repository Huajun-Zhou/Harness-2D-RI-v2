"""
Adaptive Search Environment
============================
Gap 4 Fix: Environment is static not adaptive.

Tracks operator success/failure across rounds and dynamically adjusts:
- Removes operators that fail 3 consecutive rounds
- Adds fractional powers if power-laws dominate the Pareto front
- Adjusts search depth based on performance history

Usage:
    env = AdaptiveSearchEnvironment(base_config, knowledge)
    config = env.get_current_config(round_id, theory_recommendations)
    # ... run PySR with config ...
    env.record_round_outcome(round_id, config, candidates, diagnostics)
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("HARNESS.AdaptiveSearchEnvironment")


class AdaptiveSearchEnvironment:
    """
    Dynamically adapts the PySR search configuration across rounds.

    Parameters
    ----------
    base_config : dict
        The llm_sr section of harness_config.yaml.
    knowledge : KnowledgeMemory, optional
        Knowledge memory for exclusion rules.
    adaptive_config : dict, optional
        The adaptive_search section of harness_config.yaml.
    """

    def __init__(self, base_config: dict, knowledge=None,
                 adaptive_config: dict = None):
        self.base_config = copy.deepcopy(base_config)
        self.knowledge = knowledge
        self.adaptive_config = adaptive_config or {}

        self.failure_threshold = self.adaptive_config.get(
            "operator_failure_threshold", 3
        )
        self.min_trials = self.adaptive_config.get("min_operator_trials", 2)
        self.depth_reduction = self.adaptive_config.get(
            "depth_reduction_on_timeout", 1
        )
        self.power_law_threshold = self.adaptive_config.get(
            "power_law_detection_threshold", 0.5
        )

        # Operator tracking: op_name → list of bool (True=success per round)
        self._operator_history: Dict[str, List[bool]] = {}
        # Round configs used
        self._round_configs: Dict[int, dict] = {}
        # Currently disabled operators
        self._disabled_operators: List[str] = []
        # Fractional powers flag
        self._use_fractional_powers: bool = False
        # Round outcome summaries
        self._round_outcomes: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_config(self, round_id: int,
                            theory_recommendations: Optional[Dict] = None
                            ) -> Dict:
        """
        Build the PySR config for this round based on adaptation history.

        Parameters
        ----------
        round_id : int
        theory_recommendations : dict, optional
            Suggestions from TheoryAgent (e.g. operator restrictions, depth).

        Returns
        -------
        dict
            A modified copy of the base pysr config.
        """
        config = copy.deepcopy(self.base_config.get("pysr", {}))

        # Apply knowledge-based exclusions from KnowledgeMemory
        if self.knowledge is not None:
            excluded = self._get_knowledge_excluded_operators()
            for op in excluded:
                if op not in self._disabled_operators:
                    self._disabled_operators.append(op)
                    logger.info(
                        f"[AdaptiveEnv] Disabling operator '{op}' "
                        f"based on knowledge exclusion rules"
                    )

        # Remove disabled operators
        unary_ops = list(config.get("unary_operators", []))
        binary_ops = list(config.get("binary_operators", []))
        for op in self._disabled_operators:
            if op in unary_ops:
                unary_ops.remove(op)
                logger.info(f"[AdaptiveEnv] Removed unary op: {op}")
            if op in binary_ops:
                binary_ops.remove(op)
                logger.info(f"[AdaptiveEnv] Removed binary op: {op}")
        config["unary_operators"] = unary_ops
        config["binary_operators"] = binary_ops

        # Add fractional powers if power-laws dominate
        if self._use_fractional_powers:
            extra = ["cube", "cbrt"]
            for op in extra:
                if op not in config.get("unary_operators", []):
                    config.setdefault("unary_operators", []).append(op)
            logger.info("[AdaptiveEnv] Added fractional/cube power operators")

        # Apply theory agent recommendations
        if theory_recommendations:
            if "max_depth" in theory_recommendations:
                config["maxsize"] = theory_recommendations["max_depth"] * 5
            if "operators_to_restrict" in theory_recommendations:
                for op in theory_recommendations["operators_to_restrict"]:
                    if op in config.get("unary_operators", []):
                        config["unary_operators"].remove(op)
            if "operators_to_add" in theory_recommendations:
                for op in theory_recommendations["operators_to_add"]:
                    if op not in config.get("unary_operators", []):
                        config.setdefault("unary_operators", []).append(op)

        # Ensure at least '+' remains
        if not config.get("binary_operators"):
            config["binary_operators"] = ["+", "*"]

        self._round_configs[round_id] = config
        logger.info(
            f"[AdaptiveEnv] Round {round_id} config: "
            f"unary={config.get('unary_operators')}, "
            f"binary={config.get('binary_operators')}"
        )
        return config

    def record_round_outcome(self, round_id: int, config_used: dict,
                              candidates: List, diagnostic_reports: List) -> None:
        """
        Record the outcome of a search round to inform future adaptation.

        Parameters
        ----------
        round_id : int
        config_used : dict
            The PySR config that was actually used.
        candidates : list of FormulaCandidate
            Candidates found this round.
        diagnostic_reports : list of DiagnosticReport
            Diagnostic reports for the candidates.
        """
        if not candidates:
            success = False
            best_rmse = float('inf')
            r2 = 0.0
        else:
            # Sort by test_rmse
            sorted_cands = sorted(
                candidates,
                key=lambda c: getattr(c, "test_rmse", float("inf"))
            )
            best = sorted_cands[0]
            best_rmse = getattr(best, "test_rmse", float("inf"))
            r2 = getattr(best, "r2", 0.0)
            # Success = produced at least one non-trivial accepted candidate
            # Use relative threshold: R² > 0.1 OR complexity > 1 (non-constant)
            success = (r2 > 0.1 or (getattr(best, "complexity", 1) > 1
                                     and best_rmse < float('inf')))

        # Track operator outcomes
        unary_ops = config_used.get("unary_operators", [])
        binary_ops = config_used.get("binary_operators", [])
        all_ops = unary_ops + binary_ops

        for op in all_ops:
            self._operator_history.setdefault(op, []).append(success)

        # Check if any operators should now be disabled
        for op in list(all_ops):
            should_remove, reason = self._should_remove_operator(op)
            if should_remove and op not in self._disabled_operators:
                self._disabled_operators.append(op)
                logger.info(
                    f"[AdaptiveEnv] Auto-disabling op '{op}': {reason}"
                )
                # Record in knowledge memory
                if self.knowledge is not None:
                    try:
                        formula = (
                            candidates[0].expression if candidates else "unknown"
                        )
                        self.knowledge.record_exclusion(
                            op, reason, round_id, formula
                        )
                    except Exception as e:
                        logger.debug(f"Knowledge record failed: {e}")

        # Check if we should add fractional powers
        if not self._use_fractional_powers:
            self._use_fractional_powers = self._should_add_fractional_powers(
                candidates
            )

        # Store outcome summary
        self._round_outcomes.append({
            "round_id": round_id,
            "success": success,
            "best_rmse": float(best_rmse),
            "r2": float(r2),
            "n_candidates": len(candidates),
            "n_violations": sum(
                len(d.material_violations) for d in diagnostic_reports
            ),
        })

    # ------------------------------------------------------------------
    # Adaptation logic
    # ------------------------------------------------------------------

    def _should_remove_operator(self, op: str) -> Tuple[bool, str]:
        """
        Determine if an operator should be removed.

        Returns (True, reason) if the last `failure_threshold` rounds
        using this operator all failed.
        """
        history = self._operator_history.get(op, [])
        if len(history) < self.min_trials:
            return False, ""

        recent = history[-self.failure_threshold:]
        if len(recent) >= self.failure_threshold and not any(recent):
            reason = (
                f"Operator '{op}' failed in last "
                f"{self.failure_threshold} consecutive rounds"
            )
            return True, reason

        return False, ""

    def _should_add_fractional_powers(self, candidates: List) -> bool:
        """
        Return True if power-law relationships dominate the Pareto front.

        Heuristic: if > power_law_threshold fraction of top candidates
        contain '**' or 'sqrt' or 'square', assume power laws are useful.
        """
        if not candidates:
            return False

        top_k = candidates[:max(3, len(candidates) // 2)]
        power_law_count = sum(
            1 for c in top_k
            if any(kw in getattr(c, "expression", "")
                   for kw in ("**", "sqrt", "square", "^"))
        )
        fraction = power_law_count / len(top_k)
        if fraction >= self.power_law_threshold:
            logger.info(
                f"[AdaptiveEnv] Power laws dominate Pareto front "
                f"({fraction:.0%}); will add fractional powers next round"
            )
            return True
        return False

    def _get_knowledge_excluded_operators(self) -> List[str]:
        """
        Query KnowledgeMemory for operators to exclude based on exclusion rules.
        """
        excluded = []
        if self.knowledge is None:
            return excluded

        op_names = {"log", "exp", "sqrt", "sin", "cos", "inv", "square"}
        threshold = self.knowledge.exclusion_confidence_threshold

        try:
            for entry in self.knowledge.exclusion_rules:
                if entry.confidence >= threshold:
                    # entry.pattern might be bare op name like "log" or "exp"
                    if entry.pattern in op_names:
                        excluded.append(entry.pattern)
        except Exception as e:
            logger.debug(f"Knowledge query failed: {e}")

        return excluded

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the adaptive environment state."""
        return {
            "disabled_operators": self._disabled_operators,
            "use_fractional_powers": self._use_fractional_powers,
            "operator_history": {
                op: {"total": len(hist), "successes": sum(hist)}
                for op, hist in self._operator_history.items()
            },
            "rounds_completed": len(self._round_outcomes),
        }
