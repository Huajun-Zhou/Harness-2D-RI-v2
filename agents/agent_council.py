"""
Agent Council
=============
Orchestrates the full multi-agent deliberation cycle.

Provides two main methods:
- pre_search_deliberation(round_id, context) → search_instructions dict
- post_search_deliberation(candidates, context) → (accepted, diagnostic_reports)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

from src.diagnostic_types import DeliberationResult

logger = logging.getLogger("HARNESS.AgentCouncil")


class AgentCouncil:
    """
    Coordinates DataAgent, TheoryAgent, ValidatorAgent, and MetaAgent
    for pre-search and post-search deliberation.

    Parameters
    ----------
    data_agent : DataAgent
    theory_agent : TheoryAgent
    validator_agent : ValidatorAgent
    meta_agent : MetaAgent
    knowledge : KnowledgeMemory, optional
    """

    def __init__(self, data_agent, theory_agent, validator_agent,
                 meta_agent, knowledge=None):
        self.data_agent = data_agent
        self.theory_agent = theory_agent
        self.validator_agent = validator_agent
        self.meta_agent = meta_agent
        self.knowledge = knowledge

        self._pre_analyses: Dict[int, Dict] = {}
        self._deliberation_history: List[DeliberationResult] = []

    # ------------------------------------------------------------------
    # Pre-search deliberation
    # ------------------------------------------------------------------

    def pre_search_deliberation(self, round_id: int,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run pre-search deliberation to produce search instructions.

        Steps:
        1. DataAgent analyzes feature quality
        2. TheoryAgent recommends physics-motivated features and operators
        3. MetaAgent deliberates on "feature_selection" and "search_space"
        4. Synthesize into search instructions dict

        Parameters
        ----------
        round_id : int
        context : dict
            Must contain: X_train, y_train, X_test, y_test, feature_names

        Returns
        -------
        dict with keys: features, hypothesis, max_depth, operator_restrictions,
                       knowledge_block, causal_feedback
        """
        logger.info(f"[AgentCouncil] Pre-search deliberation for round {round_id}")

        # Step 1: DataAgent analysis
        data_analysis = {}
        try:
            data_analysis = self.data_agent.analyze(context)
        except Exception as e:
            logger.warning(f"DataAgent analysis failed: {e}")

        # Step 2: TheoryAgent analysis
        theory_analysis = {}
        try:
            theory_analysis = self.theory_agent.analyze(context)
        except Exception as e:
            logger.warning(f"TheoryAgent analysis failed: {e}")

        # Step 3: Merge into feature proposal
        top_data_features = data_analysis.get("top_features", [])
        top_theory_features = theory_analysis.get("recommended_features", [])

        # Union of top features (data + theory), deduplicated, max 6
        all_top = list(dict.fromkeys(top_data_features[:3] + top_theory_features[:3]))
        feature_proposal = ", ".join(all_top[:6])

        # Step 4: Deliberate on feature selection
        feature_result = DeliberationResult(
            topic="feature_selection", proposal=feature_proposal,
            final_decision="APPROVE", winning_vote="APPROVE"
        )
        try:
            feature_result = self.meta_agent.deliberate(
                "feature_selection", feature_proposal, context
            )
            self._deliberation_history.append(feature_result)
        except Exception as e:
            logger.warning(f"Feature deliberation failed: {e}")

        # Step 5: Determine final feature set
        if feature_result.final_decision == "VETO":
            # Fall back to safe default features
            features = self._safe_default_features(
                context.get("feature_names", [])
            )
            logger.info(
                f"[AgentCouncil] Feature VETO'd; using safe defaults: {features}"
            )
        elif feature_result.final_decision == "MODIFY" and feature_result.modifications:
            # Use modified features from first modification suggestion
            features = self._parse_features_from_modification(
                feature_result.modifications[0],
                context.get("feature_names", []),
                all_top
            )
        else:
            features = all_top[:6] if all_top else self._safe_default_features(
                context.get("feature_names", [])
            )

        # Step 6: Deliberate on search space
        max_depth = 5  # default
        space_proposal = f"features={len(features)}, depth={max_depth}"
        try:
            space_result = self.meta_agent.deliberate(
                "search_space", space_proposal, context
            )
            if space_result.final_decision == "VETO":
                max_depth = max(3, max_depth - 2)
                features = features[:min(5, len(features))]
                logger.info(
                    f"[AgentCouncil] Search space VETO'd; "
                    f"reduced to depth={max_depth}, features={features}"
                )
        except Exception as e:
            logger.debug(f"Search space deliberation failed: {e}")

        # Step 7: Gather operator restrictions
        op_restrictions = theory_analysis.get("operator_restrictions", [])

        # Step 8: Knowledge block for LLM prompt
        knowledge_block = ""
        if self.knowledge is not None:
            try:
                knowledge_block = self.knowledge.format_for_llm_prompt()
            except Exception:
                pass

        # Step 9: Hypothesis from theory agent
        skeletons = theory_analysis.get("hypothesis_skeletons", [])
        hypothesis = skeletons[0] if skeletons else (
            "Linear combination of Z_X, bond_polarity, d_electron_count"
        )

        instructions = {
            "features": features,
            "hypothesis": hypothesis,
            "max_depth": max_depth,
            "iterations": 100,
            "operator_restrictions": op_restrictions,
            "knowledge_block": knowledge_block,
            "deliberation_summary": self._summarize_deliberation(feature_result),
        }

        self._pre_analyses[round_id] = {
            "data_analysis": data_analysis,
            "theory_analysis": theory_analysis,
            "instructions": instructions,
        }

        logger.info(
            f"[AgentCouncil] Pre-search done: features={features}, "
            f"max_depth={max_depth}"
        )
        return instructions

    # ------------------------------------------------------------------
    # Post-search deliberation
    # ------------------------------------------------------------------

    def post_search_deliberation(
        self, candidates: List, context: Dict[str, Any]
    ) -> Tuple[List, List]:
        """
        Validate and filter candidates after search.

        Parameters
        ----------
        candidates : list of FormulaCandidate
        context : dict
            Must contain: X_test, y_test, feature_names

        Returns
        -------
        (accepted_candidates, diagnostic_reports)
        """
        logger.info(
            f"[AgentCouncil] Post-search deliberation: "
            f"{len(candidates)} candidates"
        )

        accepted = []
        diagnostic_reports = []

        for candidate in candidates:
            formula = getattr(candidate, "expression", str(candidate))

            # Build per-candidate context
            cand_context = dict(context)
            cand_context["candidate"] = candidate

            # ValidatorAgent primary check
            try:
                verdict = self.validator_agent.vote(
                    "formula_acceptance", formula, cand_context
                )
            except Exception as e:
                logger.debug(f"Validator vote failed for {formula}: {e}")
                verdict = None

            # Collect diagnostic if available
            if (self.validator_agent.diagnostic_engine is not None and
                    context.get("X_test") is not None):
                try:
                    X_test = context["X_test"]
                    y_test = context.get("y_test")
                    predictions = self.validator_agent._get_predictions(
                        candidate, X_test
                    )
                    if predictions is not None:
                        diag = self.validator_agent.diagnostic_engine.diagnose(
                            formula, predictions, X_test, y_test
                        )
                        diagnostic_reports.append(diag)
                except Exception as e:
                    logger.debug(f"Diagnostic failed for {formula}: {e}")

            # Accept/reject
            if verdict is None or verdict.vote in ("APPROVE", "ABSTAIN", "MODIFY"):
                accepted.append(candidate)
            else:
                logger.info(
                    f"[AgentCouncil] Rejected '{formula[:60]}': "
                    f"{verdict.reason[:120]}"
                )

        logger.info(
            f"[AgentCouncil] Post-search: {len(accepted)}/{len(candidates)} accepted"
        )
        return accepted, diagnostic_reports

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_default_features(self, feature_names: List[str]) -> List[str]:
        """Return safe physics-motivated default features."""
        defaults = [
            "atomic_number_X", "bond_polarity", "d_electron_count",
            "electronegativity_diff", "atomic_number_M",
        ]
        return [f for f in defaults if f in feature_names] or feature_names[:5]

    def _parse_features_from_modification(
        self, modification: str, feature_names: List[str],
        fallback: List[str]
    ) -> List[str]:
        """Extract feature names from a modification suggestion string."""
        if not modification:
            return fallback

        found = [f for f in feature_names if f in modification]
        return found[:6] if found else fallback[:6]

    def _summarize_deliberation(self, result: DeliberationResult) -> str:
        """Short summary of a deliberation result."""
        return (
            f"{result.topic}: {result.final_decision} "
            f"(conf={result.confidence:.2f}, rounds={result.debate_rounds})"
        )
