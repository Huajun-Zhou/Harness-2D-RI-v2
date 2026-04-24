"""
Meta Agent
==========
Gap 3 Fix: Agents are sequential not deliberative.

Implements the debate protocol:
1. Collect votes from all agents
2. If split, run up to 3 rounds of argument exchange
3. Majority vote
4. Physics tiebreak if still tied (ValidatorAgent decides)
5. Synthesize MODIFY if needed

deliberate(topic, proposal, context) → DeliberationResult
"""

import logging
from typing import Dict, List, Any, Optional

from agents.base_agent import BaseAgent
from src.diagnostic_types import AgentVote, DeliberationResult

logger = logging.getLogger("HARNESS.MetaAgent")


class MetaAgent(BaseAgent):
    """
    Orchestrates multi-agent deliberation via a structured debate protocol.

    Parameters
    ----------
    agents : list of BaseAgent
        The agents participating in deliberation.
    """

    def __init__(self, name: str, role: str,
                 knowledge=None, config: dict = None,
                 agents: Optional[List[BaseAgent]] = None):
        super().__init__(name, role, knowledge, config)
        self.agents = agents or []
        self.max_debate_rounds = self.config.get("max_debate_rounds", 3)
        self.physics_tiebreak_agent = "ValidatorAgent"

    # ------------------------------------------------------------------
    # deliberate() — main entry point
    # ------------------------------------------------------------------

    def deliberate(self, topic: str, proposal: str,
                   context: Dict[str, Any]) -> DeliberationResult:
        """
        Run the full deliberation protocol for a topic/proposal.

        Protocol:
        1. First-round votes
        2. If not unanimous → up to max_debate_rounds of debate
        3. Majority vote
        4. Physics tiebreak if tied
        5. Synthesize result

        Returns
        -------
        DeliberationResult
        """
        result = DeliberationResult(
            topic=topic,
            proposal=proposal,
            final_decision="APPROVE",
            winning_vote="APPROVE",
        )

        if not self.agents:
            result.synthesis = "MetaAgent: no agents registered; defaulting to APPROVE"
            result.confidence = 0.3
            return result

        # Round 0: initial votes
        all_votes = self._collect_votes(topic, proposal, context)
        result.votes.extend(all_votes)

        decision, confidence, tied = self._tally_votes(all_votes)

        # Debate rounds if split
        debate_round = 0
        while tied and debate_round < self.max_debate_rounds:
            debate_round += 1
            self.logger.info(
                f"MetaAgent: Debate round {debate_round} on '{topic}' "
                f"(tied at round {debate_round-1})"
            )

            # Build argument context from previous votes
            argument_context = self._build_argument_context(context, all_votes)

            # New votes with enriched context
            round_votes = self._collect_votes(
                topic, proposal, argument_context
            )
            all_votes.extend(round_votes)
            result.votes.extend(round_votes)

            decision, confidence, tied = self._tally_votes(all_votes)

        result.debate_rounds = debate_round

        # Physics tiebreak if still tied
        if tied:
            decision, confidence = self._physics_tiebreak(
                topic, proposal, context, all_votes
            )
            self.logger.info(
                f"MetaAgent: Physics tiebreak → {decision} (conf={confidence:.2f})"
            )

        result.final_decision = decision
        result.winning_vote = decision
        result.confidence = confidence

        # Collect MODIFY suggestions if decision is MODIFY
        if decision == "MODIFY":
            result.modifications = [
                v.modification_suggestion
                for v in all_votes
                if v.modification_suggestion and v.vote in ("MODIFY", "VETO")
            ]

        # Synthesize
        result.synthesis = self._synthesize(result, all_votes)

        self.logger.info(
            f"MetaAgent: Deliberation on '{topic}' → {decision} "
            f"(conf={confidence:.2f}, rounds={debate_round})"
        )
        return result

    # ------------------------------------------------------------------
    # analyze() / vote() — required by BaseAgent
    # ------------------------------------------------------------------

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """MetaAgent analysis: collect analyses from all sub-agents."""
        combined: Dict[str, Any] = {}
        for agent in self.agents:
            try:
                agent_result = agent.analyze(context)
                combined[agent.name] = agent_result
            except Exception as e:
                self.logger.debug(f"Agent {agent.name} analysis failed: {e}")
        return combined

    def vote(self, topic: str, proposal: str,
             context: Dict[str, Any]) -> AgentVote:
        """MetaAgent's own vote: trust the deliberation result."""
        result = self.deliberate(topic, proposal, context)
        return self._make_vote(
            result.final_decision,
            result.confidence,
            f"MetaAgent synthesis: {result.synthesis[:200]}",
            evidence={"debate_rounds": result.debate_rounds}
        )

    def _heuristic_decision(self, context: Any) -> Dict[str, Any]:
        """Deterministic fallback."""
        return {
            "decision": "APPROVE",
            "reason": "MetaAgent heuristic: default approval",
        }

    # ------------------------------------------------------------------
    # Internal debate mechanics
    # ------------------------------------------------------------------

    def _collect_votes(self, topic: str, proposal: str,
                        context: Dict[str, Any]) -> List[AgentVote]:
        """Collect votes from all agents."""
        votes = []
        for agent in self.agents:
            try:
                v = agent.vote(topic, proposal, context)
                votes.append(v)
                self.logger.debug(
                    f"  {agent.name} voted {v.vote} "
                    f"(conf={v.confidence:.2f}): {v.reason[:80]}"
                )
            except Exception as e:
                self.logger.warning(f"Agent {agent.name} vote failed: {e}")
                # Abstain on failure
                votes.append(AgentVote(
                    agent_name=agent.name,
                    vote="ABSTAIN",
                    confidence=0.0,
                    reason=f"Vote failed: {e}"
                ))
        return votes

    def _tally_votes(self, votes: List[AgentVote]) -> tuple:
        """
        Count votes and determine decision.

        Returns (decision, confidence, is_tied)
        """
        # Weight by confidence
        vote_weights: Dict[str, float] = {}
        for v in votes:
            if v.vote == "ABSTAIN":
                continue
            vote_weights[v.vote] = vote_weights.get(v.vote, 0.0) + v.confidence

        if not vote_weights:
            return "APPROVE", 0.3, False

        total = sum(vote_weights.values())
        # Sort by weight
        ranked = sorted(vote_weights.items(), key=lambda kv: kv[1], reverse=True)

        if len(ranked) < 2:
            decision, weight = ranked[0]
            return decision, weight / total, False

        top_vote, top_weight = ranked[0]
        second_vote, second_weight = ranked[1]

        # Tied if top two are within 10% of each other
        tied = abs(top_weight - second_weight) < 0.1 * total

        # VETO has veto power: a single strong veto overrides weak approvals
        if "VETO" in vote_weights:
            veto_weight = vote_weights["VETO"]
            approve_weight = vote_weights.get("APPROVE", 0.0)
            if veto_weight >= 0.4 * total:
                return "VETO", veto_weight / total, False

        confidence = top_weight / total if total > 0 else 0.3
        return top_vote, confidence, tied

    def _physics_tiebreak(self, topic: str, proposal: str,
                           context: Dict[str, Any],
                           all_votes: List[AgentVote]) -> tuple:
        """
        Break a tie by deferring to the physics tiebreak agent (ValidatorAgent).
        """
        tiebreak_agent = None
        for agent in self.agents:
            if agent.name == self.physics_tiebreak_agent:
                tiebreak_agent = agent
                break

        if tiebreak_agent is None:
            # No tiebreak agent available — default APPROVE
            return "APPROVE", 0.5

        try:
            v = tiebreak_agent.vote(topic, proposal, context)
            return v.vote, max(0.6, v.confidence)
        except Exception as e:
            self.logger.warning(f"Tiebreak agent failed: {e}")
            return "APPROVE", 0.5

    def _build_argument_context(self, context: Dict[str, Any],
                                  previous_votes: List[AgentVote]) -> Dict[str, Any]:
        """Enrich context with previous vote arguments for debate round."""
        new_context = dict(context)
        new_context["previous_vote_arguments"] = [
            {
                "agent": v.agent_name,
                "vote": v.vote,
                "reason": v.reason,
                "evidence": v.evidence,
            }
            for v in previous_votes
        ]
        return new_context

    def _synthesize(self, result: DeliberationResult,
                     all_votes: List[AgentVote]) -> str:
        """Generate a synthesis text of the deliberation outcome."""
        veto_votes = [v for v in all_votes if v.vote == "VETO"]
        approve_votes = [v for v in all_votes if v.vote == "APPROVE"]
        modify_votes = [v for v in all_votes if v.vote == "MODIFY"]

        lines = [
            f"Decision: {result.final_decision} "
            f"(confidence={result.confidence:.2f}, "
            f"debate_rounds={result.debate_rounds})"
        ]

        if veto_votes:
            lines.append(
                f"Veto arguments: " +
                "; ".join(v.reason[:100] for v in veto_votes[:2])
            )
        if approve_votes:
            lines.append(
                f"Approval arguments: " +
                "; ".join(v.reason[:80] for v in approve_votes[:2])
            )
        if modify_votes and result.modifications:
            lines.append(
                "Modifications required: " +
                "; ".join(str(m)[:100] for m in result.modifications[:3])
            )

        return " | ".join(lines)
