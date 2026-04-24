"""
Base Agent
==========
Abstract base class for all Harness agents.

Provides:
- analyze(context) → Dict
- vote(topic, proposal, context) → AgentVote
- _call_llm() with fallback to _heuristic_decision()
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.diagnostic_types import AgentVote

logger = logging.getLogger("HARNESS.BaseAgent")


class BaseAgent(ABC):
    """
    Abstract base class for all Harness Engineering agents.

    Parameters
    ----------
    name : str
        Unique agent name (e.g. "DataAgent").
    role : str
        Description of the agent's role.
    knowledge : KnowledgeMemory, optional
        Shared knowledge memory instance.
    config : dict, optional
        Agent-specific configuration from harness_config.yaml agents section.
    """

    def __init__(self, name: str, role: str,
                 knowledge=None, config: dict = None):
        self.name = name
        self.role = role
        self.knowledge = knowledge
        self.config = config or {}
        self.logger = logging.getLogger(f"HARNESS.{name}")
        self._llm_available: Optional[bool] = None  # None = untested

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the given context and return structured findings.

        Parameters
        ----------
        context : dict
            Contains X_train, y_train, feature_names, round_id, etc.

        Returns
        -------
        dict
            Agent-specific analysis results.
        """

    @abstractmethod
    def vote(self, topic: str, proposal: str,
             context: Dict[str, Any]) -> AgentVote:
        """
        Cast a vote on a proposal.

        Parameters
        ----------
        topic : str
            What is being voted on (e.g. "feature_selection", "search_space").
        proposal : str
            The proposal being voted on.
        context : dict
            Same context as analyze().

        Returns
        -------
        AgentVote
        """

    # ------------------------------------------------------------------
    # LLM call with heuristic fallback
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Try to call the LLM. Falls back to _heuristic_decision() on any failure.
        """
        if self._llm_available is False:
            return self._heuristic_decision(prompt)

        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.config.get("llm_model", "claude-sonnet-4-20250514"),
                max_tokens=self.config.get("max_tokens", 512),
                temperature=self.config.get("temperature", 0.3),
                messages=[{"role": "user", "content": prompt}]
            )
            self._llm_available = True
            text = response.content[0].text

            import json, re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            # If no JSON, return raw text under 'text' key
            return {"text": text}

        except ImportError:
            self.logger.info(
                "anthropic package not installed; using heuristic mode"
            )
            self._llm_available = False
            return self._heuristic_decision(prompt)

        except Exception as e:
            self.logger.warning(f"LLM call failed ({e}); using heuristic mode")
            self._llm_available = False
            return self._heuristic_decision(prompt)

    @abstractmethod
    def _heuristic_decision(self, context: Any) -> Dict[str, Any]:
        """
        Pure deterministic heuristic fallback when LLM is unavailable.
        Must be implemented by each concrete agent.
        """

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _make_vote(self, vote: str, confidence: float,
                   reason: str, evidence: Dict = None,
                   modification: str = None) -> AgentVote:
        """Convenience factory for AgentVote."""
        return AgentVote(
            agent_name=self.name,
            vote=vote,
            confidence=confidence,
            reason=reason,
            evidence=evidence or {},
            modification_suggestion=modification,
        )
