"""
Diagnostic Types
================
Shared dataclasses for the Harness Engineering diagnostic and deliberation system.

All components (CausalDiagnosticEngine, agents, KnowledgeMemory) import from here
to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class MaterialViolation:
    """A single material that violated a physical constraint."""
    material_id: str          # e.g. "MoS2_H"
    metal: str                # e.g. "Mo"
    chalcogen: str            # e.g. "S"
    phase: str                # e.g. "H"
    predicted_value: float    # Model prediction
    actual_value: float       # Ground truth (if available)
    violation_type: str       # "positivity", "range", "sign", etc.
    additive_terms: Dict[str, float] = field(default_factory=dict)
    # Maps term string → its contribution to the violation


@dataclass
class ConstraintDiagnostic:
    """Diagnosis for a single constraint check."""
    check_name: str           # e.g. "positivity"
    passed: bool
    violating_row_indices: List[int] = field(default_factory=list)
    violating_material_ids: List[str] = field(default_factory=list)
    worst_violation: float = 0.0
    explanation: str = ""


@dataclass
class DiagnosticReport:
    """Full diagnostic report for one formula candidate."""
    formula: str
    constraint_diagnostics: List[ConstraintDiagnostic] = field(default_factory=list)
    material_violations: List[MaterialViolation] = field(default_factory=list)
    feature_sensitivities: Dict[str, float] = field(default_factory=dict)
    # Maps feature_name → sensitivity (finite-difference estimate)
    additive_term_contributions: Dict[str, float] = field(default_factory=dict)
    # Maps term string → mean contribution across all materials
    causal_text: str = ""
    # Human-readable causal explanation for the LLM
    overall_valid: bool = True
    violation_fraction: float = 0.0


@dataclass
class AgentVote:
    """A vote cast by an agent on a proposal."""
    agent_name: str
    vote: str                 # "APPROVE", "VETO", "MODIFY", "ABSTAIN"
    confidence: float         # 0.0 - 1.0
    reason: str               # Human-readable justification
    evidence: Dict[str, Any] = field(default_factory=dict)
    # Structured evidence backing the vote
    modification_suggestion: Optional[str] = None
    # If vote == "MODIFY", what to change


@dataclass
class DeliberationResult:
    """Result of a multi-agent deliberation on a topic."""
    topic: str
    proposal: str
    final_decision: str       # "APPROVE", "VETO", "MODIFY"
    winning_vote: str
    debate_rounds: int = 0
    votes: List[AgentVote] = field(default_factory=list)
    synthesis: str = ""       # MetaAgent's synthesis of the debate
    modifications: List[str] = field(default_factory=list)
    # If MODIFY, what modifications were agreed upon
    confidence: float = 0.0


@dataclass
class KnowledgeEntry:
    """A single entry in the knowledge memory."""
    entry_type: str           # "exclusion_rule", "success_pattern", "failure_analysis"
    pattern: str              # The pattern (e.g. "log(bond_length)")
    description: str          # Human-readable description
    evidence_count: int = 1   # How many rounds support this
    confidence: float = 0.5
    round_ids: List[int] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    # Supporting formula examples
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Extra data (r2, features, etc.)
