"""
Knowledge Memory
================
Gap 5 Fix: Knowledge is a formula list not structured memory.

Three stores:
1. exclusion_rules  — "log(bond_length) always fails"
2. success_patterns — "linear Z_X+polarity skeleton: R²>0.95 3/4 rounds"
3. failure_analyses — "ratio features generalize poorly OOD"

Persists to results/knowledge_memory.json between runs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from src.diagnostic_types import KnowledgeEntry, DiagnosticReport

logger = logging.getLogger("HARNESS.KnowledgeMemory")


class KnowledgeMemory:
    """
    Structured memory with three knowledge stores.

    Parameters
    ----------
    memory_path : Path or str, optional
        File path for JSON persistence. If None, memory is in-RAM only.
    config : dict, optional
        Config dict from harness_config.yaml knowledge_memory section.
    """

    def __init__(self, memory_path: Optional[Path] = None, config: dict = None):
        self.memory_path = Path(memory_path) if memory_path else None
        self.config = config or {}

        self.min_evidence = self.config.get("min_evidence_for_rule", 2)
        self.exclusion_confidence_threshold = self.config.get(
            "exclusion_confidence_threshold", 0.6
        )
        self.max_exclusion_rules = self.config.get("max_exclusion_rules", 20)
        self.max_success_patterns = self.config.get("max_success_patterns", 10)

        # Three stores: list of KnowledgeEntry
        self.exclusion_rules: List[KnowledgeEntry] = []
        self.success_patterns: List[KnowledgeEntry] = []
        self.failure_analyses: List[KnowledgeEntry] = []

        # Try to load from file
        if self.memory_path and self.memory_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Public recording API
    # ------------------------------------------------------------------

    def record_exclusion(self, pattern: str, reason: str,
                          round_id: int, formula: str) -> None:
        """
        Record or reinforce an exclusion rule.

        Example: record_exclusion("log(bond_length)", "always fails positivity", 3, formula)
        """
        existing = self._find_entry(self.exclusion_rules, pattern)
        if existing:
            existing.evidence_count += 1
            existing.confidence = min(
                1.0, existing.confidence + 0.15
            )
            if round_id not in existing.round_ids:
                existing.round_ids.append(round_id)
            if formula not in existing.formulas:
                existing.formulas.append(formula)
            logger.debug(
                f"[KnowledgeMemory] Reinforced exclusion '{pattern}' "
                f"(evidence={existing.evidence_count})"
            )
        else:
            entry = KnowledgeEntry(
                entry_type="exclusion_rule",
                pattern=pattern,
                description=reason,
                evidence_count=1,
                confidence=0.4,
                round_ids=[round_id],
                formulas=[formula],
            )
            self.exclusion_rules.append(entry)
            logger.debug(f"[KnowledgeMemory] New exclusion rule: '{pattern}'")

        # Trim to max size
        self._trim_store(self.exclusion_rules, self.max_exclusion_rules)

    def record_success(self, pattern: str, formula: str, r2: float,
                        round_id: int, features: List[str]) -> None:
        """
        Record a successful formula pattern.

        Example: record_success("linear Z_X+polarity", formula, 0.95, 4, features)
        """
        existing = self._find_entry(self.success_patterns, pattern)
        if existing:
            existing.evidence_count += 1
            existing.confidence = min(1.0, existing.confidence + 0.1)
            if round_id not in existing.round_ids:
                existing.round_ids.append(round_id)
            if formula not in existing.formulas:
                existing.formulas.append(formula)
            # Track best R²
            if r2 > existing.metadata.get("best_r2", 0.0):
                existing.metadata["best_r2"] = r2
        else:
            template = self._extract_structural_template(formula)
            entry = KnowledgeEntry(
                entry_type="success_pattern",
                pattern=pattern,
                description=f"Successful pattern with R²={r2:.4f}",
                evidence_count=1,
                confidence=0.5,
                round_ids=[round_id],
                formulas=[formula],
                metadata={
                    "best_r2": r2,
                    "features": features,
                    "structural_template": template,
                }
            )
            self.success_patterns.append(entry)
            logger.debug(f"[KnowledgeMemory] New success pattern: '{pattern}'")

        self._trim_store(self.success_patterns, self.max_success_patterns)

    def record_failure(self, features: List[str],
                        diagnostic: DiagnosticReport, round_id: int) -> None:
        """
        Record a failure analysis from a DiagnosticReport.

        Extracts patterns (log/exp/sqrt usage, feature subsets, structural templates).
        """
        formula = diagnostic.formula

        # 1. Check for common failing patterns
        for func_name in ("log(", "exp(", "sqrt("):
            if func_name in formula:
                pattern = func_name.rstrip("(")
                reason = (
                    f"{func_name} term failed with "
                    f"{diagnostic.violation_fraction*100:.0f}% violations"
                )
                self.record_exclusion(pattern, reason, round_id, formula)

        # 2. Record feature subset failure
        feature_key = "features:" + ",".join(sorted(features))
        existing = self._find_entry(self.failure_analyses, feature_key)
        if existing:
            existing.evidence_count += 1
            existing.round_ids.append(round_id)
            existing.formulas.append(formula)
        else:
            entry = KnowledgeEntry(
                entry_type="failure_analysis",
                pattern=feature_key,
                description=(
                    f"Feature set {sorted(features)} failed: "
                    f"{diagnostic.causal_text[:200]}"
                ),
                evidence_count=1,
                confidence=0.4,
                round_ids=[round_id],
                formulas=[formula],
                metadata={
                    "violation_fraction": diagnostic.violation_fraction,
                    "feature_sensitivities": diagnostic.feature_sensitivities,
                }
            )
            self.failure_analyses.append(entry)
            logger.debug(
                f"[KnowledgeMemory] Failure analysis recorded for round {round_id}"
            )

    # ------------------------------------------------------------------
    # LLM prompt formatting
    # ------------------------------------------------------------------

    def format_for_llm_prompt(self) -> str:
        """
        Format the three knowledge stores as a text block for injection into
        the LLM prompt.
        """
        lines = ["## Structured Knowledge Memory", ""]

        # Exclusion rules
        if self.exclusion_rules:
            lines.append("### Exclusion Rules (AVOID these patterns):")
            for entry in self.exclusion_rules:
                if entry.confidence >= self.exclusion_confidence_threshold:
                    lines.append(
                        f"  - AVOID '{entry.pattern}': {entry.description} "
                        f"(confidence={entry.confidence:.2f}, "
                        f"evidence={entry.evidence_count} rounds)"
                    )
        else:
            lines.append("### Exclusion Rules: (none yet)")

        lines.append("")

        # Success patterns
        if self.success_patterns:
            lines.append("### Success Patterns (PREFER these):")
            for entry in sorted(self.success_patterns,
                                 key=lambda e: e.metadata.get("best_r2", 0),
                                 reverse=True):
                r2 = entry.metadata.get("best_r2", 0)
                lines.append(
                    f"  - '{entry.pattern}': R²={r2:.4f}, "
                    f"evidence={entry.evidence_count} rounds. "
                    f"Example: {entry.formulas[-1][:80] if entry.formulas else 'N/A'}"
                )
        else:
            lines.append("### Success Patterns: (none yet)")

        lines.append("")

        # Failure analyses
        if self.failure_analyses:
            lines.append("### Failure Analyses:")
            for entry in self.failure_analyses[-3:]:  # last 3
                lines.append(
                    f"  - {entry.pattern}: {entry.description[:150]}"
                )
        else:
            lines.append("### Failure Analyses: (none yet)")

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        """Save knowledge memory to JSON."""
        target = path or self.memory_path
        if target is None:
            logger.warning("[KnowledgeMemory] No path set; cannot save.")
            return

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "exclusion_rules": [asdict(e) for e in self.exclusion_rules],
                "success_patterns": [asdict(e) for e in self.success_patterns],
                "failure_analyses": [asdict(e) for e in self.failure_analyses],
            }
            with open(target, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"[KnowledgeMemory] Saved to {target}")
        except Exception as e:
            logger.warning(f"[KnowledgeMemory] Save failed: {e}")

    def load(self, path: Optional[Path] = None) -> None:
        """Load knowledge memory from JSON."""
        target = path or self.memory_path
        if target is None or not target.exists():
            return

        try:
            with open(target, "r") as f:
                data = json.load(f)

            self.exclusion_rules = [
                KnowledgeEntry(**e) for e in data.get("exclusion_rules", [])
            ]
            self.success_patterns = [
                KnowledgeEntry(**e) for e in data.get("success_patterns", [])
            ]
            self.failure_analyses = [
                KnowledgeEntry(**e) for e in data.get("failure_analyses", [])
            ]
            logger.info(
                f"[KnowledgeMemory] Loaded {len(self.exclusion_rules)} exclusions, "
                f"{len(self.success_patterns)} successes, "
                f"{len(self.failure_analyses)} failures from {target}"
            )
        except Exception as e:
            logger.warning(f"[KnowledgeMemory] Load failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_entry(self, store: List[KnowledgeEntry],
                     pattern: str) -> Optional[KnowledgeEntry]:
        """Find an existing entry by pattern string."""
        for entry in store:
            if entry.pattern == pattern:
                return entry
        return None

    def _trim_store(self, store: List[KnowledgeEntry], max_size: int) -> None:
        """Trim store to max_size, keeping highest-confidence entries."""
        if len(store) > max_size:
            store.sort(key=lambda e: e.confidence, reverse=True)
            del store[max_size:]

    def _extract_structural_template(self, formula_str: str) -> str:
        """
        Extract structural complexity tier from formula.

        - 'linear': only + and * with no functions
        - 'power': contains ** or square
        - 'transcendental': contains exp, log, sqrt
        - 'rational': contains / with non-constant denominator
        """
        if re.search(r'\b(exp|log)\b', formula_str):
            return "transcendental"
        if re.search(r'\b(sqrt|square)\b', formula_str) or "**" in formula_str:
            return "power"
        if "/" in formula_str:
            return "rational"
        return "linear"
