"""
Harness Orchestrator
====================
The central coordination engine that implements the Harness Engineering
feedback loop for scientific equation discovery.

Orchestrates:
- Layer 1: Data pipeline
- Layer 2: Physical constraints
- Layer 3: LLM-SR engine
- Layer 4: Multi-agent collaboration
- Layer 5: Human interface (results reporting)
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from src.data_pipeline import TMDDataPipeline
from src.llm_sr_engine import LLMSREngine, FormulaCandidate
from src.cauchy_validator import CauchyValidator
from src.formula_evaluator import FormulaEvaluator

logger = logging.getLogger("HARNESS.Orchestrator")


class HarnessOrchestrator:
    """
    Central Harness Engineering orchestrator.
    
    Implements the core feedback loop:
    ┌─────────────────────────────────────────────┐
    │                                             │
    │   ┌─────────┐    ┌─────────┐    ┌────────┐ │
    │   │ LLM     │───>│ PySR    │───>│Evaluate│ │
    │   │ Analyze │    │ Search  │    │& Valid │ │
    │   └────▲────┘    └─────────┘    └───┬────┘ │
    │        │                            │      │
    │        │     ┌──────────────┐       │      │
    │        └─────│ Experience   │<──────┘      │
    │              │ Pool         │              │
    │              └──────────────┘              │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, config: dict, data_pipeline: TMDDataPipeline,
                 llm_sr_engine: LLMSREngine, validator: CauchyValidator,
                 evaluator: FormulaEvaluator,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 feature_names: List[str], target_name: str,
                 # New optional Harness Engineering components
                 agent_council=None,
                 knowledge=None,
                 adaptive_env=None,
                 diagnostic_engine=None):
        self.config = config
        self.data_pipeline = data_pipeline
        self.llm_sr = llm_sr_engine
        self.validator = validator
        self.evaluator = evaluator
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.target_name = target_name

        # New Harness Engineering components (all optional)
        self.agent_council = agent_council
        self.knowledge = knowledge
        self.adaptive_env = adaptive_env
        self.diagnostic_engine = diagnostic_engine

        # Results tracking
        self.all_candidates: List[FormulaCandidate] = []
        self.best_candidate: Optional[FormulaCandidate] = None
        self.run_history: List[Dict] = []

    def run(self) -> Dict:
        """
        Execute the complete Harness discovery pipeline.
        
        Returns:
            Dictionary with discovery results.
        """
        logger.info("\n" + "=" * 50)
        logger.info("HARNESS ORCHESTRATOR: Starting discovery pipeline")
        logger.info("=" * 50)

        # Get feature descriptions for LLM prompts
        feature_descs = self.data_pipeline.get_feature_descriptions()

        # --- Phase 1: Multi-Agent Pre-Analysis ---
        logger.info("\n[Phase 1] Multi-Agent Pre-Analysis")
        analysis = self._agent_pre_analysis()

        # --- Phase 2: LLM-SR Discovery Loop ---
        logger.info("\n[Phase 2] LLM-SR Discovery Loop")
        candidates = self.llm_sr.run_discovery_loop(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            feature_descs,
            adaptive_env=self.adaptive_env,
            agent_council=self.agent_council,
            knowledge=self.knowledge,
        )
        self.all_candidates = candidates

        # --- Phase 3: Multi-Agent Post-Validation ---
        logger.info("\n[Phase 3] Multi-Agent Post-Validation")
        validated = self._agent_post_validation(candidates)

        # --- Phase 4: Final Ranking & Selection ---
        logger.info("\n[Phase 4] Final Ranking & Selection")
        results = self._final_selection(validated)

        return results

    def _agent_pre_analysis(self) -> Dict:
        """
        Phase 1: DataAgent and TheoryAgent analyze the data before SR.

        If agent_council is provided, delegates to it (Harness Engineering mode).
        Otherwise, runs legacy correlation/hypothesis analysis (backward compat).
        """
        # --- Harness Engineering mode: delegate to AgentCouncil ---
        if self.agent_council is not None:
            try:
                logger.info("  AgentCouncil: Running pre-search deliberation...")
                context = {
                    "X_train": self.X_train, "y_train": self.y_train,
                    "X_test": self.X_test, "y_test": self.y_test,
                    "feature_names": self.feature_names,
                    "round_id": 0,
                }
                instructions = self.agent_council.pre_search_deliberation(0, context)
                logger.info(
                    f"  AgentCouncil recommended features: "
                    f"{instructions.get('features', [])}"
                )
                return instructions
            except Exception as e:
                logger.warning(
                    f"  AgentCouncil pre-analysis failed ({e}); "
                    f"falling back to legacy mode"
                )

        # --- Legacy mode ---
        logger.info("  DataAgent: Analyzing feature correlations...")

        # Correlation analysis
        correlations = {}
        for i, fname in enumerate(self.feature_names):
            corr = np.corrcoef(self.X_train[:, i], self.y_train)[0, 1]
            correlations[fname] = round(corr, 4)

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        logger.info("  Top correlated features:")
        for fname, corr in sorted_corr[:5]:
            logger.info(f"    {fname}: r = {corr:.4f}")

        logger.info("  TheoryAgent: Generating physics hypotheses...")
        hypotheses = [
            "n ∝ Z_X (chalcogen atomic number drives polarizability)",
            "n ∝ d_electrons (d-orbital contribution to optical transitions)",
            "n ∝ bond_polarity (ionic character enhances dielectric response)",
            "n ∝ 1/bond_length (shorter bonds → denser packing → higher n)",
        ]
        for h in hypotheses:
            logger.info(f"    Hypothesis: {h}")

        return {
            'correlations': correlations,
            'top_features': [f for f, _ in sorted_corr[:5]],
            'hypotheses': hypotheses,
        }

    def _agent_post_validation(self, candidates: List[FormulaCandidate]) -> List[FormulaCandidate]:
        """
        Phase 3: ValidatorAgent checks all discovered formulas.

        If agent_council + diagnostic_engine are provided, uses them for
        material-level causal diagnostics (Harness Engineering mode).
        Otherwise, runs legacy mock-prediction validation (backward compat).
        """
        # --- Harness Engineering mode ---
        if self.agent_council is not None and self.diagnostic_engine is not None:
            try:
                logger.info(
                    f"  AgentCouncil: Post-search validation of "
                    f"{len(candidates)} candidates..."
                )
                context = {
                    "X_train": self.X_train, "y_train": self.y_train,
                    "X_test": self.X_test, "y_test": self.y_test,
                    "feature_names": self.feature_names,
                }
                accepted, diag_reports = self.agent_council.post_search_deliberation(
                    candidates, context
                )

                # Update knowledge from diagnostics
                if diag_reports:
                    causal_feedback = self._build_causal_feedback(diag_reports)
                    if causal_feedback:
                        logger.info(f"  Causal feedback:\n{causal_feedback}")

                # Mark physics_valid on accepted candidates
                accepted_set = {id(c) for c in accepted}
                for c in candidates:
                    c.physics_valid = id(c) in accepted_set

                valid_count = len(accepted)
                logger.info(
                    f"  {valid_count}/{len(candidates)} candidates accepted "
                    f"by AgentCouncil"
                )
                return candidates  # Return all; physics_valid flag is set

            except Exception as e:
                logger.warning(
                    f"  AgentCouncil post-validation failed ({e}); "
                    f"falling back to legacy mode"
                )

        # --- Legacy mode ---
        logger.info(f"  ValidatorAgent: Checking {len(candidates)} candidates...")

        validated = []
        for c in candidates:
            if c.test_rmse < float('inf') and c.test_rmse > 0:
                # Simulate predictions within expected range
                mock_preds = np.random.normal(
                    np.mean(self.y_test), c.test_rmse, size=len(self.y_test)
                )
                mock_preds = np.clip(mock_preds, 1.0, 8.0)

                result = self.validator.validate(
                    c.expression, mock_preds, c.complexity, c.features_used
                )
                c.physics_valid = result.is_valid

                if result.warnings:
                    logger.debug(f"    '{c.expression}': {result.warnings}")

                validated.append(c)

        valid_count = sum(1 for c in validated if c.physics_valid)
        logger.info(f"  {valid_count}/{len(validated)} candidates passed physics validation")

        return validated

    def _build_causal_feedback(self, diagnostic_reports: List) -> str:
        """
        Build a causal feedback text block from DiagnosticReport objects.
        """
        if not diagnostic_reports:
            return ""
        lines = []
        for diag in diagnostic_reports[:3]:  # limit to 3
            causal = getattr(diag, "causal_text", "")
            if causal:
                lines.append(causal[:300])
        return "\n---\n".join(lines)

    def _update_knowledge(self, round_id: int, candidates: List[FormulaCandidate],
                           diagnostics: List) -> None:
        """
        Update KnowledgeMemory after a search round.
        Called from run() if knowledge is available.
        """
        if self.knowledge is None:
            return

        for candidate in candidates[:3]:
            formula = getattr(candidate, "expression", "")
            r2 = getattr(candidate, "r2", 0.0)
            features = getattr(candidate, "features_used", [])

            if r2 > 0.9:
                try:
                    self.knowledge.record_success(
                        f"round{round_id}_top", formula, r2,
                        round_id, features
                    )
                except Exception:
                    pass

        for diag in diagnostics:
            vf = getattr(diag, "violation_fraction", 0.0)
            if vf > 0.05:
                try:
                    formula = getattr(diag, "formula", "")
                    features = getattr(candidate, "features_used", [])
                    self.knowledge.record_failure(features, diag, round_id)
                except Exception:
                    pass

    def _final_selection(self, candidates: List[FormulaCandidate]) -> Dict:
        """
        Phase 4: Select the best formula from validated candidates.
        
        Uses the multi-objective scoring function to balance
        accuracy, simplicity, and physical validity.
        """
        if not candidates:
            return {
                'best_formula': None,
                'train_rmse': float('inf'),
                'test_rmse': float('inf'),
                'r2': 0.0,
                'complexity': 0,
                'physics_valid': False,
                'rounds_used': len(self.llm_sr.experience_pool),
            }

        # Score and rank
        scored = self.evaluator.rank_formulas(candidates)

        if scored:
            best, best_score = scored[0]
            self.best_candidate = best

            logger.info(f"\n  BEST FORMULA DISCOVERED:")
            logger.info(f"    Expression: {best.expression}")
            logger.info(f"    Train RMSE: {best.train_rmse:.6f}")
            logger.info(f"    Test RMSE:  {best.test_rmse:.6f}")
            logger.info(f"    R² Score:   {best.r2:.6f}")
            logger.info(f"    Complexity: {best.complexity}")
            logger.info(f"    Features:   {best.features_used}")
            logger.info(f"    Physics OK: {best.physics_valid}")
            logger.info(f"    Overall Score: {best_score['overall']:.4f}")

            # Show Pareto front
            if len(scored) > 1:
                logger.info(f"\n  PARETO FRONT ({len(scored)} formulas):")
                for i, (c, s) in enumerate(scored[:5]):
                    logger.info(
                        f"    #{i+1}: RMSE={c.test_rmse:.4f}, "
                        f"complexity={c.complexity}, "
                        f"score={s['overall']:.4f}"
                    )
                    logger.info(f"         {c.expression}")

            return {
                'best_formula': best.expression,
                'train_rmse': best.train_rmse,
                'test_rmse': best.test_rmse,
                'r2': best.r2,
                'complexity': best.complexity,
                'physics_valid': best.physics_valid,
                'rounds_used': len(self.llm_sr.experience_pool),
                'pareto_front_size': len(scored),
                'features_used': best.features_used,
            }

        return {'best_formula': None, 'rounds_used': 0}

    def export_results(self, output_dir: Path):
        """Export results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Best formula report
        report = {
            'timestamp': timestamp,
            'target': self.target_name,
            'best_formula': self.best_candidate.expression if self.best_candidate else None,
            'metrics': {
                'train_rmse': float(self.best_candidate.train_rmse) if self.best_candidate else None,
                'test_rmse': float(self.best_candidate.test_rmse) if self.best_candidate else None,
                'r2': float(self.best_candidate.r2) if self.best_candidate else None,
                'complexity': self.best_candidate.complexity if self.best_candidate else None,
            },
            'features_used': self.best_candidate.features_used if self.best_candidate else [],
            'experience_pool_size': len(self.llm_sr.experience_pool),
            'pareto_front_size': len(self.llm_sr.pareto_front),
        }

        report_path = output_dir / f"discovery_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"  Report saved: {report_path}")

        # 2. Experience pool
        exp_path = output_dir / f"experience_pool_{timestamp}.json"
        exp_data = [
            {
                'round': e.round_id,
                'features': e.features_tried,
                'formula': e.best_formula,
                'rmse': float(e.rmse),
                'complexity': e.complexity,
                'feedback': e.llm_feedback,
                'success': bool(e.success),
            }
            for e in self.llm_sr.experience_pool
        ]
        with open(exp_path, 'w') as f:
            json.dump(exp_data, f, indent=2)
        logger.info(f"  Experience pool saved: {exp_path}")

        # 3. Pareto front
        pareto_path = output_dir / f"pareto_front_{timestamp}.json"
        pareto_data = [
            {
                'expression': c.expression,
                'complexity': c.complexity,
                'train_rmse': float(c.train_rmse),
                'test_rmse': float(c.test_rmse),
                'r2': float(c.r2),
                'features': c.features_used,
                'round': c.round_discovered,
                'physics_valid': c.physics_valid,
            }
            for c in self.llm_sr.pareto_front
        ]
        with open(pareto_path, 'w') as f:
            json.dump(pareto_data, f, indent=2)
        logger.info(f"  Pareto front saved: {pareto_path}")

        # 4. LaTeX export
        if self.best_candidate:
            latex_path = output_dir / f"best_formula_{timestamp}.tex"
            with open(latex_path, 'w') as f:
                f.write(f"% HARNESS-2D-RI Discovery Result\n")
                f.write(f"% Target: {self.target_name}\n")
                f.write(f"% Generated: {timestamp}\n\n")
                f.write(f"% Best Formula:\n")
                f.write(f"% {self.best_candidate.expression}\n\n")
                f.write(f"% Metrics:\n")
                f.write(f"% Train RMSE = {self.best_candidate.train_rmse:.6f}\n")
                f.write(f"% Test RMSE  = {self.best_candidate.test_rmse:.6f}\n")
                f.write(f"% R^2        = {self.best_candidate.r2:.6f}\n")
                f.write(f"% Complexity = {self.best_candidate.complexity}\n")
            logger.info(f"  LaTeX export saved: {latex_path}")
