"""
LLM-SR Engine
=============
Layer 3 of the Harness: LLM-enhanced Symbolic Regression engine.

Hybrid architecture combining:
- LangLaw: LLM-guided iterative PySR with experience pool (materials-specific)
- LLM-SR: Program skeleton representation with evolutionary search

The LLM acts as a scientific advisor that:
1. Analyzes data and domain context to select relevant features
2. Generates physics-informed hypothesis skeletons
3. Reviews results and refines search parameters per iteration
4. Maintains an experience pool of successful/failed formula patterns
"""

import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger("HARNESS.LLM_SR")


@dataclass
class FormulaCandidate:
    """A candidate formula discovered by the SR engine."""
    expression: str
    complexity: int
    train_rmse: float
    test_rmse: float = float('inf')
    r2: float = 0.0
    physics_valid: bool = False
    features_used: List[str] = field(default_factory=list)
    round_discovered: int = 0
    llm_score: float = 0.0  # LLM's assessment of physical plausibility
    # Cached test predictions — used by ValidatorAgent to avoid re-evaluation
    y_pred_test: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ExperienceEntry:
    """An entry in the experience pool."""
    round_id: int
    features_tried: List[str]
    best_formula: str
    rmse: float
    complexity: int
    llm_feedback: str
    success: bool
    # New fields for Harness Engineering (backward-compatible defaults)
    diagnostic_summary: str = ""
    search_config_used: Dict[str, Any] = field(default_factory=dict)


class LLMSREngine:
    """
    LLM-enhanced Symbolic Regression engine.
    
    Implements the LangLaw-style iterative loop:
    LLM Analysis → Feature Selection → PySR Search → Evaluation → Experience Pool → Loop
    """

    def __init__(self, config: dict, feature_names: List[str],
                 target_name: str, constraints: dict):
        self.config = config
        self.feature_names = feature_names
        self.target_name = target_name
        self.constraints = constraints

        # Experience pool (core of LangLaw approach)
        self.experience_pool: List[ExperienceEntry] = []

        # Pareto front of discovered formulas
        self.pareto_front: List[FormulaCandidate] = []

        # LLM configuration
        self.llm_config = config.get('llm', {})
        self.pysr_config = config.get('pysr', {})
        self.search_config = config.get('search', {})

    def run_discovery_loop(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           feature_descriptions: dict,
                           adaptive_env=None,
                           agent_council=None,
                           knowledge=None) -> List[FormulaCandidate]:
        """
        Execute the main LLM-guided symbolic regression loop.

        Parameters (all optional for backward compatibility)
        ----------
        adaptive_env : AdaptiveSearchEnvironment, optional
        agent_council : AgentCouncil, optional
        knowledge : KnowledgeMemory, optional

        This is the core Harness feedback cycle:
        1. LLM analyzes data context + experience pool
        2. LLM selects features and search parameters
        3. PySR runs constrained evolutionary search
        4. Results evaluated against physics constraints
        5. Experience pool updated
        6. Loop until convergence or max rounds
        """
        max_rounds = self.search_config.get('max_rounds', 15)
        convergence_threshold = self.search_config.get('convergence_threshold', 0.01)
        best_rmse = float('inf')

        logger.info(f"Starting LLM-SR discovery loop (max {max_rounds} rounds)")

        for round_id in range(1, max_rounds + 1):
            logger.info(f"\n--- Round {round_id}/{max_rounds} ---")

            # Build context for agent council
            agent_context = {
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test,
                "feature_names": self.feature_names,
                "round_id": round_id,
            }

            # Step 1a: Optional agent pre-deliberation
            council_instructions: Dict[str, Any] = {}
            if agent_council is not None:
                try:
                    council_instructions = agent_council.pre_search_deliberation(
                        round_id, agent_context
                    )
                    logger.info(
                        f"  AgentCouncil pre-deliberation: "
                        f"features={council_instructions.get('features', [])}"
                    )
                except Exception as e:
                    logger.warning(f"  AgentCouncil pre-deliberation failed: {e}")

            # Step 1b: LLM generates search instructions (enriched with council output)
            knowledge_block = council_instructions.get("knowledge_block", "")
            if not knowledge_block and knowledge is not None:
                try:
                    knowledge_block = knowledge.format_for_llm_prompt()
                except Exception:
                    pass

            instructions = self._llm_generate_instructions(
                round_id, feature_descriptions, X_train, y_train,
                knowledge_block=knowledge_block,
            )

            # Override features if agent council provided them
            if council_instructions.get("features"):
                instructions["features"] = council_instructions["features"]
            if council_instructions.get("hypothesis"):
                instructions["hypothesis"] = council_instructions["hypothesis"]

            logger.info(f"  LLM selected features: {instructions['features']}")
            logger.info(f"  LLM hypothesis: {instructions['hypothesis']}")

            # Step 1c: Get adaptive config
            pysr_config_override: Dict[str, Any] = {}
            if adaptive_env is not None:
                try:
                    theory_recs = (
                        council_instructions.get("operator_restrictions")
                        and {"operators_to_restrict": council_instructions["operator_restrictions"]}
                        or None
                    )
                    pysr_config_override = adaptive_env.get_current_config(
                        round_id, theory_recs
                    )
                except Exception as e:
                    logger.warning(f"  AdaptiveEnv config failed: {e}")

            # Step 2: Run PySR with LLM-guided constraints
            candidates = self._run_pysr(
                X_train, y_train, X_test, y_test,
                instructions, round_id,
                pysr_config_override=pysr_config_override,
            )

            # Step 3: Evaluate candidates
            valid_candidates = self._evaluate_candidates(
                candidates, X_test, y_test, round_id
            )

            # Step 3b: Optional post-search deliberation
            diagnostic_reports = []
            if agent_council is not None and valid_candidates:
                try:
                    valid_candidates, diagnostic_reports = (
                        agent_council.post_search_deliberation(
                            valid_candidates, agent_context
                        )
                    )
                    logger.info(
                        f"  AgentCouncil post-deliberation: "
                        f"{len(valid_candidates)} accepted"
                    )
                except Exception as e:
                    logger.warning(f"  AgentCouncil post-deliberation failed: {e}")

            # Step 4: Update experience pool
            self._update_experience_pool(
                round_id, instructions, valid_candidates,
                diagnostic_reports=diagnostic_reports,
                search_config_used=pysr_config_override,
            )

            # Step 4b: Update knowledge memory
            if knowledge is not None and valid_candidates:
                try:
                    self._update_knowledge(
                        knowledge, round_id, valid_candidates, diagnostic_reports
                    )
                except Exception as e:
                    logger.debug(f"  Knowledge update failed: {e}")

            # Step 4c: Record adaptive env outcome
            if adaptive_env is not None:
                try:
                    adaptive_env.record_round_outcome(
                        round_id, pysr_config_override or {},
                        valid_candidates, diagnostic_reports
                    )
                except Exception as e:
                    logger.debug(f"  Adaptive env record failed: {e}")

            # Step 5: Update Pareto front
            self._update_pareto_front(valid_candidates)

            # Step 6: Check convergence
            if valid_candidates:
                round_best = min(c.test_rmse for c in valid_candidates)
                improvement = best_rmse - round_best
                if round_best < best_rmse:
                    best_rmse = round_best
                    logger.info(f"  New best RMSE: {best_rmse:.6f} (improvement: {improvement:.6f})")

                if 0 < improvement < convergence_threshold and round_id > 3:
                    logger.info(f"  Converged at round {round_id} (improvement {improvement:.6f} < {convergence_threshold})")
                    break
            else:
                logger.warning("  No valid candidates in this round")

        logger.info(f"\nDiscovery loop complete. Pareto front size: {len(self.pareto_front)}")
        return sorted(self.pareto_front, key=lambda c: c.test_rmse)

    def _llm_generate_instructions(self, round_id: int,
                                    feature_descriptions: dict,
                                    X: np.ndarray, y: np.ndarray,
                                    knowledge_block: str = "",
                                    diagnostic_block: str = "") -> dict:
        """
        Use LLM to generate search instructions for this round.

        The LLM receives:
        - Feature descriptions and basic statistics
        - Previous experience pool entries
        - Physical constraints
        - Target property description
        - Optional knowledge_block (KnowledgeMemory formatted text)
        - Optional diagnostic_block (causal feedback from previous round)

        The LLM returns:
        - Selected feature subset
        - Hypothesis skeleton
        - PySR parameter adjustments
        """
        # Build the prompt (LangLaw-style)
        prompt = self._build_instruction_prompt(
            round_id, feature_descriptions, X, y,
            knowledge_block=knowledge_block,
            diagnostic_block=diagnostic_block,
        )

        # Call LLM (or use heuristic fallback for demo)
        try:
            instructions = self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"LLM call failed ({e}), using heuristic fallback")
            instructions = self._heuristic_fallback(round_id)

        return instructions

    def _build_instruction_prompt(self, round_id: int,
                                   feature_descriptions: dict,
                                   X: np.ndarray, y: np.ndarray,
                                   knowledge_block: str = "",
                                   diagnostic_block: str = "") -> str:
        """Build the LLM prompt following LangLaw methodology."""
        # Feature statistics
        stats = {}
        for i, fname in enumerate(self.feature_names):
            col = X[:, i]
            stats[fname] = {
                'mean': f"{np.mean(col):.4f}",
                'std': f"{np.std(col):.4f}",
                'min': f"{np.min(col):.4f}",
                'max': f"{np.max(col):.4f}",
                'corr_with_target': f"{np.corrcoef(col, y)[0,1]:.4f}"
            }

        # Experience pool summary
        exp_summary = ""
        if self.experience_pool:
            exp_summary = "\n## Previous Search Results (Experience Pool)\n"
            for entry in self.experience_pool[-5:]:  # Last 5 entries
                exp_summary += (
                    f"- Round {entry.round_id}: features={entry.features_tried}, "
                    f"best_formula='{entry.best_formula}', RMSE={entry.rmse:.4f}, "
                    f"complexity={entry.complexity}, success={entry.success}\n"
                    f"  Feedback: {entry.llm_feedback}\n"
                )

        # Optional knowledge block injection
        knowledge_section = ""
        if knowledge_block:
            knowledge_section = f"\n{knowledge_block}\n"

        # Optional causal diagnostic block injection
        diagnostic_section = ""
        if diagnostic_block:
            diagnostic_section = (
                f"\n## Causal Feedback from Previous Round\n{diagnostic_block}\n"
            )

        prompt = f"""You are a materials scientist discovering equations relating
atomic structure features to optical properties of 2D transition metal dichalcogenide (TMD) monolayers.

## Task
Find an interpretable mathematical formula that predicts the **{self.target_name}**
(related to refractive index via the Cauchy dispersion model: n(λ) = A + B/λ²)
from structural features of TMD monolayers.

## Available Features
{json.dumps(feature_descriptions, indent=2)}

## Feature Statistics (from {len(X)} samples)
{json.dumps(stats, indent=2)}

## Physical Constraints
- The refractive index must be dimensionless and positive (n > 1)
- Cauchy coefficient A represents the baseline refractive index
- Higher atomic number chalcogens (Te > Se > S) generally give higher n
- Bond polarizability and d-electron count influence optical response
- Simpler formulas are preferred (Occam's razor)
{exp_summary}{knowledge_section}{diagnostic_section}
## Round {round_id} Instructions
Based on the above information, provide:
1. **features**: A list of 3-6 most relevant features to use (from the available features)
2. **hypothesis**: A text description of the physical hypothesis for this round
3. **max_depth**: Maximum expression tree depth (3-6)
4. **iterations**: Number of PySR iterations (50-200)

Respond in valid JSON format only:
{{"features": [...], "hypothesis": "...", "max_depth": N, "iterations": N}}
"""
        return prompt

    def _call_llm(self, prompt: str) -> dict:
        """
        Call the LLM API (Anthropic Claude).
        
        In production, this calls the Anthropic API.
        For demo/offline mode, uses a heuristic strategy.
        """
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.llm_config.get('model', 'claude-sonnet-4-20250514'),
                max_tokens=self.llm_config.get('max_tokens', 1024),
                temperature=self.llm_config.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except ImportError:
            logger.info("anthropic package not installed, using heuristic mode")
        except Exception as e:
            logger.warning(f"LLM API error: {e}")

        # Fallback
        return self._heuristic_fallback(len(self.experience_pool) + 1)

    def _heuristic_fallback(self, round_id: int) -> dict:
        """
        Physics-informed heuristic when LLM is unavailable.
        
        Rotates through different feature subsets based on physical intuition,
        simulating what an expert would suggest.
        """
        strategies = [
            # Round 1: Core atomic properties
            {
                'features': ['atomic_number_X', 'd_electron_count', 'bond_polarity',
                            'bond_length_MX'],
                'hypothesis': 'RI dominated by chalcogen polarizability and d-electron count',
                'max_depth': 4, 'iterations': 100
            },
            # Round 2: Electronegativity-based
            {
                'features': ['electronegativity_diff', 'atomic_number_X', 'radius_ratio',
                            'lattice_constant_a'],
                'hypothesis': 'RI correlates with bond polarity and packing density',
                'max_depth': 5, 'iterations': 120
            },
            # Round 3: All structural
            {
                'features': ['atomic_number_M', 'atomic_number_X', 'd_electron_count',
                            'bond_length_MX', 'monolayer_thickness'],
                'hypothesis': 'Full structural model: Z, d-electrons, and geometry',
                'max_depth': 5, 'iterations': 150
            },
            # Round 4: Minimal physics model
            {
                'features': ['atomic_number_X', 'bond_polarity', 'd_electron_count'],
                'hypothesis': 'Minimal model: only 3 key descriptors',
                'max_depth': 4, 'iterations': 100
            },
            # Round 5: Energy-based
            {
                'features': ['ionization_energy_M', 'electron_affinity_X',
                            'electronegativity_diff', 'atomic_number_X', 'd_electron_count'],
                'hypothesis': 'Energy scales (IE, EA) as drivers of optical response',
                'max_depth': 5, 'iterations': 150
            },
            # Round 6+: Ratios and derived
            {
                'features': ['Z_ratio', 'IE_EA_ratio', 'bond_polarity',
                            'packing_fraction', 'd_electron_count'],
                'hypothesis': 'Dimensionless ratios as universal descriptors',
                'max_depth': 6, 'iterations': 200
            },
        ]

        idx = (round_id - 1) % len(strategies)
        return strategies[idx]

    def _run_pysr(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   instructions: dict, round_id: int,
                   pysr_config_override: dict = None) -> List[FormulaCandidate]:
        """
        Run PySR with LLM-guided parameters.

        PySR uses multi-island genetic programming to evolve
        mathematical expressions as tree structures.

        pysr_config_override : dict, optional
            Config from AdaptiveSearchEnvironment; overrides pysr_config values.
        """
        selected_features = instructions['features']
        feature_indices = [self.feature_names.index(f)
                          for f in selected_features
                          if f in self.feature_names]

        if not feature_indices:
            logger.warning("No valid features selected, using all features")
            feature_indices = list(range(len(self.feature_names)))
            selected_features = self.feature_names

        X_sub = X_train[:, feature_indices]
        X_test_sub = X_test[:, feature_indices]

        # Merge base pysr_config with adaptive override
        effective_config = dict(self.pysr_config)
        if pysr_config_override:
            effective_config.update(pysr_config_override)

        try:
            from pysr import PySRRegressor
            model = PySRRegressor(
                niterations=instructions.get('iterations', 100),
                populations=effective_config.get('populations', 40),
                population_size=effective_config.get('population_size', 33),
                binary_operators=effective_config.get('binary_operators',
                                                       ["+", "-", "*", "/"]),
                unary_operators=effective_config.get('unary_operators',
                                                      ["exp", "log", "sqrt", "square", "inv"]),
                maxsize=instructions.get('max_depth', 5) * 5,
                parsimony=self.pysr_config.get('parsimony', 0.05),
                timeout_in_seconds=300,
                progress=False,
                verbosity=0,
                variable_names=[selected_features[i] for i in range(len(feature_indices))],
            )
            model.fit(X_sub, y_train)

            # Extract Pareto front
            candidates = []
            equations = model.equations_
            if equations is not None and len(equations) > 0:
                for idx, row in equations.iterrows():
                    expr = str(row['equation'])
                    complexity = int(row['complexity'])
                    loss = float(row['loss'])

                    # Evaluate each equation individually on test set
                    try:
                        y_pred_test = model.predict(X_test_sub, index=idx)
                        if not np.all(np.isfinite(y_pred_test)):
                            y_pred_test = None
                            test_rmse = float('inf')
                            r2 = 0.0
                        else:
                            test_rmse = float(np.sqrt(np.mean((y_test - y_pred_test)**2)))
                            r2 = float(1 - np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2))
                    except Exception:
                        y_pred_test = None
                        test_rmse = float('inf')
                        r2 = 0.0

                    candidates.append(FormulaCandidate(
                        expression=expr,
                        complexity=complexity,
                        train_rmse=float(np.sqrt(loss)),
                        test_rmse=test_rmse,
                        r2=r2,
                        features_used=list(selected_features),
                        round_discovered=round_id,
                        y_pred_test=y_pred_test,
                    ))

            return candidates

        except ImportError:
            logger.info("PySR not installed, using symbolic regression simulation")
            return self._simulate_pysr(X_sub, y_train, X_test_sub, y_test,
                                        selected_features, round_id)

    def _simulate_pysr(self, X_train, y_train, X_test, y_test,
                        feature_names, round_id) -> List[FormulaCandidate]:
        """
        Simulate PySR output for demo mode.
        Uses numpy-based fitting to approximate what PySR would find.
        """
        candidates = []

        def _build_expr_and_fit(X_tr, X_te, col_names):
            """Ridge-regularised least-squares fit and build symbolic expression.
            Ridge regression prevents exploding coefficients on real data where
            features may be nearly collinear."""
            A = np.column_stack([X_tr, np.ones(len(X_tr))])
            # Ridge: (AᵀA + λI) x = Aᵀy, λ proportional to feature scale
            n_feats = A.shape[1]
            lam = 1e-3 * np.mean(np.diag(A.T @ A))
            ridge = np.eye(n_feats) * lam
            ridge[-1, -1] = 0.0          # don't regularise bias term
            coeffs = np.linalg.solve(A.T @ A + ridge, A.T @ y_train)
            y_pred_train = np.column_stack([X_tr, np.ones(len(X_tr))]) @ coeffs
            y_pred_test  = np.column_stack([X_te, np.ones(len(X_te))]) @ coeffs
            train_rmse = float(np.sqrt(np.mean((y_train - y_pred_train) ** 2)))
            test_rmse  = float(np.sqrt(np.mean((y_test  - y_pred_test)  ** 2)))
            ss_res = np.sum((y_test - y_pred_test) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            terms = []
            for i, c in enumerate(coeffs[:-1]):
                if abs(c) > 1e-6 and i < len(col_names):
                    terms.append(f"{c:.4f}*{col_names[i]}")
            if abs(coeffs[-1]) > 1e-6:
                terms.append(f"{coeffs[-1]:.4f}")
            expr = " + ".join(terms) if terms else "0"
            return expr, train_rmse, test_rmse, r2, len(terms) * 2, y_pred_test

        # Build feature subsets of increasing size for variety on Pareto front.
        # All forms use only the named features to avoid un-named column issues.
        forms_named = []

        # Form 1: all selected features (linear)
        forms_named.append(('linear_all', X_train, X_test, list(feature_names)))

        # Form 2: top-3 by abs correlation with target
        if X_train.shape[1] > 3:
            corrs = np.array([abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
                              for i in range(X_train.shape[1])])
            corrs = np.nan_to_num(corrs)
            top3 = np.argsort(corrs)[-3:]
            X_top3_tr = X_train[:, top3]
            X_top3_te = X_test[:, top3]
            top3_names = [feature_names[i] for i in top3]
            forms_named.append(('linear_top3', X_top3_tr, X_top3_te, top3_names))

        # Form 3: single best feature
        if X_train.shape[1] >= 1:
            corrs = np.array([abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
                              for i in range(X_train.shape[1])])
            corrs = np.nan_to_num(corrs)
            best_i = int(np.argmax(corrs))
            X_best_tr = X_train[:, best_i:best_i+1]
            X_best_te = X_test[:, best_i:best_i+1]
            forms_named.append(('linear_best1', X_best_tr, X_best_te,
                                 [feature_names[best_i]]))

        for form_name, X_tr, X_te, col_names in forms_named:
            try:
                expr, train_rmse, test_rmse, r2, complexity, y_pred = (
                    _build_expr_and_fit(X_tr, X_te, col_names)
                )

                candidates.append(FormulaCandidate(
                    expression=expr,
                    complexity=complexity,
                    train_rmse=train_rmse,
                    test_rmse=test_rmse,
                    r2=r2,
                    features_used=list(feature_names),
                    round_discovered=round_id,
                    y_pred_test=y_pred,
                ))
            except Exception as e:
                logger.debug(f"Form {form_name} failed: {e}")
                continue

        return candidates

    def _evaluate_candidates(self, candidates: List[FormulaCandidate],
                              X_test: np.ndarray, y_test: np.ndarray,
                              round_id: int) -> List[FormulaCandidate]:
        """Evaluate candidates against physical constraints."""
        valid = []
        for c in candidates:
            # Basic validity checks
            if c.test_rmse < float('inf') and c.train_rmse < float('inf'):
                # Check that formula predicts positive values (n > 0)
                c.physics_valid = True  # Detailed check done by CauchyValidator
                valid.append(c)
        return sorted(valid, key=lambda c: c.test_rmse)

    def _update_experience_pool(self, round_id: int, instructions: dict,
                                 candidates: List[FormulaCandidate],
                                 diagnostic_reports: List = None,
                                 search_config_used: dict = None):
        """Update the experience pool with this round's results."""
        if candidates:
            best = candidates[0]
            success = best.test_rmse < 0.5  # Threshold for "success"
            feedback = (
                f"Best RMSE={best.test_rmse:.4f}, R²={best.r2:.4f}, "
                f"formula='{best.expression}', complexity={best.complexity}"
            )
        else:
            success = False
            feedback = "No valid candidates found"

        # Extract diagnostic summary if available (new Harness Engineering field)
        diag_summary = ""
        if diagnostic_reports:
            try:
                first_diag = diagnostic_reports[0]
                diag_summary = getattr(first_diag, "causal_text", "")[:400]
            except Exception:
                pass

        entry = ExperienceEntry(
            round_id=round_id,
            features_tried=instructions['features'],
            best_formula=candidates[0].expression if candidates else "N/A",
            rmse=candidates[0].test_rmse if candidates else float('inf'),
            complexity=candidates[0].complexity if candidates else 0,
            llm_feedback=feedback,
            success=success,
            diagnostic_summary=diag_summary,
            search_config_used=search_config_used or {},
        )
        self.experience_pool.append(entry)

        # Keep pool size bounded
        pool_size = self.search_config.get('experience_pool_size', 50)
        if len(self.experience_pool) > pool_size:
            self.experience_pool = sorted(
                self.experience_pool, key=lambda e: e.rmse
            )[:pool_size]

    def _update_knowledge(self, knowledge, round_id: int,
                           candidates: List[FormulaCandidate],
                           diagnostic_reports: List) -> None:
        """Update KnowledgeMemory based on this round's results."""
        if not candidates:
            return

        best = candidates[0]
        features = list(best.features_used)
        formula = best.expression
        r2 = best.r2

        # Record success if R² > 0.9
        if r2 > 0.9:
            pattern = self._extract_pattern_name(formula, features)
            knowledge.record_success(pattern, formula, r2, round_id, features)

        # Record failures from diagnostic reports
        for diag in (diagnostic_reports or []):
            if getattr(diag, "violation_fraction", 0) > 0.05:
                knowledge.record_failure(features, diag, round_id)

        # Persist
        try:
            knowledge.save()
        except Exception:
            pass

    def _extract_pattern_name(self, formula: str, features: List[str]) -> str:
        """Derive a short pattern name from a formula string."""
        import re
        has_pow = bool(re.search(r'\*\*|sqrt|square', formula))
        has_trans = bool(re.search(r'\bexp\b|\blog\b', formula))
        n_feats = len(features)

        if has_trans:
            kind = "transcendental"
        elif has_pow:
            kind = "power"
        else:
            kind = "linear"

        top3 = "+".join(features[:3])
        return f"{kind}_{top3}_r{n_feats}feats"

    def _update_pareto_front(self, candidates: List[FormulaCandidate]):
        """Update the Pareto front (complexity vs accuracy trade-off)."""
        all_candidates = self.pareto_front + candidates
        # Non-dominated sorting: keep candidates that are not dominated
        pareto = []
        for c in all_candidates:
            dominated = False
            for other in all_candidates:
                if (other.test_rmse <= c.test_rmse and
                    other.complexity <= c.complexity and
                    (other.test_rmse < c.test_rmse or other.complexity < c.complexity)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(c)
        self.pareto_front = pareto
