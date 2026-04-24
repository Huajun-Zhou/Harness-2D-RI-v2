#!/usr/bin/env python3
"""
HARNESS-2D-RI: Main Entry Point
================================
Harness Engineering framework for discovering structure-refractive index
equations in 2D TMD materials via LLM-enhanced symbolic regression.

Usage:
    python main.py --config configs/harness_config.yaml --target cauchy_A
    python main.py --config configs/harness_config.yaml --target n_at_1064nm --rounds 10
"""

import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

from src.harness_orchestrator import HarnessOrchestrator
from src.data_pipeline import TMDDataPipeline
from src.llm_sr_engine import LLMSREngine
from src.cauchy_validator import CauchyValidator
from src.formula_evaluator import FormulaEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'results/harness_{datetime.now():%Y%m%d_%H%M%S}.log')
    ]
)
logger = logging.getLogger("HARNESS-2D-RI")


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="HARNESS-2D-RI: Discover structure-RI equations for 2D materials"
    )
    parser.add_argument(
        '--config', type=str, default='configs/harness_config.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--target', type=str, default='cauchy_A',
        choices=['cauchy_A', 'cauchy_B', 'n_at_1064nm', 'n_at_785nm'],
        help='Target property to discover equation for'
    )
    parser.add_argument(
        '--rounds', type=int, default=None,
        help='Override max LLM-guided search rounds'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run in demo mode with synthetic data (default when no --source given)'
    )
    parser.add_argument(
        '--source', type=str, default=None,
        choices=['jarvis', 'c2db', 'refractiveindex', 'csv', 'auto', 'synthetic'],
        help=(
            'Real data source to use instead of synthetic data.\n'
            '  jarvis          — JARVIS-DFT (pip install jarvis-tools, no download)\n'
            '  c2db            — C2DB ASE db (needs --c2db-path)\n'
            '  refractiveindex — experimental YAML clone (needs --ri-db-path)\n'
            '  csv             — pre-featurised CSV (needs --data-csv)\n'
            '  auto            — try all sources in priority order\n'
            '  synthetic       — physics-based synthetic data (same as --demo)'
        )
    )
    parser.add_argument(
        '--data-csv', type=str, default=None,
        dest='csv_path',
        help='Path to pre-featurised CSV (required when --source csv). '
             'Example: data/extended/tier2_binary.csv'
    )
    parser.add_argument(
        '--c2db-path', type=str, default=None,
        dest='c2db_path',
        help='Path to c2db.db ASE database file (required when --source c2db)'
    )
    parser.add_argument(
        '--ri-db-path', type=str, default=None,
        dest='ri_db_path',
        help='Path to refractiveindex.info-database clone (required when --source refractiveindex)'
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config)
    if args.rounds:
        config['llm_sr']['search']['max_rounds'] = args.rounds

    logger.info("=" * 70)
    logger.info("HARNESS-2D-RI: Harness Engineering for 2D Material RI Discovery")
    logger.info("=" * 70)
    logger.info(f"Target property: {args.target}")
    logger.info(f"Max search rounds: {config['llm_sr']['search']['max_rounds']}")

    # Resolve data source: --source wins; --demo sets synthetic; default = synthetic
    data_source = args.source or ('synthetic' if args.demo else 'synthetic')
    if data_source != 'synthetic':
        logger.info(f"  Data source: {data_source} (real data mode)")
    else:
        logger.info("  Data source: synthetic (use --source jarvis for real data)")

    # --- Layer 1: Data Pipeline ---
    logger.info("\n[Layer 1] Initializing data pipeline...")
    data_pipeline = TMDDataPipeline(
        config['data'],
        demo_mode=args.demo,
        data_source=data_source,
        c2db_path=args.c2db_path,
        ri_db_path=args.ri_db_path,
        csv_path=args.csv_path,
    )
    X_train, y_train, X_test, y_test, feature_names = data_pipeline.prepare(
        target=args.target
    )
    logger.info(f"  Training samples: {len(X_train)}, Features: {len(feature_names)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features: {feature_names}")

    # --- Layer 2: Physical Constraints ---
    logger.info("\n[Layer 2] Initializing physical constraints...")
    validator = CauchyValidator(config['constraints'])
    evaluator = FormulaEvaluator(
        constraints_config=config['constraints'],
        X_test=X_test, y_test=y_test
    )

    # --- Layer 3: LLM-SR Engine ---
    logger.info("\n[Layer 3] Initializing LLM-SR engine...")
    llm_sr = LLMSREngine(
        config=config['llm_sr'],
        feature_names=feature_names,
        target_name=args.target,
        constraints=config['constraints']
    )

    # --- Genuine Harness Engineering Components (Gaps 1-5) ---
    harness_components: dict = {}
    try:
        from src.causal_diagnostic_engine import CausalDiagnosticEngine
        from src.knowledge_memory import KnowledgeMemory
        from src.adaptive_search_environment import AdaptiveSearchEnvironment
        from agents.data_agent import DataAgent
        from agents.theory_agent import TheoryAgent
        from agents.validator_agent import ValidatorAgent
        from agents.meta_agent import MetaAgent
        from agents.agent_council import AgentCouncil

        output_dir = Path(config['output']['save_dir'])
        output_dir.mkdir(exist_ok=True)

        # Material metadata for causal diagnostics
        # 'phase' column only exists in synthetic data; real datasets use 'jid'/'uid' etc.
        if data_pipeline.df is not None:
            df_ = data_pipeline.df
            meta_cols = [c for c in ['metal', 'chalcogen', 'phase'] if c in df_.columns]
            if 'phase' not in df_.columns:
                df_ = df_.assign(phase=0)
            material_metadata = df_[['metal', 'chalcogen', 'phase']].copy()
        else:
            material_metadata = None

        # Gap 5: Structured knowledge memory
        km_path = Path(config.get('knowledge_memory', {}).get(
            'persistence_path', 'results/knowledge_memory.json'))
        knowledge = KnowledgeMemory(memory_path=km_path)

        # Gap 1+2: Causal diagnostic engine
        diagnostic_engine = CausalDiagnosticEngine(
            validator, feature_names, material_metadata)

        # Gap 4: Adaptive search environment
        adaptive_env = AdaptiveSearchEnvironment(config['llm_sr'], knowledge)

        # Gap 3: Deliberative agents
        agent_cfg = config.get('agents', {})
        delib_cfg = config.get('deliberation', {})
        data_ag = DataAgent('DataAgent', 'Feature analysis', knowledge, agent_cfg)
        theory_ag = TheoryAgent('TheoryAgent', 'Physics reasoning', knowledge, agent_cfg)
        validator_ag = ValidatorAgent(
            'ValidatorAgent', 'Physics validation', knowledge, agent_cfg,
            diagnostic_engine, X_train, y_train, X_test, y_test, feature_names)
        # Merge deliberation config into agent config for MetaAgent
        meta_cfg = {**agent_cfg, **delib_cfg}
        meta_ag = MetaAgent(
            'MetaAgent', 'Deliberation orchestration', knowledge, meta_cfg,
            [data_ag, theory_ag, validator_ag])
        council = AgentCouncil(data_ag, theory_ag, validator_ag, meta_ag, knowledge)

        harness_components = dict(
            agent_council=council,
            knowledge=knowledge,
            adaptive_env=adaptive_env,
            diagnostic_engine=diagnostic_engine,
        )
        logger.info("[Harness Engineering] Genuine Harness components initialised "
                    "(gaps 1-5 active)")
    except Exception as e:
        logger.warning(f"[Harness Engineering] Falling back to legacy pipeline mode: {e}")

    # --- Layer 4 + 5: Harness Orchestrator (Multi-Agent + Human Interface) ---
    logger.info("\n[Layer 4-5] Launching Harness Orchestrator...")
    orchestrator = HarnessOrchestrator(
        config=config,
        data_pipeline=data_pipeline,
        llm_sr_engine=llm_sr,
        validator=validator,
        evaluator=evaluator,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        target_name=args.target,
        **harness_components,
    )

    # --- Execute Harness Loop ---
    results = orchestrator.run()

    # --- Report Results ---
    logger.info("\n" + "=" * 70)
    logger.info("HARNESS DISCOVERY COMPLETE")
    logger.info("=" * 70)

    if results['best_formula']:
        logger.info(f"\n  Best Formula: {results['best_formula']}")
        logger.info(f"  Train RMSE:   {results['train_rmse']:.6f}")
        logger.info(f"  Test RMSE:    {results['test_rmse']:.6f}")
        logger.info(f"  R² Score:     {results['r2']:.6f}")
        logger.info(f"  Complexity:   {results['complexity']}")
        logger.info(f"  Physics OK:   {results['physics_valid']}")
        logger.info(f"  Rounds Used:  {results['rounds_used']}")

        # Export results
        output_dir = Path(config['output']['save_dir'])
        output_dir.mkdir(exist_ok=True)
        orchestrator.export_results(output_dir)
        logger.info(f"\n  Results saved to: {output_dir}")
    else:
        logger.warning("  No valid formula discovered. Try increasing rounds or relaxing constraints.")

    return results


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    main()
