"""
A5 — Leave-one-out module ablation on Tier 2.

Runs 5 variants of the Harness pipeline, each time disabling exactly one
governance module:
  FULL        — all four modules active
  -AGENTS     — no agent_council
  -CAUSAL     — no diagnostic_engine
  -ADAPTIVE   — no adaptive_env
  -MEMORY     — no knowledge memory

Each variant is repeated over N_SEEDS PySR seeds (PySR itself is
non-deterministic); we report mean ± std test RMSE / R² and the
best expression per variant.

Reduced-budget config: 5 rounds × 40 islands × 50 iterations, veto enabled.
A single variant-seed combination takes ~4-5 min; 5 × 3 = 15 runs → ~60-75 min.

Results:
  results/ablation_loo/<variant>_seed<N>.json   per-run verdicts
  results/ablation_loo_summary.csv              aggregated table
  results/ablation_loo_report.md                markdown summary
"""

import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Silence the orchestrator's info-level spam; keep warnings+
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logging.getLogger("HARNESS-2D-RI").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.harness_orchestrator import HarnessOrchestrator
from src.data_pipeline import TMDDataPipeline
from src.llm_sr_engine import LLMSREngine
from src.cauchy_validator import CauchyValidator
from src.formula_evaluator import FormulaEvaluator

from src.causal_diagnostic_engine import CausalDiagnosticEngine
from src.knowledge_memory import KnowledgeMemory
from src.adaptive_search_environment import AdaptiveSearchEnvironment
from agents.data_agent import DataAgent
from agents.theory_agent import TheoryAgent
from agents.validator_agent import ValidatorAgent
from agents.meta_agent import MetaAgent
from agents.agent_council import AgentCouncil


VARIANTS = [
    ('FULL',      set()),
    ('-AGENTS',   {'agent_council'}),
    ('-CAUSAL',   {'diagnostic_engine'}),
    ('-ADAPTIVE', {'adaptive_env'}),
    ('-MEMORY',   {'knowledge'}),
]

N_SEEDS = 3
REDUCED_ROUNDS = 5
REDUCED_ITERS = 50


def build_pipeline(config, ablate: set):
    """Build a fresh pipeline with the named modules zeroed out."""
    data_pipeline = TMDDataPipeline(
        config['data'],
        demo_mode=False,
        data_source='csv',
        csv_path='data/extended/tier2_binary.csv',
    )
    X_train, y_train, X_test, y_test, feat_names = data_pipeline.prepare(target='cauchy_A')

    validator = CauchyValidator(config['constraints'])
    evaluator = FormulaEvaluator(constraints_config=config['constraints'],
                                 X_test=X_test, y_test=y_test)

    llm_sr = LLMSREngine(config=config['llm_sr'],
                         feature_names=feat_names,
                         target_name='cauchy_A',
                         constraints=config['constraints'])

    # Build every component; then drop the ablated ones.
    components: dict = {}
    df_ = data_pipeline.df
    if 'phase' not in df_.columns:
        df_ = df_.assign(phase=0)
    material_metadata = df_[['metal', 'chalcogen', 'phase']].copy()

    km_path = Path(f"results/ablation_loo/km_{time.time_ns()}.json")
    knowledge = KnowledgeMemory(memory_path=km_path)
    diagnostic_engine = CausalDiagnosticEngine(validator, feat_names, material_metadata)
    adaptive_env = AdaptiveSearchEnvironment(config['llm_sr'], knowledge)

    agent_cfg = config.get('agents', {})
    delib_cfg = config.get('deliberation', {})
    data_ag = DataAgent('DataAgent', 'Feature analysis', knowledge, agent_cfg)
    theory_ag = TheoryAgent('TheoryAgent', 'Physics reasoning', knowledge, agent_cfg)
    validator_ag = ValidatorAgent(
        'ValidatorAgent', 'Physics validation', knowledge, agent_cfg,
        diagnostic_engine, X_train, y_train, X_test, y_test, feat_names)
    meta_cfg = {**agent_cfg, **delib_cfg}
    meta_ag = MetaAgent('MetaAgent', 'Deliberation', knowledge, meta_cfg,
                        [data_ag, theory_ag, validator_ag])
    council = AgentCouncil(data_ag, theory_ag, validator_ag, meta_ag, knowledge)

    components['agent_council']     = None if 'agent_council'     in ablate else council
    components['knowledge']         = None if 'knowledge'         in ablate else knowledge
    components['adaptive_env']      = None if 'adaptive_env'      in ablate else adaptive_env
    components['diagnostic_engine'] = None if 'diagnostic_engine' in ablate else diagnostic_engine

    orchestrator = HarnessOrchestrator(
        config=config, data_pipeline=data_pipeline,
        llm_sr_engine=llm_sr, validator=validator, evaluator=evaluator,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        feature_names=feat_names, target_name='cauchy_A',
        **components,
    )
    return orchestrator


def run_variant(variant_name: str, ablate: set, seed: int, config_base: dict):
    """Run one variant × seed combination."""
    config = deepcopy(config_base)
    # Reduced budget: keep islands/pop, cut iterations and rounds
    config['llm_sr']['search']['max_rounds'] = REDUCED_ROUNDS
    config['llm_sr']['pysr']['niterations'] = REDUCED_ITERS
    # The output dir is shared; we don't collide because each run writes into memory
    config['output']['save_dir'] = f"results/ablation_loo"

    np.random.seed(seed)
    t0 = time.time()
    print(f"\n[{datetime.now():%H:%M:%S}] ▶ {variant_name}  seed={seed}  "
          f"(ablating: {sorted(ablate) or '∅'})", flush=True)
    try:
        orchestrator = build_pipeline(config, ablate)
        result = orchestrator.run()
    except Exception as e:
        print(f"  ✗ run failed: {e}", flush=True)
        result = {'best_formula': None, 'train_rmse': float('nan'),
                  'test_rmse': float('nan'), 'r2': float('nan'),
                  'complexity': float('nan'), 'rounds_used': 0,
                  'physics_valid': False, 'error': str(e)}

    result['variant']   = variant_name
    result['seed']      = seed
    result['ablated']   = sorted(ablate)
    result['wallclock'] = round(time.time() - t0, 1)

    out_dir = Path('results/ablation_loo')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant_name.replace('-', 'NO_')}_seed{seed}.json"
    out_path.write_text(json.dumps(
        {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
         for k, v in result.items() if k != 'best_candidate'}, indent=2,
        default=str))

    print(f"  → RMSE_test={result.get('test_rmse', float('nan')):.3f}  "
          f"R²={result.get('r2', float('nan')):+.3f}  "
          f"cplx={result.get('complexity', 'NA')}  "
          f"⏱ {result['wallclock']}s", flush=True)
    return result


def main():
    with open('configs/harness_config.yaml') as f:
        config_base = yaml.safe_load(f)

    all_runs = []
    t_all = time.time()
    for variant_name, ablate in VARIANTS:
        for seed in range(N_SEEDS):
            r = run_variant(variant_name, ablate, seed, config_base)
            all_runs.append(r)

    # ---- Aggregate --------------------------------------------------------
    df = pd.DataFrame([{
        'variant':    r['variant'],
        'seed':       r['seed'],
        'train_rmse': r.get('train_rmse', float('nan')),
        'test_rmse':  r.get('test_rmse',  float('nan')),
        'r2':         r.get('r2',         float('nan')),
        'complexity': r.get('complexity', float('nan')),
        'formula':    r.get('best_formula') or '(failed)',
        'wallclock':  r.get('wallclock', float('nan')),
    } for r in all_runs])

    agg = df.groupby('variant').agg(
        test_rmse_mean=('test_rmse', 'mean'),
        test_rmse_std =('test_rmse', 'std'),
        r2_mean       =('r2', 'mean'),
        r2_std        =('r2', 'std'),
        cplx_mean     =('complexity', 'mean'),
    ).reset_index()
    # Preserve VARIANTS order
    order = {v[0]: i for i, v in enumerate(VARIANTS)}
    agg['_ord'] = agg['variant'].map(order)
    agg = agg.sort_values('_ord').drop(columns='_ord')

    print("\n" + "=" * 60)
    print("LOO Ablation summary  (mean ± std across seeds):")
    print(agg.to_string(index=False))

    df.to_csv('results/ablation_loo_runs.csv', index=False)
    agg.to_csv('results/ablation_loo_summary.csv', index=False)
    print(f"\nTotal wall time: {(time.time() - t_all) / 60:.1f} min")

    # ---- Markdown report --------------------------------------------------
    md = ["# A5 — Leave-one-out module ablation (Tier 2, N=193)\n\n"]
    md.append(f"Budget per run: {REDUCED_ROUNDS} rounds × "
              f"{config_base['llm_sr']['pysr']['populations']} islands × "
              f"{REDUCED_ITERS} iterations.   N_seeds={N_SEEDS}.\n\n")
    md.append("| variant | test RMSE | R² | complexity |\n|---|---|---|---|\n")
    full_row = agg[agg['variant'] == 'FULL'].iloc[0]
    full_rmse = full_row['test_rmse_mean']
    for _, row in agg.iterrows():
        delta = row['test_rmse_mean'] - full_rmse
        md.append(
            f"| `{row['variant']}` | "
            f"{row['test_rmse_mean']:.3f} ± {row['test_rmse_std']:.3f}"
            f"{' (Δ=+' + f'{delta:.3f}' + ')' if row['variant'] != 'FULL' and not np.isnan(delta) else ''} | "
            f"{row['r2_mean']:+.3f} ± {row['r2_std']:.3f} | "
            f"{row['cplx_mean']:.1f} |\n"
        )
    md.append("\n_Positive Δ = removing this module HURTS; "
              "negative Δ = removing it HELPS (candidate for deletion)._\n\n")

    # Per-variant best formula (seed with min test_rmse)
    md.append("## Best formula per variant\n\n")
    for variant_name, _ in VARIANTS:
        sub = df[df['variant'] == variant_name].sort_values('test_rmse')
        if len(sub) == 0:
            continue
        best = sub.iloc[0]
        md.append(
            f"- **{variant_name}** (seed {int(best['seed'])}, "
            f"test RMSE {best['test_rmse']:.3f}):  \n"
            f"  `{best['formula']}`\n"
        )
    Path('results/ablation_loo_report.md').write_text(''.join(md))
    print("Saved → results/ablation_loo_summary.csv, ablation_loo_report.md")


if __name__ == '__main__':
    main()
