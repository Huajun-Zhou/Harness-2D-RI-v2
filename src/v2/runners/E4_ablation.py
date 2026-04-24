"""
E4 — Leave-one-module-out ablation on the fixed 4-agent council.

Re-runs the v1 "A5" ablation but on the DataAgent-fixed pipeline, so each
governance module is genuinely present (or genuinely removed) rather
than silently inactive like v1. Measures the held-out-slice RMSE
contribution of each governance layer.

Five variants, 3 seeds each. Same 80/20 Tier-2 split E1 uses (seed=42),
reduced per-variant budget (5 rounds × 40 islands × 50 iterations) to
keep the total campaign compact (~15 runs × ~30s = ~8 min).

Variants:
  FULL         — all four governance modules active (matches E1)
  -AGENTS      — no AgentCouncil (PySR + LLM + adaptive + memory)
  -CAUSAL      — no CausalDiagnosticEngine (agents have no sensitivity feedback)
  -ADAPTIVE    — no AdaptiveSearchEnvironment (operator set frozen)
  -MEMORY      — no KnowledgeMemory (no cross-round success/failure recall)

The config carries:
  variant       : one of the five above
  n_seeds       : default 3
  n_rounds      : default 5
  (+ standard harness_config keys)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


log = logging.getLogger("v2.runners.E4")


VARIANT_TO_DISABLE = {
    "FULL":      set(),
    "-AGENTS":   {"agent_council"},
    "-CAUSAL":   {"diagnostic_engine"},
    "-ADAPTIVE": {"adaptive_env"},
    "-MEMORY":   {"knowledge"},
}


def _install_file_log(run_dir: Path) -> logging.Handler:
    fh = logging.FileHandler(run_dir / "e4_run.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger("HARNESS").addHandler(fh)
    logging.getLogger("v2").addHandler(fh)
    return fh


def _single_run(
    variant: str, seed_idx: int, run_seed: int, run_dir: Path, cfg: dict,
) -> dict[str, Any]:
    """One (variant, seed) Harness run. Disables whichever module the
    variant names."""
    from src.cauchy_validator import CauchyValidator
    from src.formula_evaluator import FormulaEvaluator
    from src.llm_sr_engine import LLMSREngine
    from src.data_pipeline import TMDDataPipeline
    from src.harness_orchestrator import HarnessOrchestrator
    from src.causal_diagnostic_engine import CausalDiagnosticEngine
    from src.knowledge_memory import KnowledgeMemory
    from src.adaptive_search_environment import AdaptiveSearchEnvironment
    from agents.data_agent import DataAgent
    from agents.theory_agent import TheoryAgent
    from agents.validator_agent import ValidatorAgent
    from agents.meta_agent import MetaAgent
    from agents.agent_council import AgentCouncil
    from src.v2.seeds import SeedManager

    sm = SeedManager(run_seed=run_seed)
    # Seed for this particular (variant, seed_idx) combination
    this_seed = int(sm.derive("e4_single", variant=variant, seed_idx=seed_idx))
    disable = VARIANT_TO_DISABLE[variant]

    # Reduced per-run budget so the whole 15-run campaign is quick.
    cfg_local = dict(cfg)
    cfg_local["llm_sr"] = dict(cfg["llm_sr"])
    cfg_local["llm_sr"]["search"] = dict(cfg["llm_sr"]["search"])
    cfg_local["llm_sr"]["search"]["max_rounds"] = cfg.get("n_rounds", 5)
    cfg_local["llm_sr"]["pysr"] = dict(cfg["llm_sr"]["pysr"])
    cfg_local["llm_sr"]["pysr"]["niterations"] = cfg.get("niterations", 50)

    log.info("  run variant=%s seed_idx=%d this_seed=%d disable=%s",
             variant, seed_idx, this_seed, disable or "{}")

    # Data
    data_pipeline = TMDDataPipeline(
        cfg_local["data"], demo_mode=False, data_source="csv",
        csv_path=cfg_local.get("data_csv", "data/extended/tier2_binary.csv"),
    )
    X_train, y_train, X_test, y_test, feat_names = data_pipeline.prepare(
        target=cfg_local.get("target", "cauchy_A"),
    )

    # Physical constraints
    validator = CauchyValidator(cfg_local["constraints"])
    evaluator = FormulaEvaluator(
        constraints_config=cfg_local["constraints"],
        X_test=X_test, y_test=y_test,
    )
    llm_sr = LLMSREngine(
        config=cfg_local["llm_sr"], feature_names=feat_names,
        target_name=cfg_local.get("target", "cauchy_A"),
        constraints=cfg_local["constraints"],
    )

    # Metadata for the optional diagnostic engine
    if data_pipeline.df is not None:
        df_ = data_pipeline.df
        if "phase" not in df_.columns:
            df_ = df_.assign(phase=0)
        for col in ("metal", "chalcogen"):
            if col not in df_.columns:
                df_[col] = df_.get(col, df_.get("anion", ""))
        material_metadata = df_[["metal", "chalcogen", "phase"]].copy()
    else:
        material_metadata = None

    # Build governance modules, then zero-out the ones this variant disables
    knowledge = None if "knowledge" in disable else KnowledgeMemory(
        memory_path=run_dir / f"knowledge_memory_{variant}_s{seed_idx}.json"
    )
    diagnostic_engine = None if "diagnostic_engine" in disable else \
        CausalDiagnosticEngine(validator, feat_names, material_metadata)
    adaptive_env = None if "adaptive_env" in disable else \
        AdaptiveSearchEnvironment(cfg_local["llm_sr"], knowledge)

    # The agent council is a nested structure — disabling it zeros out the
    # whole council, not individual agents.
    if "agent_council" in disable:
        council = None
    else:
        agent_cfg = cfg_local.get("agents", {})
        delib_cfg = cfg_local.get("deliberation", {})
        data_ag = DataAgent("DataAgent", "Feature analysis", knowledge, agent_cfg)
        theory_ag = TheoryAgent("TheoryAgent", "Physics reasoning", knowledge, agent_cfg)
        validator_ag = ValidatorAgent(
            "ValidatorAgent", "Physics validation", knowledge, agent_cfg,
            diagnostic_engine, X_train, y_train, X_test, y_test, feat_names,
        )
        meta_ag = MetaAgent(
            "MetaAgent", "Deliberation", knowledge,
            {**agent_cfg, **delib_cfg},
            [data_ag, theory_ag, validator_ag],
        )
        council = AgentCouncil(data_ag, theory_ag, validator_ag, meta_ag, knowledge)

    orchestrator = HarnessOrchestrator(
        config=cfg_local, data_pipeline=data_pipeline,
        llm_sr_engine=llm_sr, validator=validator, evaluator=evaluator,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feat_names, target_name=cfg_local.get("target", "cauchy_A"),
        agent_council=council, knowledge=knowledge,
        adaptive_env=adaptive_env, diagnostic_engine=diagnostic_engine,
    )
    results = orchestrator.run()

    return {
        "variant": variant,
        "seed_idx": seed_idx,
        "this_seed": this_seed,
        "best_formula": results.get("best_formula"),
        "complexity": results.get("complexity"),
        "train_rmse": float(results.get("train_rmse", float("nan"))),
        "test_rmse": float(results.get("test_rmse", float("nan"))),
        "r2": float(results.get("r2", float("nan"))),
        "rounds_used": results.get("rounds_used"),
    }


def run(config_path: Path, run_seed: int, run_dir: Path) -> dict[str, Any]:
    """
    Run all five ablation variants, each over n_seeds independent seeds,
    in a single process. Returns the aggregated table.
    """
    file_handler = _install_file_log(run_dir)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_seeds = int(cfg.get("n_seeds", 3))
    variants = list(VARIANT_TO_DISABLE.keys())
    log.info("E4 ablation: %d variants × %d seeds = %d runs",
             len(variants), n_seeds, len(variants) * n_seeds)

    all_runs: list[dict] = []
    for variant in variants:
        for seed_idx in range(n_seeds):
            t0 = time.time()
            try:
                row = _single_run(variant, seed_idx, run_seed, run_dir, cfg)
                row["wall_seconds"] = time.time() - t0
                all_runs.append(row)
                log.info("    done variant=%s seed=%d  "
                         "cplx=%s test=%.4f  (+%.1fs)",
                         variant, seed_idx, row["complexity"],
                         row["test_rmse"], row["wall_seconds"])
            except Exception as e:
                log.error("    FAILED variant=%s seed=%d: %s", variant, seed_idx, e)
                all_runs.append({
                    "variant": variant, "seed_idx": seed_idx,
                    "error": str(e),
                })

    # Aggregate per-variant
    summary: dict[str, Any] = {}
    for v in variants:
        v_runs = [r for r in all_runs if r["variant"] == v and "test_rmse" in r]
        test_rmses = [r["test_rmse"] for r in v_runs]
        cplxs = [r["complexity"] for r in v_runs if r["complexity"] is not None]
        def _ms(xs):
            if not xs: return (float("nan"), float("nan"))
            a = np.asarray(xs, dtype=float)
            return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
        m_t, s_t = _ms(test_rmses)
        m_c, s_c = _ms(cplxs)
        summary[v] = {
            "n_seeds_success": len(v_runs),
            "mean_test_rmse": m_t, "std_test_rmse": s_t,
            "mean_complexity": m_c, "std_complexity": s_c,
        }

    # Δ vs FULL
    full = summary.get("FULL")
    if full and not np.isnan(full["mean_test_rmse"]):
        for v in variants:
            if v == "FULL":
                continue
            if "mean_test_rmse" in summary[v]:
                summary[v]["delta_vs_FULL"] = (
                    summary[v]["mean_test_rmse"] - full["mean_test_rmse"]
                )

    output = {
        "run_seed": run_seed,
        "n_seeds": n_seeds,
        "all_runs": all_runs,
        "summary": summary,
    }
    (run_dir / "ablation_summary.json").write_text(json.dumps(output, indent=2))

    log.info("")
    log.info("E4 ablation summary (held-out-slice RMSE):")
    for v in variants:
        s = summary[v]
        delta = f"Δ={s.get('delta_vs_FULL', 0):+.4f}" if "delta_vs_FULL" in s else "reference"
        log.info("  %-12s %.4f ± %.4f   cplx %.1f ± %.1f   %s",
                 v, s["mean_test_rmse"], s["std_test_rmse"],
                 s["mean_complexity"], s["std_complexity"], delta)

    logging.getLogger("HARNESS").removeHandler(file_handler)
    logging.getLogger("v2").removeHandler(file_handler)
    file_handler.close()
    return output
