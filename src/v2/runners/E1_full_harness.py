"""
E1 — Full 4-agent Harness reference run on Tier-2.

This is the first experiment in the v2 campaign. It reruns the Harness
workflow on the same Tier-2 split v1 used (seed=42, 80/20 of N=193), but
with the DataAgent runtime bug fixed so all four council agents actually
vote during deliberation.

Outputs (written to run_dir):
  - manifest.json            : already written by harness_runner.py
  - harness_run.log          : full DEBUG log of the orchestrator
  - best_formula.tex         : LaTeX-wrapped headline expression
  - discovery_report.json    : timestamps, features used, metrics, pareto size
  - pareto_front.json        : full Pareto front across all rounds
  - experience_pool.json     : knowledge memory final state
  - agent_vote_tally.json    : per-round breakdown of agent votes (E1 new)

The `agent_vote_tally.json` is the piece v1 could not produce because
DataAgent was silently ABSTAINing. It is the primary evidence for the
claim "the fixed council has four active agents".
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


log = logging.getLogger("v2.runners.E1")


# -- helpers ---------------------------------------------------------------


def _install_file_log(run_dir: Path) -> logging.Handler:
    """Attach a file handler to HARNESS loggers so every log line is captured."""
    fh = logging.FileHandler(run_dir / "harness_run.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger("HARNESS").addHandler(fh)
    logging.getLogger("v2").addHandler(fh)
    return fh


def _parse_vote_tally(log_path: Path) -> dict[str, Any]:
    """
    Scan the orchestrator log and extract evidence that each agent actually
    participated in deliberation. Three signals at INFO level:

      * "DataAgent analysis: ..."           → DataAgent ran and produced output
      * "TheoryAgent: ... operator..."      → TheoryAgent ran
      * "[AgentCouncil] Pre-search done:"   → MetaAgent closed deliberation
      * "Agent X vote failed: ..."          → v1-style silent failure (BAD)

    The v1 bug was that DataAgent's vote() always raised NameError, so the
    "DataAgent analysis" line was still emitted (from analyze() which was
    called separately) but the agent's VOTE was never counted. Here we
    additionally parse the per-agent vote trace if it's in the log at
    DEBUG level, and fall back to the analysis-line-presence signal if
    only INFO is captured.
    """
    if not log_path.is_file():
        return {"log_missing": True}

    # (agent, vote) counts from DEBUG-level MetaAgent trace.
    vote_counts: dict[tuple[str, str], int] = Counter()
    failures: list[dict[str, str]] = []
    activity_counts: Counter[str] = Counter()

    vote_re = re.compile(
        r"(DataAgent|TheoryAgent|ValidatorAgent|MetaAgent)\s+voted\s+"
        r"(APPROVE|VETO|MODIFY|ABSTAIN)",
    )
    fail_re = re.compile(
        r"Agent (\w+) vote failed: (.+?)(?:\s*$)",
    )
    activity_res = {
        "DataAgent":      re.compile(r"\bDataAgent analysis:"),
        "TheoryAgent":    re.compile(r"\bTheoryAgent:\s*\d+ operator"),
        "MetaAgent":      re.compile(r"\bMetaAgent: Deliberation on"),
        # ValidatorAgent only logs at veto time or when it approves a formula;
        # count post-search approval tallies as evidence of participation.
        "ValidatorAgent": re.compile(
            r"\[AgentCouncil\] Post-search(?: deliberation)?: "
        ),
    }

    for line in log_path.read_text().splitlines():
        m = vote_re.search(line)
        if m:
            vote_counts[(m.group(1), m.group(2))] += 1
        m = fail_re.search(line)
        if m:
            failures.append({"agent": m.group(1), "error": m.group(2)})
        for ag, regex in activity_res.items():
            if regex.search(line):
                activity_counts[ag] += 1

    per_agent: dict[str, dict[str, int]] = defaultdict(dict)
    for (ag, vt), n in vote_counts.items():
        per_agent[ag][vt] = n

    data_active = (activity_counts.get("DataAgent", 0) > 0)

    return {
        "per_agent_votes_debug": per_agent,
        "per_agent_activity_counts": dict(activity_counts),
        "failures": failures,
        "total_failures": len(failures),
        "data_agent_active": bool(data_active),
    }


# -- main entry ------------------------------------------------------------


def run(config_path: Path, run_seed: int, run_dir: Path) -> dict[str, Any]:
    """
    Execute the E1 campaign and return a summary dict. The caller
    (`harness_runner.py`) is responsible for writing the manifest; this
    function writes everything else into `run_dir`.
    """
    file_handler = _install_file_log(run_dir)

    # Lazy imports so that the rest of the package isn't polluted by heavy
    # orchestrator imports during `python harness_runner.py --help`.
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

    # -- config --
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve E1-specific settings. E1 config may extend / override
    # harness_config.yaml via a `__extends__` entry for now we just merge
    # whatever is provided.
    e1_target = config.get("target", "cauchy_A")
    e1_data_csv = config.get("data_csv", "data/extended/tier2_binary.csv")

    log.info("E1 full-Harness reference run starting")
    log.info("  target:    %s", e1_target)
    log.info("  data_csv:  %s", e1_data_csv)
    log.info("  run_seed:  %s", run_seed)
    log.info("  run_dir:   %s", run_dir)

    # -- data --
    log.info("[1/5] Loading Tier-2 via data_pipeline (csv source)")
    data_pipeline = TMDDataPipeline(
        config["data"],
        demo_mode=False,
        data_source="csv",
        csv_path=e1_data_csv,
    )
    X_train, y_train, X_test, y_test, feature_names = data_pipeline.prepare(
        target=e1_target
    )
    log.info("  train=%d test=%d features=%d",
             len(X_train), len(X_test), len(feature_names))

    # -- physical constraints --
    validator = CauchyValidator(config["constraints"])
    evaluator = FormulaEvaluator(
        constraints_config=config["constraints"],
        X_test=X_test, y_test=y_test,
    )

    # -- LLM-SR engine --
    log.info("[2/5] Initialising LLM-SR engine")
    llm_sr = LLMSREngine(
        config=config["llm_sr"],
        feature_names=feature_names,
        target_name=e1_target,
        constraints=config["constraints"],
    )

    # -- governance layer --
    log.info("[3/5] Initialising governance layer (4-agent council)")
    if data_pipeline.df is not None:
        df_ = data_pipeline.df
        if "phase" not in df_.columns:
            df_ = df_.assign(phase=0)
        for col in ("metal", "chalcogen"):
            if col not in df_.columns:
                # csv source uses 'anion' not 'chalcogen'
                df_[col] = df_.get(col, df_.get("anion", ""))
        material_metadata = df_[["metal", "chalcogen", "phase"]].copy()
    else:
        material_metadata = None

    km_path = run_dir / "knowledge_memory.json"
    knowledge = KnowledgeMemory(memory_path=km_path)
    diagnostic_engine = CausalDiagnosticEngine(
        validator, feature_names, material_metadata
    )
    adaptive_env = AdaptiveSearchEnvironment(config["llm_sr"], knowledge)

    agent_cfg = config.get("agents", {})
    delib_cfg = config.get("deliberation", {})
    data_ag = DataAgent("DataAgent", "Feature analysis", knowledge, agent_cfg)
    theory_ag = TheoryAgent("TheoryAgent", "Physics reasoning", knowledge, agent_cfg)
    validator_ag = ValidatorAgent(
        "ValidatorAgent", "Physics validation", knowledge, agent_cfg,
        diagnostic_engine, X_train, y_train, X_test, y_test, feature_names,
    )
    meta_ag = MetaAgent(
        "MetaAgent", "Deliberation orchestration", knowledge,
        {**agent_cfg, **delib_cfg},
        [data_ag, theory_ag, validator_ag],
    )
    council = AgentCouncil(data_ag, theory_ag, validator_ag, meta_ag, knowledge)

    # -- orchestrator --
    log.info("[4/5] Launching Harness Orchestrator")
    orchestrator = HarnessOrchestrator(
        config=config,
        data_pipeline=data_pipeline,
        llm_sr_engine=llm_sr,
        validator=validator,
        evaluator=evaluator,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        target_name=e1_target,
        agent_council=council,
        knowledge=knowledge,
        adaptive_env=adaptive_env,
        diagnostic_engine=diagnostic_engine,
    )
    results = orchestrator.run()

    # -- persist --
    log.info("[5/5] Persisting outputs to %s", run_dir)
    orchestrator.export_results(run_dir)

    # Write a summary + agent vote tally (the main E1-specific artefact).
    summary = {
        "run_seed": run_seed,
        "target": e1_target,
        "data_csv": e1_data_csv,
        "best_formula": results.get("best_formula"),
        "train_rmse": float(results.get("train_rmse", float("nan"))),
        "test_rmse": float(results.get("test_rmse", float("nan"))),
        "r2": float(results.get("r2", float("nan"))),
        "complexity": results.get("complexity"),
        "rounds_used": results.get("rounds_used"),
        "physics_valid": bool(results.get("physics_valid")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features_used": results.get("features_used", []),
    }
    (run_dir / "e1_summary.json").write_text(json.dumps(summary, indent=2))

    vote_tally = _parse_vote_tally(run_dir / "harness_run.log")
    (run_dir / "agent_vote_tally.json").write_text(
        json.dumps(vote_tally, indent=2, default=str)
    )

    # Clean up the file handler so subsequent experiments don't double-log.
    logging.getLogger("HARNESS").removeHandler(file_handler)
    logging.getLogger("v2").removeHandler(file_handler)
    file_handler.close()

    log.info("E1 complete.")
    log.info("  headline: %s", summary["best_formula"])
    log.info("  test RMSE: %.4f, R²: %.4f, complexity: %s",
             summary["test_rmse"], summary["r2"], summary["complexity"])
    if vote_tally.get("log_missing"):
        log.warning("  vote tally: log file missing")
    else:
        log.info("  total vote failures: %d", vote_tally["total_failures"])
        log.info("  data_agent_active:   %s", vote_tally["data_agent_active"])
    return summary
