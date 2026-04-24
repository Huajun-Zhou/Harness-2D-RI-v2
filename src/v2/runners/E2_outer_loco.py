"""
E2 — Outer-LOCO Harness discovery.

This is the reviewer's main ask and the strongest generalisation claim the
paper can make. For each target anion `a` in the 10-anion Tier-2 alphabet,
we:

  1. Partition Tier-2 → (outer_test = samples with anion==a,
                          outer_pool = samples with anion!=a).
  2. Permutation-split outer_pool 80/20 (seeded per fold) into
     (inner_train, inner_screening).
  3. Run the full 4-agent Harness on (inner_train, inner_screening) — the
     agent council, knowledge memory, adaptive operators, and PySR all
     only ever see the 9 non-`a` anions.
  4. Freeze the headline + full Pareto front.
  5. Evaluate the frozen headline and every Pareto candidate on outer_test
     (the a-only anion samples) — this is the leakage-free test metric.

The paper's §5 "Matched baselines" (E3) reuses the same outer splits so
Harness vs Vanilla-PySR can be compared fold-for-fold.

This runner deliberately does NOT call `data_pipeline.prepare()` (which
hardcodes seed=42). It loads the CSV directly and builds the feature
matrices itself, using the same 20-feature Harness dictionary as E1.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


log = logging.getLogger("v2.runners.E2")


# 17 shared descriptors (see paper §2 descriptor catalogue) plus the three
# Harness-only structural extras. Order must match what PySR will see as
# x0..xN. Matches E1's feature vector.
E2_FEATURE_NAMES = [
    "atomic_number_M",  "atomic_number_X",
    "electronegativity_M", "electronegativity_X",
    "atomic_radius_M",  "atomic_radius_X",
    "ionization_energy_M", "electron_affinity_X",
    "d_electron_count",
    "bond_length_MX",   "lattice_constant_a", "monolayer_thickness",
    "coordination_number", "point_group_encoded",
    "electronegativity_diff", "radius_ratio",
    "bond_polarity", "packing_fraction",
    "Z_ratio", "IE_EA_ratio",
]

# Tier-2 CSV uses short-form names; map long → short where they differ.
_LONG_TO_SHORT = {
    "atomic_number_M": "Z_M", "atomic_number_X": "Z_X",
    "electronegativity_M": "chi_M", "electronegativity_X": "chi_X",
    "atomic_radius_M": "r_M", "atomic_radius_X": "r_X",
    "ionization_energy_M": "IE_M", "electron_affinity_X": "EA_X",
    "d_electron_count": "d_e",
    "bond_length_MX": "d_MX",
    "lattice_constant_a": "a_lat",
    "monolayer_thickness": "t",
    "electronegativity_diff": "abs_delta_chi",
    "radius_ratio": "r_ratio",
    # same-name passthrough:
    "bond_polarity": "bond_polarity", "Z_ratio": "Z_ratio",
    "IE_EA_ratio": "IE_EA_ratio",
}


# -- helpers -----------------------------------------------------------------


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols: list[np.ndarray] = []
    for long_name in E2_FEATURE_NAMES:
        short = _LONG_TO_SHORT.get(long_name)
        if short and short in df.columns:
            cols.append(df[short].to_numpy(dtype=float))
        else:
            # Structural extras not present in the flat CSV (coordination_number,
            # point_group_encoded, packing_fraction) — fill zeros to match
            # E1's feature vector; same zero-fill convention as the smoke test.
            cols.append(np.zeros(len(df), dtype=float))
    return np.column_stack(cols)


def _install_file_log(run_dir: Path) -> logging.Handler:
    fh = logging.FileHandler(run_dir / "harness_run.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger("HARNESS").addHandler(fh)
    logging.getLogger("v2").addHandler(fh)
    return fh


class _PipelineShim:
    """Minimum-surface stand-in for TMDDataPipeline that the orchestrator
    needs at runtime. E2 builds its own feature matrices, so the only
    thing the orchestrator actually reads is the feature-description
    dict used in LLM prompts."""

    def __init__(self, feature_names: list[str]):
        self._names = list(feature_names)

    def get_feature_descriptions(self) -> dict[str, str]:
        # Minimal but informative — the LLM prompt only needs enough
        # to distinguish "atomic number" from "electronegativity" etc.
        descs = {
            "atomic_number_M": "Metal atomic number (Z_M, integer)",
            "atomic_number_X": "Anion atomic number (Z_X, integer)",
            "electronegativity_M": "Pauling electronegativity of metal (χ_M)",
            "electronegativity_X": "Pauling electronegativity of anion (χ_X)",
            "atomic_radius_M": "Metal atomic radius (Å)",
            "atomic_radius_X": "Anion atomic radius (Å)",
            "ionization_energy_M": "Metal first ionisation energy (eV)",
            "electron_affinity_X": "Anion electron affinity (eV)",
            "d_electron_count": "Metal d-electron count (d_e, integer)",
            "bond_length_MX": "M–X bond length (Å)",
            "lattice_constant_a": "In-plane lattice constant (Å)",
            "monolayer_thickness": "Monolayer thickness (Å)",
            "coordination_number": "Metal coordination number",
            "point_group_encoded": "Crystal point-group (integer code)",
            "electronegativity_diff": "|χ_M − χ_X|",
            "radius_ratio": "r_M / r_X",
            "bond_polarity": "Bond polarity index (χ_X − χ_M)/(χ_X + χ_M)",
            "packing_fraction": "Estimated atomic packing fraction",
            "Z_ratio": "Z_M / Z_X",
            "IE_EA_ratio": "IE_M / EA_X",
        }
        return {n: descs.get(n, n) for n in self._names}


def _parse_vote_tally(log_path: Path) -> dict[str, Any]:
    """Identical to E1's tally; duplicated intentionally so E2 runs are
    self-contained even if someone copies just the E2 package."""
    if not log_path.is_file():
        return {"log_missing": True}

    failures: list[dict[str, str]] = []
    activity_counts: Counter[str] = Counter()
    fail_re = re.compile(r"Agent (\w+) vote failed: (.+?)(?:\s*$)")
    activity_res = {
        "DataAgent":      re.compile(r"\bDataAgent analysis:"),
        "TheoryAgent":    re.compile(r"\bTheoryAgent:\s*\d+ operator"),
        "MetaAgent":      re.compile(r"\bMetaAgent: Deliberation on"),
        "ValidatorAgent": re.compile(r"\[AgentCouncil\] Post-search(?: deliberation)?: "),
    }
    for line in log_path.read_text().splitlines():
        m = fail_re.search(line)
        if m:
            failures.append({"agent": m.group(1), "error": m.group(2)})
        for ag, regex in activity_res.items():
            if regex.search(line):
                activity_counts[ag] += 1
    return {
        "per_agent_activity_counts": dict(activity_counts),
        "failures": failures,
        "total_failures": len(failures),
        "data_agent_active": activity_counts.get("DataAgent", 0) > 0,
    }


# -- main entry --------------------------------------------------------------


def run(config_path: Path, run_seed: int, run_dir: Path) -> dict[str, Any]:
    """
    Execute ONE outer-LOCO fold. Which anion is held out is taken from the
    config file (key: ``outer_anion``), so the caller drives a 10-fold
    sweep by pointing at 10 different configs.
    """
    file_handler = _install_file_log(run_dir)

    # Lazy imports (see E1 runner for rationale).
    from src.cauchy_validator import CauchyValidator
    from src.formula_evaluator import FormulaEvaluator
    from src.llm_sr_engine import LLMSREngine
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
    from src.v2.expr_eval import rmse_on

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    outer_anion = cfg["outer_anion"]           # e.g. "O"
    target = cfg.get("target", "cauchy_A")
    data_csv = cfg.get("data_csv", "data/extended/tier2_binary.csv")
    stable_lo, stable_hi = cfg.get("stability_range", [0.0, 20.0])

    sm = SeedManager(run_seed=run_seed)
    inner_split_seed = sm.derive("e2_inner_split", anion=outer_anion)

    log.info("E2 fold outer_anion=%s  run_seed=%s  inner_split_seed=%s",
             outer_anion, run_seed, inner_split_seed)

    # -- 1. load + partition Tier-2 --
    df = pd.read_csv(data_csv)
    if outer_anion not in set(df["anion"].unique()):
        raise ValueError(f"anion {outer_anion!r} not present in {data_csv}")

    outer_test_df = df[df["anion"] == outer_anion].reset_index(drop=True)
    outer_pool_df = df[df["anion"] != outer_anion].reset_index(drop=True)
    log.info("  outer_test N=%d   outer_pool N=%d",
             len(outer_test_df), len(outer_pool_df))

    X_pool = _build_feature_matrix(outer_pool_df)
    y_pool = outer_pool_df[target].to_numpy(dtype=float)
    X_outer = _build_feature_matrix(outer_test_df)
    y_outer = outer_test_df[target].to_numpy(dtype=float)

    # -- 2. inner 80/20 split on outer_pool --
    rng = np.random.default_rng(inner_split_seed)
    perm = rng.permutation(len(outer_pool_df))
    cut = int(0.8 * len(perm))
    tr_idx, sc_idx = perm[:cut], perm[cut:]
    X_train, y_train = X_pool[tr_idx], y_pool[tr_idx]
    X_screen, y_screen = X_pool[sc_idx], y_pool[sc_idx]
    log.info("  inner_train N=%d  inner_screen N=%d",
             len(X_train), len(X_screen))

    # -- 3. physical constraints + LLM-SR engine --
    validator = CauchyValidator(cfg["constraints"])
    evaluator = FormulaEvaluator(
        constraints_config=cfg["constraints"],
        X_test=X_screen, y_test=y_screen,
    )
    llm_sr = LLMSREngine(
        config=cfg["llm_sr"],
        feature_names=E2_FEATURE_NAMES,
        target_name=target,
        constraints=cfg["constraints"],
    )

    # -- 4. governance layer --
    material_metadata = outer_pool_df[["metal", "anion"]].rename(
        columns={"anion": "chalcogen"}
    ).assign(phase=0).reset_index(drop=True)

    km_path = run_dir / "knowledge_memory.json"
    knowledge = KnowledgeMemory(memory_path=km_path)
    diagnostic_engine = CausalDiagnosticEngine(
        validator, E2_FEATURE_NAMES, material_metadata
    )
    adaptive_env = AdaptiveSearchEnvironment(cfg["llm_sr"], knowledge)

    agent_cfg = cfg.get("agents", {})
    delib_cfg = cfg.get("deliberation", {})
    data_ag = DataAgent("DataAgent", "Feature analysis", knowledge, agent_cfg)
    theory_ag = TheoryAgent("TheoryAgent", "Physics reasoning", knowledge, agent_cfg)
    validator_ag = ValidatorAgent(
        "ValidatorAgent", "Physics validation", knowledge, agent_cfg,
        diagnostic_engine, X_train, y_train, X_screen, y_screen,
        E2_FEATURE_NAMES,
    )
    meta_ag = MetaAgent(
        "MetaAgent", "Deliberation", knowledge,
        {**agent_cfg, **delib_cfg},
        [data_ag, theory_ag, validator_ag],
    )
    council = AgentCouncil(data_ag, theory_ag, validator_ag, meta_ag, knowledge)

    # -- 5. orchestrator --
    log.info("  launching orchestrator (10 rounds)")
    orchestrator = HarnessOrchestrator(
        config=cfg,
        data_pipeline=_PipelineShim(E2_FEATURE_NAMES),
        llm_sr_engine=llm_sr,
        validator=validator,
        evaluator=evaluator,
        X_train=X_train, y_train=y_train,
        X_test=X_screen, y_test=y_screen,
        feature_names=E2_FEATURE_NAMES,
        target_name=target,
        agent_council=council,
        knowledge=knowledge,
        adaptive_env=adaptive_env,
        diagnostic_engine=diagnostic_engine,
    )
    results = orchestrator.run()
    orchestrator.export_results(run_dir)

    # -- 6. leakage-free evaluation on outer_test --
    # The orchestrator-reported best_formula uses x0..xK referring to the
    # features_used PySR received (which is usually the subset the council
    # approved, not the full 20). Project outer_test_df to that same
    # subset in the same column order.
    best_expr = results.get("best_formula")
    best_feats = results.get("features_used") or E2_FEATURE_NAMES
    if best_expr is None:
        log.error("  orchestrator returned no best_formula; aborting eval")
        outer_metrics = {"error": "no best_formula"}
    else:
        X_outer_best = _project_features(outer_test_df, best_feats)
        outer_metrics = rmse_on(
            best_expr, best_feats, X_outer_best, y_outer,
            stable_range=(stable_lo, stable_hi),
        )
    log.info("  outer_test metrics: %s", outer_metrics)

    # Evaluate every Pareto-front candidate on the held-out anion so the
    # paper can plot complexity vs held-out RMSE and pick a representative
    # at a defensible complexity level.
    pareto_evals = []
    for c in getattr(llm_sr, "pareto_front", []):
        cand_feats = list(c.features_used) if c.features_used else E2_FEATURE_NAMES
        X_outer_cand = _project_features(outer_test_df, cand_feats)
        ev = rmse_on(
            c.expression, cand_feats, X_outer_cand, y_outer,
            stable_range=(stable_lo, stable_hi),
        )
        pareto_evals.append({
            "expression": c.expression,
            "complexity": c.complexity,
            "features_used": cand_feats,
            "inner_train_rmse": float(c.train_rmse),
            "inner_screen_rmse": float(c.test_rmse),
            "outer_rmse": ev["rmse"],
            "outer_mae": ev["mae"],
            "outer_r2": ev["r2"],
            "outer_n_finite": ev["n_finite"],
            "outer_n_total": ev["n_total"],
        })

    # -- 7. persist fold summary --
    fold_summary = {
        "outer_anion": outer_anion,
        "run_seed": run_seed,
        "inner_split_seed": int(inner_split_seed),
        "n_outer_test": int(len(outer_test_df)),
        "n_outer_pool": int(len(outer_pool_df)),
        "n_inner_train": int(len(X_train)),
        "n_inner_screen": int(len(X_screen)),
        "target": target,
        "headline": {
            "expression": best_expr,
            "complexity": results.get("complexity"),
            "features_used": results.get("features_used", []),
            "inner_train_rmse": float(results.get("train_rmse", float("nan"))),
            "inner_screen_rmse": float(results.get("test_rmse", float("nan"))),
            "inner_screen_r2": float(results.get("r2", float("nan"))),
        },
        "outer_test_metrics": outer_metrics,
        "pareto_outer_evals": pareto_evals,
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2))

    vote_tally = _parse_vote_tally(run_dir / "harness_run.log")
    (run_dir / "agent_vote_tally.json").write_text(
        json.dumps(vote_tally, indent=2, default=str)
    )

    logging.getLogger("HARNESS").removeHandler(file_handler)
    logging.getLogger("v2").removeHandler(file_handler)
    file_handler.close()

    log.info("E2 fold anion=%s complete. outer RMSE=%.4f (N_finite=%d/%d)",
             outer_anion,
             outer_metrics.get("rmse", float("nan")),
             outer_metrics.get("n_finite", 0),
             outer_metrics.get("n_total", 0))
    return fold_summary


def _project_features(df: pd.DataFrame, names: list[str] | None) -> np.ndarray:
    """Build a feature matrix that matches the column order PySR saw when
    generating an expression. When `features_used` is a subset of the full
    20-descriptor dictionary, PySR uses x0..xK for just those K features in
    the given order, so the outer evaluation must present the columns in
    the same order."""
    if not names:
        return _build_feature_matrix(df)
    cols: list[np.ndarray] = []
    for long_name in names:
        short = _LONG_TO_SHORT.get(long_name, long_name)
        if short in df.columns:
            cols.append(df[short].to_numpy(dtype=float))
        else:
            # Harness-only extras (coordination_number, point_group_encoded,
            # packing_fraction) are zero-filled identically to training; any
            # expression that references them will just see zeros.
            cols.append(np.zeros(len(df), dtype=float))
    return np.column_stack(cols)
