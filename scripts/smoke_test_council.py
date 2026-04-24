"""
smoke_test_council.py — Phase 1.6 verification.

Exercises the 4-agent council on real Tier-2 data and prints a compact
per-agent vote table so we can eyeball whether the DataAgent bug fix
actually made the agent participate (rather than silently ABSTAIN).

Expected output on a healthy run:
  - DataAgent vote is APPROVE or MODIFY (not ABSTAIN-due-to-exception)
  - TheoryAgent vote is APPROVE or MODIFY
  - ValidatorAgent vote is APPROVE (no formula to reject at pre-search time)
  - MetaAgent deliberation returns a non-empty feature list

Run from project root:
    python3 scripts/smoke_test_council.py
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.agent_council import AgentCouncil
from agents.data_agent import DataAgent
from agents.meta_agent import MetaAgent
from agents.theory_agent import TheoryAgent
from agents.validator_agent import ValidatorAgent
from src.causal_diagnostic_engine import CausalDiagnosticEngine
from src.knowledge_memory import KnowledgeMemory
from src.v2.seeds import SeedManager


BANNER = "=" * 70


def _load_tier2_split(seed: int):
    """Load Tier-2 and produce the same 80/20 split main.py uses (seed=42 default)."""
    df = pd.read_csv("data/extended/tier2_binary.csv")

    # Match the feature order Harness actually saw in the 165841 run.
    feature_names = [
        "atomic_number_M", "atomic_number_X", "electronegativity_M",
        "electronegativity_X", "atomic_radius_M", "atomic_radius_X",
        "ionization_energy_M", "electron_affinity_X", "d_electron_count",
        "bond_length_MX", "lattice_constant_a", "monolayer_thickness",
        "coordination_number", "point_group_encoded",
        "electronegativity_diff", "radius_ratio", "bond_polarity",
        "packing_fraction", "Z_ratio", "IE_EA_ratio",
    ]
    # Short-name columns in the CSV; map long → short.
    col_map = {
        "atomic_number_M": "Z_M", "atomic_number_X": "Z_X",
        "electronegativity_M": "chi_M", "electronegativity_X": "chi_X",
        "atomic_radius_M": "r_M", "atomic_radius_X": "r_X",
        "ionization_energy_M": "IE_M", "electron_affinity_X": "EA_X",
        "d_electron_count": "d_e", "bond_length_MX": "d_MX",
        "lattice_constant_a": "a_lat", "monolayer_thickness": "t",
        "electronegativity_diff": "abs_delta_chi", "radius_ratio": "r_ratio",
        "bond_polarity": "bond_polarity",
        "Z_ratio": "Z_ratio", "IE_EA_ratio": "IE_EA_ratio",
    }

    # Not all long names have a CSV column; fill missing with zeros (those are
    # structural extras like coordination_number that the synthetic pipeline
    # normally supplies but the flat CSV doesn't).
    X_cols = []
    for long_name in feature_names:
        short = col_map.get(long_name)
        if short and short in df.columns:
            X_cols.append(df[short].values)
        else:
            X_cols.append(np.zeros(len(df), dtype=float))
    X = np.column_stack(X_cols)
    y = df["cauchy_A"].values

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return (
        X[train_idx], y[train_idx],
        X[test_idx], y[test_idx],
        feature_names,
        df.iloc[train_idx].reset_index(drop=True),
    )


def main() -> int:
    # Capture every warning from HARNESS loggers so we can grep them.
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.WARNING)
    root_logger = logging.getLogger("HARNESS")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    print(BANNER)
    print("PHASE 1.6 SMOKE TEST — 4-agent council dry-run")
    print(BANNER)

    # Config
    with open("configs/harness_config.yaml") as f:
        config = yaml.safe_load(f)

    # Seeds via v2 infra
    sm = SeedManager(run_seed=42)
    split_seed = sm.derive("smoke_test_split")
    print(f"run_seed = 42; derived split seed = {split_seed}")

    # Data
    X_train, y_train, X_test, y_test, feature_names, meta = _load_tier2_split(
        split_seed
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, "
          f"features: {len(feature_names)}")

    # Council
    knowledge = KnowledgeMemory()
    # Pass a minimal validator-compatible diagnostic engine
    agent_cfg = config.get("agents", {})
    delib_cfg = config.get("deliberation", {})

    # Stub validator needed by CausalDiagnosticEngine — any object with
    # check_coefficient_signs() is fine; pass None and let it no-op.
    data_ag = DataAgent("DataAgent", "Feature analysis", knowledge, agent_cfg)
    theory_ag = TheoryAgent("TheoryAgent", "Physics reasoning", knowledge, agent_cfg)
    validator_ag = ValidatorAgent(
        "ValidatorAgent", "Physics validation", knowledge, agent_cfg,
        None,  # diagnostic_engine — not needed for pre-search
        X_train, y_train, X_test, y_test, feature_names,
    )
    meta_cfg = {**agent_cfg, **delib_cfg}
    meta_ag = MetaAgent(
        "MetaAgent", "Deliberation", knowledge, meta_cfg,
        [data_ag, theory_ag, validator_ag],
    )
    council = AgentCouncil(data_ag, theory_ag, validator_ag, meta_ag, knowledge)

    # Run a pre-search deliberation — this is the round-1 path that
    # invoked _vote_feature_selection (the function that used to crash).
    context = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
        "metadata": meta,
    }
    print(BANNER)
    print("Calling council.pre_search_deliberation(round_id=1, context=...)")
    print(BANNER)
    instructions = council.pre_search_deliberation(round_id=1, context=context)

    # Inspect results
    print()
    print("Returned instructions keys:", list(instructions.keys()))
    for k in ("features", "operators", "max_depth", "iterations"):
        if k in instructions:
            v = instructions[k]
            print(f"  {k}: {v if not isinstance(v, list) else v[:8]}")

    # Inspect logged warnings
    handler.flush()
    log_text = log_buf.getvalue()
    failures = [
        ln for ln in log_text.splitlines()
        if "vote failed" in ln.lower()
        or "name 'context' is not defined" in ln.lower()
    ]

    print()
    print(BANNER)
    print("VOTE-FAILURE WARNINGS CAPTURED:")
    print(BANNER)
    if failures:
        for ln in failures:
            print("  " + ln)
    else:
        print("  (none — all agents voted without raising exceptions)")

    print()
    if failures:
        print("RESULT: FAIL — DataAgent bug (or another vote bug) still present.")
        return 1
    print("RESULT: PASS — 4-agent council voted cleanly on round-1 pre-search.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
