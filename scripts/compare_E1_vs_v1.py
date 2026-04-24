"""
compare_E1_vs_v1.py — Phase 2.1.d diff tool.

Reads the v2 E1 run outputs and the v1 legacy outputs, aligns them on the
same 80/20 split and five-feature space, and prints a side-by-side table of:

  - headline expression (symbolic)
  - complexity
  - train / test RMSE
  - R²
  - agent-vote health
  - top-3 Pareto candidates

Also flags if the v2 expression is literally the same symbolic form as v1
(a reassuring negative) or structurally different (in which case the
paper's §6 "representative law" discussion needs a rewrite around the
new expression).

Usage:
    python3 scripts/compare_E1_vs_v1.py \
        --e1 experiments/E1_full_harness_fixed_council/runs/E1_seed42_phase2

If --e1 is omitted, the script picks the latest run directory under
experiments/E1_full_harness_fixed_council/runs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(p: Path) -> Any:
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _fmt(x: Any, width: int = 12, prec: int = 4) -> str:
    if x is None:
        return "-".rjust(width)
    if isinstance(x, float):
        return f"{x:>{width}.{prec}f}"
    return str(x).rjust(width)


def _pick_latest(parent: Path) -> Path | None:
    runs = [p for p in parent.glob("*") if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime)[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--e1", type=Path, default=None,
        help="E1 run dir. Defaults to latest in experiments/E1_.../runs/",
    )
    ap.add_argument(
        "--v1-discovery",
        type=Path,
        default=Path("legacy_v1/results_v1/discovery_report_20260420_165841.json"),
    )
    ap.add_argument(
        "--v1-pareto",
        type=Path,
        default=Path("legacy_v1/results_v1/tier2_stabveto_pareto_top10.csv"),
    )
    args = ap.parse_args()

    e1_dir = args.e1 or _pick_latest(
        Path("experiments/E1_full_harness_fixed_council/runs")
    )
    if e1_dir is None or not e1_dir.is_dir():
        print("ERROR: no E1 run directory found.")
        return 1
    print(f"Comparing E1 ({e1_dir.name}) vs v1 (165841)")
    print("=" * 72)

    # --- load e1 ---
    e1_summary = _load_json(e1_dir / "e1_summary.json") or {}
    e1_votes = _load_json(e1_dir / "agent_vote_tally.json") or {}
    e1_manifest = _load_json(e1_dir / "manifest.json") or {}

    # --- load v1 ---
    v1 = _load_json(args.v1_discovery) or {}

    # --- headline comparison ---
    print()
    print("HEADLINE EXPRESSION")
    print("-" * 72)
    print(f"  v1:  {v1.get('best_formula', '?')}")
    print(f"  v2:  {e1_summary.get('best_formula', '?')}")
    same = v1.get("best_formula") == e1_summary.get("best_formula")
    print(f"  same symbolic form: {same}")

    # --- metrics table ---
    v1m = v1.get("metrics", {})
    print()
    print("METRICS")
    print("-" * 72)
    print(f"  {'':<18}{'v1':>14}{'v2 (E1)':>14}{'Δ':>12}")
    for label, key in [
        ("train RMSE",      "train_rmse"),
        ("test RMSE",       "test_rmse"),
        ("R²",              "r2"),
        ("complexity",      "complexity"),
    ]:
        v1v = v1m.get(key)
        v2v = e1_summary.get(key)
        try:
            delta = v2v - v1v if isinstance(v1v, (int, float)) and isinstance(v2v, (int, float)) else None
        except TypeError:
            delta = None
        print(f"  {label:<18}"
              f"{_fmt(v1v):>14}"
              f"{_fmt(v2v):>14}"
              f"{_fmt(delta):>12}")

    # --- agent vote health ---
    print()
    print("AGENT VOTE HEALTH (the whole point of Phase 1.3 fix)")
    print("-" * 72)
    print(f"  vote-failure warnings (v2 E1): {e1_votes.get('total_failures', '?')}")
    print(f"  DataAgent active (v2 E1):      {e1_votes.get('data_agent_active', '?')}")
    per = e1_votes.get("per_agent", {})
    for agent in ("DataAgent", "TheoryAgent", "ValidatorAgent", "MetaAgent"):
        print(f"    {agent:<18} {per.get(agent, {})}")

    # --- features used ---
    print()
    print("FEATURES USED")
    print("-" * 72)
    print(f"  v1: {v1.get('features_used', '?')}")
    print(f"  v2: {e1_summary.get('features_used', '?')}")

    # --- manifest highlights ---
    print()
    print("v2 RUN MANIFEST HIGHLIGHTS")
    print("-" * 72)
    print(f"  run_seed:   {e1_manifest.get('run_seed')}")
    print(f"  git.short:  {e1_manifest.get('git', {}).get('short')}")
    print(f"  git.dirty:  {e1_manifest.get('git', {}).get('dirty')}")
    print(f"  pysr:       {e1_manifest.get('pkg_versions', {}).get('pysr')}")
    print(f"  config_sha: {e1_manifest.get('config_sha256', '')[:12]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
