"""
compare_E2_vs_v1_LOCO.py — compare true outer-LOCO (E2) against v1's
leaky LOCO table.

v1 Table 5 reports RMSE on each anion for the frozen Eq. 1. That Eq. 1
was discovered on the full Tier-2 including the anion it was later
"evaluated" against, so the comparison is:

  - v1 RMSE   : information-leaked evaluation
  - v2 E2 RMSE: true outer-LOCO (Harness never saw the anion)

If the v2 numbers are materially worse, the paper must retract the v1
"zero-shot transfer" framing — that is the key honest-science update
this whole v2 rewrite is built around.

Usage:
    python3 scripts/compare_E2_vs_v1_LOCO.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


V1_LOCO_CSV = Path("legacy_v1/results_v1/loco_new_headline.csv")
E2_RUNS = Path("experiments/E2_outer_loco/runs")


def load_v1() -> dict[str, dict]:
    rows = {}
    with V1_LOCO_CSV.open() as f:
        for r in csv.DictReader(f):
            rows[r["anion"]] = {
                "N": int(r["N"]),
                "harness_rmse": float(r["Harness"]),
                "rf_rmse": float(r["RF"]),
                "ridge_rmse": float(r["Ridge"]),
            }
    return rows


def load_e2() -> dict[str, dict]:
    out = {}
    for p in sorted(E2_RUNS.glob("fold_*_seed*/fold_summary.json")):
        s = json.loads(p.read_text())
        om = s["outer_test_metrics"]
        out[s["outer_anion"]] = {
            "N": s["n_outer_test"],
            "outer_rmse": om.get("rmse"),
            "outer_r2": om.get("r2"),
            "complexity": s["headline"]["complexity"],
            "expression": s["headline"]["expression"],
        }
    return out


def main() -> int:
    v1 = load_v1()
    e2 = load_e2()

    anion_order = ["O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P"]

    print()
    print("E2 outer-LOCO vs v1 Table 5 (leaky LOCO)")
    print("=" * 88)
    print(f"  {'anion':<6}{'N':>4}"
          f"{'v1 LOCO':>12}{'v2 outer':>12}{'Δ(v2-v1)':>12}"
          f"{'cplx':>8}  direction")
    print("-" * 88)

    v2_rmses, v1_rmses, ns = [], [], []
    for a in anion_order:
        v1r = v1.get(a)
        e2r = e2.get(a)
        if v1r is None or e2r is None:
            continue
        delta = e2r["outer_rmse"] - v1r["harness_rmse"]
        arrow = "↑ worse" if delta > 0 else "↓ better"
        print(f"  {a:<6}{v1r['N']:>4}"
              f"{v1r['harness_rmse']:>12.4f}"
              f"{e2r['outer_rmse']:>12.4f}"
              f"{delta:>+12.4f}"
              f"{e2r['complexity']:>8}  {arrow}")
        v1_rmses.append(v1r["harness_rmse"])
        v2_rmses.append(e2r["outer_rmse"])
        ns.append(v1r["N"])

    if v1_rmses:
        w = sum(ns)
        v1_wavg = sum(r * n for r, n in zip(v1_rmses, ns)) / w
        v2_wavg = sum(r * n for r, n in zip(v2_rmses, ns)) / w
        print("-" * 88)
        print(f"  weighted avg       {v1_wavg:>12.4f}{v2_wavg:>12.4f}"
              f"{v2_wavg - v1_wavg:>+12.4f}")

    print()
    print("Interpretation:")
    print("  If v2 RMSE is larger than v1 across most anions, v1 materially")
    print("  overstated its LOCO transfer performance because the expression")
    print("  had already seen every anion during discovery. The v2 numbers")
    print("  are the leakage-free truth that belongs in the paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
