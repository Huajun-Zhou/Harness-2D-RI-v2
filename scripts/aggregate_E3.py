"""
aggregate_E3.py — Phase 2.3.e E3 roll-up + head-to-head with E2.

For every E3 fold, pulls policy_summary (4 policies × mean ± std outer
RMSE and complexity across seeds). Produces:

  - experiments/E3_matched_pysr_baselines/e3_fold_table.csv
        rows: (fold, policy) — mean/std/complexity/per-seed outer RMSE
  - experiments/E3_matched_pysr_baselines/e3_weighted_summary.json
        weighted outer RMSE per policy across folds
  - experiments/E3_matched_pysr_baselines/harness_vs_matched.csv
        one row per fold: E2 Harness outer vs each E3 policy outer
  - stdout: side-by-side fold table

Usage:
    python3 scripts/aggregate_E3.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


E2_RUNS = Path("experiments/E2_outer_loco/runs")
E3_RUNS = Path("experiments/E3_matched_pysr_baselines/runs")
ANIONS = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P")

POLICIES = ("V1_train", "V2_screen", "V3_cplx_capped", "V4_stability_only")


def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _weighted(vals: list[float], weights: list[int]) -> float:
    pairs = [(v, w) for v, w in zip(vals, weights)
             if v is not None and not math.isnan(v) and w > 0]
    if not pairs:
        return float("nan")
    total_w = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total_w


def load_e2() -> dict[str, dict]:
    out = {}
    for anion in ANIONS:
        p = E2_RUNS / f"fold_{anion}_seed42" / "fold_summary.json"
        if not p.is_file():
            continue
        s = json.loads(p.read_text())
        om = s["outer_test_metrics"]
        out[anion] = {
            "N": s["n_outer_test"],
            "harness_outer_rmse": om.get("rmse"),
            "harness_outer_r2": om.get("r2"),
            "harness_complexity": s["headline"]["complexity"],
        }
    return out


def load_e3() -> dict[str, dict]:
    out = {}
    for anion in ANIONS:
        p = E3_RUNS / f"fold_{anion}_seed42" / "fold_summary.json"
        if not p.is_file():
            continue
        out[anion] = json.loads(p.read_text())
    return out


def main() -> int:
    e2 = load_e2()
    e3 = load_e3()

    # -- per-fold table (E3 only) --
    out_e3_csv = Path("experiments/E3_matched_pysr_baselines/e3_fold_table.csv")
    out_e3_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_e3_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "outer_anion", "policy", "n_seeds_success",
            "mean_outer_rmse", "std_outer_rmse",
            "mean_complexity", "std_complexity",
        ])
        for anion in ANIONS:
            if anion not in e3:
                continue
            ps = e3[anion]["policy_summary"]
            for policy in POLICIES:
                r = ps[policy]
                w.writerow([
                    anion, policy, r["n_seeds_success"],
                    f"{r['mean_outer_rmse']:.4f}",
                    f"{r['std_outer_rmse']:.4f}",
                    f"{r['mean_complexity']:.1f}",
                    f"{r['std_complexity']:.1f}",
                ])

    # -- weighted summary per policy --
    weighted = {"n_folds": len(e3), "per_policy": {}}
    for policy in POLICIES:
        means = [e3[a]["policy_summary"][policy]["mean_outer_rmse"]
                 for a in ANIONS if a in e3]
        weights = [e3[a]["n_outer_test"] for a in ANIONS if a in e3]
        weighted["per_policy"][policy] = {
            "weighted_outer_rmse": _weighted(means, weights),
            "mean_outer_rmse": _safe_mean(means),
        }
    # Harness weighted (from E2) for the same fold set
    h_means = [e2[a]["harness_outer_rmse"] for a in ANIONS if a in e3 and a in e2]
    h_weights = [e2[a]["N"] for a in ANIONS if a in e3 and a in e2]
    weighted["harness_weighted_outer_rmse"] = _weighted(h_means, h_weights)
    Path("experiments/E3_matched_pysr_baselines/e3_weighted_summary.json").write_text(
        json.dumps(weighted, indent=2)
    )

    # -- head-to-head table --
    out_h2h = Path("experiments/E3_matched_pysr_baselines/harness_vs_matched.csv")
    with out_h2h.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "outer_anion", "N",
            "harness_outer_rmse", "harness_complexity",
            "V1_train_mean", "V1_train_std", "V1_train_cplx",
            "V2_screen_mean", "V2_screen_std", "V2_screen_cplx",
            "V3_cplx_capped_mean", "V3_cplx_capped_std", "V3_cplx_capped_cplx",
            "V4_stability_only_mean", "V4_stability_only_std", "V4_stability_only_cplx",
        ])
        for anion in ANIONS:
            if anion not in e3 or anion not in e2:
                continue
            row = [anion, e2[anion]["N"],
                   f"{e2[anion]['harness_outer_rmse']:.4f}",
                   e2[anion]["harness_complexity"]]
            for policy in POLICIES:
                r = e3[anion]["policy_summary"][policy]
                row.extend([
                    f"{r['mean_outer_rmse']:.4f}",
                    f"{r['std_outer_rmse']:.4f}",
                    f"{r['mean_complexity']:.1f}",
                ])
            w.writerow(row)

    # -- stdout --
    print()
    print("E3 matched-PySR baselines — outer RMSE (mean ± std across 5 seeds)")
    print("=" * 110)
    print(f"  {'anion':<6}{'N':>4}"
          f"{'Harness':>12}"
          f"{'V1_train':>18}"
          f"{'V2_screen':>18}"
          f"{'V3_cplx_cap':>18}"
          f"{'V4_stab_only':>18}")
    print("-" * 110)
    for anion in ANIONS:
        if anion not in e3 or anion not in e2:
            continue
        h = e2[anion]
        ps = e3[anion]["policy_summary"]
        line = f"  {anion:<6}{h['N']:>4}{h['harness_outer_rmse']:>12.4f}"
        for policy in POLICIES:
            r = ps[policy]
            line += f"  {r['mean_outer_rmse']:>7.4f} ± {r['std_outer_rmse']:.4f}"
        print(line)
    print("-" * 110)

    # Weighted averages footer
    wp = weighted["per_policy"]
    hw = weighted["harness_weighted_outer_rmse"]
    print(f"  {'weighted':<10}"
          f"{hw:>12.4f}"
          f"  {wp['V1_train']['weighted_outer_rmse']:>7.4f}          "
          f"  {wp['V2_screen']['weighted_outer_rmse']:>7.4f}          "
          f"  {wp['V3_cplx_capped']['weighted_outer_rmse']:>7.4f}          "
          f"  {wp['V4_stability_only']['weighted_outer_rmse']:>7.4f}")
    print()
    print("CSV files:")
    print(f"  {out_e3_csv}")
    print(f"  {out_h2h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
