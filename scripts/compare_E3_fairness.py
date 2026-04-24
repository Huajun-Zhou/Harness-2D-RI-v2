"""
compare_E3_fairness.py — answers the reviewer's core methodology question.

"When Harness and Vanilla-PySR are evaluated under the SAME leakage-free
outer-LOCO protocol, does Harness still win?"

For each fold and each of four matched-PySR policies, this prints:

  delta = mean(vanilla_outer_rmse) - harness_outer_rmse

Negative ⇒ vanilla PySR beats Harness (bad for the paper).
Positive ⇒ Harness beats vanilla (supports the methodology claim).

Also prints per-fold "complexity tax" — did Harness produce a materially
smaller expression than the vanilla PySR policy at similar outer RMSE?

Usage:
    python3 scripts/compare_E3_fairness.py
"""

from __future__ import annotations

import json
from pathlib import Path


ANIONS = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P")
POLICIES = ("V1_train", "V2_screen", "V3_cplx_capped", "V4_stability_only")


def _load_e2() -> dict:
    out = {}
    for a in ANIONS:
        p = Path(f"experiments/E2_outer_loco/runs/fold_{a}_seed42/fold_summary.json")
        if p.is_file():
            s = json.loads(p.read_text())
            out[a] = {
                "harness_rmse": s["outer_test_metrics"].get("rmse"),
                "harness_cplx": s["headline"]["complexity"],
                "N": s["n_outer_test"],
            }
    return out


def _load_e3() -> dict:
    out = {}
    for a in ANIONS:
        p = Path(f"experiments/E3_matched_pysr_baselines/runs/fold_{a}_seed42/fold_summary.json")
        if p.is_file():
            out[a] = json.loads(p.read_text())
    return out


def main() -> int:
    e2 = _load_e2()
    e3 = _load_e3()
    folds = [a for a in ANIONS if a in e2 and a in e3]
    if not folds:
        print("Neither E2 nor E3 has folds yet.")
        return 1

    print()
    print("Harness-vs-matched-PySR — outer RMSE delta (vanilla − Harness)")
    print("Positive ⇒ Harness wins ⇒ supports the methodology claim.")
    print("=" * 96)
    print(f"  {'anion':<6}{'N':>4}{'H_rmse':>9}{'H_cplx':>8}"
          f"{'V1_Δ':>9}{'V2_Δ':>9}{'V3_Δ':>9}{'V4_Δ':>9}"
          f"{'Harness wins?':>18}")
    print("-" * 96)

    wins_any = 0
    wins_v2 = 0  # V2 is the fairest comparison
    for a in folds:
        H = e2[a]
        ps = e3[a]["policy_summary"]
        deltas = {p: ps[p]["mean_outer_rmse"] - H["harness_rmse"] for p in POLICIES}
        # Flag: did Harness beat V2 (the fairest vanilla variant)?
        flag = "Y" if deltas["V2_screen"] > 0 else "N"
        if any(d > 0 for d in deltas.values()):
            wins_any += 1
        if deltas["V2_screen"] > 0:
            wins_v2 += 1
        print(f"  {a:<6}{H['N']:>4}"
              f"{H['harness_rmse']:>9.4f}{H['harness_cplx']:>8}"
              f"{deltas['V1_train']:>+9.4f}"
              f"{deltas['V2_screen']:>+9.4f}"
              f"{deltas['V3_cplx_capped']:>+9.4f}"
              f"{deltas['V4_stability_only']:>+9.4f}"
              f"{flag:>18}")
    print("-" * 96)

    # Weighted-RMSE comparison (paper's aggregate number)
    weights = [e2[a]["N"] for a in folds]
    W = sum(weights)
    h_wavg = sum(e2[a]["harness_rmse"] * e2[a]["N"]
                 for a in folds) / W
    print(f"\nWeighted outer RMSE by sample count (N={W}):")
    print(f"  Harness (E2)                  : {h_wavg:.4f}")
    for p in POLICIES:
        v_wavg = sum(e3[a]["policy_summary"][p]["mean_outer_rmse"] * e2[a]["N"]
                     for a in folds) / W
        delta = v_wavg - h_wavg
        print(f"  {p:<30}: {v_wavg:.4f}   Δ={delta:+.4f}  "
              f"{'Harness wins' if delta > 0 else 'Harness loses'}")

    print()
    print(f"Harness beat at least one vanilla policy on {wins_any}/{len(folds)} folds.")
    print(f"Harness beat the fair V2_screen policy on  {wins_v2}/{len(folds)} folds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
