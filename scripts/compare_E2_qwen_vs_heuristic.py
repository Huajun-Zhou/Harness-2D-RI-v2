"""
compare_E2_qwen_vs_heuristic.py — did enabling the LLM change anything?

Three columns per fold:
  - Heuristic-Harness (original E2, LLM API was unreachable)
  - Qwen-Harness      (E2 rerun with LLM_PROVIDER=qwen)
  - V2 vanilla PySR   (E3 fair-selected mean)

If Qwen-Harness consistently beats both, the v2 paper's methodology
claim has legs. If it only matches heuristic-Harness, the LLM is not
adding value and Path C (pivot to 'audit' framing) is the right call.

Usage:
    python3 scripts/compare_E2_qwen_vs_heuristic.py
"""

from __future__ import annotations

import json
from pathlib import Path


ANIONS = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P")


def _load_fold_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> int:
    e2_heur = Path("experiments/E2_outer_loco/runs")
    e2_qwen = Path("experiments/E2_outer_loco/runs")
    e3 = Path("experiments/E3_matched_pysr_baselines/runs")

    rows = []
    for a in ANIONS:
        h = _load_fold_summary(e2_heur / f"fold_{a}_seed42" / "fold_summary.json")
        q = _load_fold_summary(e2_qwen / f"fold_{a}_seed42_qwen" / "fold_summary.json")
        v = _load_fold_summary(e3 / f"fold_{a}_seed42" / "fold_summary.json")
        if h is None or q is None or v is None:
            continue
        rows.append({
            "anion": a,
            "N": h["n_outer_test"],
            "heur_rmse": h["outer_test_metrics"].get("rmse"),
            "heur_cplx": h["headline"]["complexity"],
            "qwen_rmse": q["outer_test_metrics"].get("rmse"),
            "qwen_cplx": q["headline"]["complexity"],
            "qwen_expr": q["headline"]["expression"],
            "qwen_feats": q["headline"]["features_used"],
            "v2_rmse":   v["policy_summary"]["V2_screen"]["mean_outer_rmse"],
            "v2_cplx":   v["policy_summary"]["V2_screen"]["mean_complexity"],
        })

    if not rows:
        print("no folds with all three runs yet")
        return 1

    print()
    print("Qwen-Harness vs Heuristic-Harness vs V2 vanilla PySR — outer RMSE")
    print("=" * 104)
    print(f"  {'anion':<6}{'N':>4}"
          f"{'heuristic':>11}{'Qwen':>11}"
          f"{'ΔvsH':>9}"
          f"{'V2 vanilla':>13}"
          f"{'ΔvsV2':>9}"
          f"{'cplx heur/qwen/V2':>22}")
    print("-" * 104)

    wins_qwen_vs_heur = 0
    wins_qwen_vs_v2 = 0
    for r in rows:
        d_heur = r["qwen_rmse"] - r["heur_rmse"]
        d_v2 = r["qwen_rmse"] - r["v2_rmse"]
        if d_heur < 0: wins_qwen_vs_heur += 1
        if d_v2 < 0:   wins_qwen_vs_v2 += 1
        cplx_str = f"{r['heur_cplx']}/{r['qwen_cplx']}/{r['v2_cplx']:.1f}"
        print(f"  {r['anion']:<6}{r['N']:>4}"
              f"{r['heur_rmse']:>11.4f}{r['qwen_rmse']:>11.4f}"
              f"{d_heur:>+9.4f}"
              f"{r['v2_rmse']:>13.4f}"
              f"{d_v2:>+9.4f}"
              f"{cplx_str:>22}")
    print("-" * 104)

    weights = [r["N"] for r in rows]
    W = sum(weights)
    def _w(key: str) -> float:
        return sum(r[key] * r["N"] for r in rows) / W
    hw = _w("heur_rmse"); qw = _w("qwen_rmse"); vw = _w("v2_rmse")
    print(f"  {'weighted':<10}"
          f"{hw:>11.4f}{qw:>11.4f}"
          f"{qw - hw:>+9.4f}"
          f"{vw:>13.4f}"
          f"{qw - vw:>+9.4f}")
    print()
    print(f"Qwen-Harness beats heuristic-Harness on {wins_qwen_vs_heur}/{len(rows)} folds")
    print(f"Qwen-Harness beats V2 vanilla PySR   on {wins_qwen_vs_v2}/{len(rows)} folds")
    print()
    print("Interpretation guide:")
    print("  - If Qwen wins vs V2 weighted, the methodology claim is supported.")
    print("  - If Qwen ties or loses, the LLM is not the bottleneck; governance is.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
