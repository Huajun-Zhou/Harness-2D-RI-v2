"""
E3_progress.py — quick status view of E3 run, callable at any time.

Shows per-fold completion and the O-fold-style "Harness vs vanilla" delta
for every fold that has already finished. Safe to run while the driver
is still mid-campaign.
"""

from __future__ import annotations

import json
from pathlib import Path


ANIONS = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P")
POLICIES = ("V1_train", "V2_screen", "V3_cplx_capped", "V4_stability_only")


def main() -> int:
    e3_root = Path("experiments/E3_matched_pysr_baselines/runs")
    e2_root = Path("experiments/E2_outer_loco/runs")

    done = [a for a in ANIONS
            if (e3_root / f"fold_{a}_seed42" / "fold_summary.json").is_file()]
    pending = [a for a in ANIONS if a not in done]

    print(f"E3 progress: {len(done)}/{len(ANIONS)} folds complete.")
    print(f"  done:    {done}")
    print(f"  pending: {pending}")
    print()

    if not done:
        return 0

    print(f"  {'anion':<6}{'N':>4}{'H':>8}{'V1':>10}{'V2':>10}{'V3':>10}{'V4':>10}"
          f"{'V2 vs H':>12}")
    print("-" * 80)
    for a in done:
        e2 = json.loads((e2_root / f"fold_{a}_seed42" / "fold_summary.json").read_text())
        e3 = json.loads((e3_root / f"fold_{a}_seed42" / "fold_summary.json").read_text())
        h_rmse = e2["outer_test_metrics"]["rmse"]
        N = e2["n_outer_test"]
        row = f"  {a:<6}{N:>4}{h_rmse:>8.4f}"
        for p in POLICIES:
            m = e3["policy_summary"][p]["mean_outer_rmse"]
            row += f"{m:>10.4f}"
        delta = e3["policy_summary"]["V2_screen"]["mean_outer_rmse"] - h_rmse
        flag = "Harness wins" if delta > 0 else "V2 wins"
        row += f"   {delta:>+.4f}  {flag}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
