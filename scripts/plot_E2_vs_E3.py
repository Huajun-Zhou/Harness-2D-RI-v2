"""
plot_E2_vs_E3.py — produce the head-to-head figure for the v2 paper.

Reads:
  experiments/E2_outer_loco/runs/fold_*_seed42/fold_summary.json
  experiments/E3_matched_pysr_baselines/runs/fold_*_seed42/fold_summary.json

Produces a single figure:
  experiments/E3_matched_pysr_baselines/figures/harness_vs_matched_outer_rmse.pdf

Per-fold grouped bar chart: outer RMSE for
  Harness (E2, single value)
  V1 vanilla-PySR train-selected (mean ± std over 5 seeds)
  V2 vanilla-PySR screening-selected (matched fair comparison)
  V3 vanilla-PySR complexity-capped ≤11
  V4 vanilla-PySR stability-only

Annotate: smaller is better. Rare-anion folds (F, P) visually marked.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ANIONS = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P")
POLICIES = ("V1_train", "V2_screen", "V3_cplx_capped", "V4_stability_only")
POLICY_LABELS = {
    "V1_train":          "Vanilla PySR, train-RMSE selected (v1 protocol)",
    "V2_screen":         "Vanilla PySR, screening-selected (fair)",
    "V3_cplx_capped":    "Vanilla PySR, complexity ≤ 11",
    "V4_stability_only": "Vanilla PySR, stability-only filter",
}
POLICY_COLORS = {
    "V1_train":          "#c94c4c",
    "V2_screen":         "#e89f4c",
    "V3_cplx_capped":    "#4ca9e8",
    "V4_stability_only": "#6fb36a",
}


def load_e2() -> dict:
    out = {}
    for a in ANIONS:
        p = Path(f"experiments/E2_outer_loco/runs/fold_{a}_seed42/fold_summary.json")
        if p.is_file():
            s = json.loads(p.read_text())
            out[a] = {
                "N": s["n_outer_test"],
                "harness_rmse": s["outer_test_metrics"].get("rmse"),
                "harness_cplx": s["headline"]["complexity"],
            }
    return out


def load_e3() -> dict:
    out = {}
    for a in ANIONS:
        p = Path(f"experiments/E3_matched_pysr_baselines/runs/fold_{a}_seed42/fold_summary.json")
        if p.is_file():
            out[a] = json.loads(p.read_text())
    return out


def main() -> int:
    e2 = load_e2()
    e3 = load_e3()
    folds = [a for a in ANIONS if a in e2 and a in e3]
    if not folds:
        print("no overlapping folds found; run E2 and E3 first")
        return 1

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(folds))
    n_groups = 1 + len(POLICIES)
    width = 0.8 / n_groups

    # Harness bar
    h_vals = [e2[a]["harness_rmse"] for a in folds]
    ax.bar(x - 0.4 + 0.5 * width, h_vals, width, label="Harness (E2)",
           color="#222222", edgecolor="black", zorder=3)

    # Vanilla-PySR bars (one per policy)
    for j, policy in enumerate(POLICIES):
        means = [e3[a]["policy_summary"][policy]["mean_outer_rmse"] for a in folds]
        stds = [e3[a]["policy_summary"][policy]["std_outer_rmse"] for a in folds]
        pos = x - 0.4 + (j + 1.5) * width
        ax.bar(pos, means, width, yerr=stds, capsize=2,
               label=POLICY_LABELS[policy],
               color=POLICY_COLORS[policy], edgecolor="black", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(N={e2[a]['N']})" for a in folds])
    ax.set_xlabel("held-out anion (outer-LOCO fold)")
    ax.set_ylabel("outer-test RMSE (lower is better)")
    ax.set_title("Outer-LOCO performance: Harness vs matched vanilla PySR baselines")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Mark rare-anion folds
    for a, xi in zip(folds, x):
        if e2[a]["N"] < 10:
            ax.axvspan(xi - 0.45, xi + 0.45, color="#ffeecc", alpha=0.3, zorder=1)

    out_dir = Path("experiments/E3_matched_pysr_baselines/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "harness_vs_matched_outer_rmse.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
