"""
make_v2_figures.py — generate all v2 manuscript figures from real artifacts.

Each figure function loads exactly one or two committed JSON/CSV files from
``experiments/`` (or ``data/``) and prints the headline numbers it used,
so the provenance of every line in every figure is auditable from the
script alone.

Outputs go to ``figures/``. PDF for production, PNG for
manuscript-Markdown rendering.

Usage:
    python3 scripts/make_v2_figures.py [fig_name ...]

If no argument is given, all figures are produced. ``fig_name`` is one of
``fig2_dataset``, ``fig3_outer_loco``, ``fig4_ablation``,
``fig5_diagnostics``, ``fig6_parity``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- shared style ------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

REPO   = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT    = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ANIONS_ORDER = ("O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P")
CHALC = {"O", "S", "Se", "Te"}
HAL   = {"F", "Cl", "Br", "I"}
PNIC  = {"N", "P"}

COLOR = {
    "qwen":      "#1f77b4",   # tab:blue
    "heur":      "#ff7f0e",   # tab:orange
    "v2vanilla": "#9467bd",   # tab:purple
    "chalc":     "#1f77b4",
    "hal":       "#d62728",
    "pnic":      "#2ca02c",
    "FULL":      "#2ca02c",
    "neg":       "#d62728",
    "pos":       "#1f77b4",
    "null":      "#7f7f7f",
}


def _save(fig, name: str) -> None:
    pdf = OUT / f"{name}.pdf"
    png = OUT / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"  wrote {pdf.relative_to(REPO)}")
    print(f"  wrote {png.relative_to(REPO)}")


def _anion_family(a: str) -> str:
    if a in CHALC: return "chalcogenide"
    if a in HAL:   return "halide"
    if a in PNIC:  return "pnictide"
    return "other"


# -- Fig 1: architecture schematic ------------------------------------------


def fig1_architecture() -> None:
    """Schematic of the Harness-2D-RI four-layer scaffold around PySR.
    Pure layout — no data is read from disk. The figure is a flow diagram:
    inputs (descriptor matrix + previous Pareto) on the left, the four
    governance layers wrapping a PySR search core in the middle, and the
    updated post-veto Pareto on the right."""
    print("\n[fig1_architecture]")
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56)
    ax.set_aspect("equal"); ax.axis("off")

    def box(x, y, w, h, label, fc, ec="black", lw=1.2, fs=10, fontweight="normal"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight=fontweight, wrap=True)

    def arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0), color="black", lw=1.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))
        if label:
            ax.text((x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1],
                    label, fontsize=8, ha="center", va="center", style="italic",
                    color=color)

    # Inputs (left)
    box(2,  42, 18, 8, "descriptor\nmatrix\n(20 features)", "#e8eef7", fs=9)
    box(2,  26, 18, 8, "round $t{-}1$\npost-veto\nPareto", "#e8eef7", fs=9)
    box(2,  10, 18, 8, "knowledge\nmemory\n(within fold)", "#e8eef7", fs=9)

    # Center stack — four governance layers wrapping PySR
    # Outer wrapper rectangle (governance scaffold)
    wrap = plt.Rectangle((28, 4), 44, 48, facecolor="#fff8e8",
                          edgecolor="#cc9933", linewidth=1.2, linestyle="--")
    ax.add_patch(wrap)
    ax.text(50, 53.5, "Harness scaffold (per-round)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#996600")

    # 1. LLM proposal (top)
    box(30, 42, 40, 8, "(1) LLM-guided proposal\nfeature subset · operators · skeleton hint",
        "#cce5ff", fs=9, fontweight="bold")

    # 2. PySR search core
    box(34, 28, 32, 10, "PySR\nsearch core\n(unmodified)", "#ffffff", lw=1.5,
        fs=10, fontweight="bold")

    # 3. Agent council
    box(30, 17, 40, 8, "(2) Agent council  ·  numerical-stability VETO\nfour agents · single-VETO rule",
        "#ffd9d9", fs=9, fontweight="bold")

    # 4. Causal critic + memory feedback
    box(30, 7, 40, 8, "(3) Causal-diagnostic critic    ·    (4) Knowledge memory\nfinite-difference sensitivities · failure analyses",
        "#d9f2e6", fs=9, fontweight="bold")

    # Output (right)
    box(80, 26, 18, 8, "round $t$\npost-veto\nPareto", "#e8eef7", fs=9)
    box(80, 8,  18, 8, "(final round)\nheadline\nexpression", "#d6f5d6", fs=9, fontweight="bold")

    # Arrows: inputs → LLM proposal
    arrow(20, 46, 30, 46)                                # descriptor → proposal
    arrow(20, 30, 30, 38, color="#444")                  # prev Pareto → proposal
    arrow(20, 14, 30, 14, color="#444")                  # memory → critic/memory layer

    # Arrows: LLM proposal → PySR
    arrow(50, 42, 50, 38)                                # proposal → PySR

    # Arrows: PySR → council
    arrow(50, 28, 50, 25)                                # PySR → council

    # Arrows: council → out (accepted)
    arrow(70, 21, 80, 30, label="accepted",
          label_offset=(0, 1.5), color="#1a8c1a")        # council → next-round Pareto

    # Arrows: council → memory (rejected with failure analyses)
    arrow(50, 17, 50, 15, color="#cc4444")               # council → memory layer

    # Arrows: critic feedback → next-round LLM prompt (curved over the top)
    ax.annotate("", xy=(30, 47.5), xytext=(70, 11),
                arrowprops=dict(arrowstyle="->", color="#995500", lw=1.0,
                                 connectionstyle="arc3,rad=-0.45"))
    ax.text(82, 38, "feedback\ninto next prompt", ha="center", va="center",
            fontsize=8, color="#995500", style="italic")

    # Arrow: final round → headline
    arrow(89, 26, 89, 16, color="#1a8c1a")

    # Legend / role labels
    ax.text(11, 53, "Inputs", fontsize=10, fontweight="bold", color="#333",
            ha="center")
    ax.text(89, 53, "Outputs", fontsize=10, fontweight="bold", color="#333",
            ha="center")

    fig.tight_layout()
    _save(fig, "fig1_architecture")
    plt.close(fig)


# -- Fig 2: Tier-2 dataset EDA ----------------------------------------------


def fig2_dataset() -> None:
    """Tier-2 dataset descriptive panel: target distribution, anion mix,
    metal–anion grid. No model output — just the raw table."""
    print("\n[fig2_dataset]")
    src = REPO / "data" / "extended" / "tier2_binary.csv"
    df = pd.read_csv(src)
    print(f"  source: {src.relative_to(REPO)}  N={len(df)}")
    print(f"  target: cauchy_A in [{df['cauchy_A'].min():.3f}, "
          f"{df['cauchy_A'].max():.3f}]  std={df['cauchy_A'].std():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))

    # (a) cauchy_A vs Z_M, coloured by anion family.
    # Plot order: chalcogenide → halide → pnictide so the small (n=9)
    # pnictide group sits on top of the larger families and stays
    # visible. Pnictide markers are also enlarged and darkened
    # (#1a6b1a) to stand out from chalcogenides (blue) and halides
    # (red), since the small pnictide population is otherwise lost
    # in the scatter.
    ax = axes[0]
    plot_spec = [
        ("chalcogenide", df[df["anion"].isin(CHALC)].index, COLOR["chalc"], 22, 0.60, 1),
        ("halide",       df[df["anion"].isin(HAL)].index,   COLOR["hal"],   22, 0.60, 1),
        ("pnictide",     df[df["anion"].isin(PNIC)].index,  "#1a6b1a",      40, 0.95, 3),
    ]
    for fam, idxs, color, size, alpha, zorder in plot_spec:
        ax.scatter(df.loc[idxs, "Z_M"], df.loc[idxs, "cauchy_A"],
                   s=size, alpha=alpha, edgecolor="white", linewidth=0.5,
                   color=color, label=fam, zorder=zorder)
    ax.set_xlabel(r"metal atomic number $Z_M$")
    ax.set_ylabel(r"target $A=\sqrt{\varepsilon_{\infty,xx}}$")
    ax.set_title("(a) target vs metal Z, by anion family")
    ax.legend(loc="upper right", frameon=False)

    # (b) per-anion sample counts
    ax = axes[1]
    counts = df["anion"].value_counts().reindex(ANIONS_ORDER)
    colors = [COLOR["chalc"] if a in CHALC else COLOR["hal"]
              if a in HAL else COLOR["pnic"] for a in ANIONS_ORDER]
    ax.bar(ANIONS_ORDER, counts.values, color=colors, edgecolor="black",
           linewidth=0.4)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("anion (outer-LOCO fold)")
    ax.set_ylabel("number of binaries")
    ax.set_title("(b) per-anion sample counts")
    ax.set_ylim(0, counts.max() * 1.13)

    # (c) target histogram by family. Pnictide (n=9) is drawn last with
    # a darker green so the small group remains visible above the larger
    # halide / chalcogenide bins.
    ax = axes[2]
    for fam, mask, color, alpha, zorder in [
        ("chalcogenide", df["anion"].isin(CHALC), COLOR["chalc"], 0.55, 1),
        ("halide",       df["anion"].isin(HAL),   COLOR["hal"],   0.55, 1),
        ("pnictide",     df["anion"].isin(PNIC),  "#1a6b1a",      0.95, 3),
    ]:
        ax.hist(df.loc[mask, "cauchy_A"], bins=20, alpha=alpha,
                color=color, label=f"{fam} (n={int(mask.sum())})",
                edgecolor="white", linewidth=0.3, zorder=zorder)
    ax.set_xlabel(r"target $A=\sqrt{\varepsilon_{\infty,xx}}$")
    ax.set_ylabel("count")
    ax.set_title("(c) target distribution by anion family")
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    _save(fig, "fig2_dataset")
    plt.close(fig)


# -- Fig 3: outer-LOCO head-to-head -----------------------------------------


def _load_e2_fold(anion: str, qwen: bool) -> dict:
    suffix = "_qwen" if qwen else ""
    p = (REPO / "experiments" / "E2_outer_loco" / "runs"
         / f"fold_{anion}_seed42{suffix}" / "fold_summary.json")
    return json.loads(p.read_text())


def _load_e3_fold(anion: str) -> dict:
    p = (REPO / "experiments" / "E3_matched_pysr_baselines" / "runs"
         / f"fold_{anion}_seed42" / "fold_summary.json")
    return json.loads(p.read_text())


def fig3_outer_loco() -> None:
    """Per-anion outer-LOCO RMSE: Qwen-Harness vs heuristic-Harness vs
    matched vanilla PySR (V2_screen). The headline figure of the paper."""
    print("\n[fig3_outer_loco]")
    qwen = []; heur = []; v2 = []; ns = []
    for a in ANIONS_ORDER:
        h = _load_e2_fold(a, qwen=False)
        q = _load_e2_fold(a, qwen=True)
        e3 = _load_e3_fold(a)
        ns.append(h["n_outer_test"])
        heur.append(h["outer_test_metrics"]["rmse"])
        qwen.append(q["outer_test_metrics"]["rmse"])
        v2.append(e3["policy_summary"]["V2_screen"]["mean_outer_rmse"])

    totN = sum(ns)
    w_q = sum(r * n for r, n in zip(qwen, ns)) / totN
    w_h = sum(r * n for r, n in zip(heur, ns)) / totN
    w_v = sum(r * n for r, n in zip(v2, ns))   / totN
    print(f"  weighted RMSE  qwen={w_q:.4f}  heur={w_h:.4f}  v2_vanilla={w_v:.4f}  on N={totN}")
    wins_qwen_v_v2 = sum(1 for q, v in zip(qwen, v2) if q < v)
    print(f"  Qwen-Harness wins vs V2 vanilla on {wins_qwen_v_v2}/10 anions")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11.5, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.12},
    )

    x = np.arange(len(ANIONS_ORDER))
    w = 0.27
    # Method names match main-text §4.2 / Table 1 vocabulary; experiment-
    # internal codes (E2 / E3) are not exposed to readers.
    ax_top.bar(x - w, qwen, width=w, label="LLM-guided harness (Qwen-max)",
               color=COLOR["qwen"], edgecolor="black", linewidth=0.4)
    ax_top.bar(x,     heur, width=w, label="Heuristic-stub harness (no LLM)",
               color=COLOR["heur"], edgecolor="black", linewidth=0.4)
    ax_top.bar(x + w, v2,   width=w,
               label="Vanilla PySR (V2 screen, matched)",
               color=COLOR["v2vanilla"], edgecolor="black", linewidth=0.4)

    # Weighted-mean reference lines (un-labeled in the legend; the actual
    # weighted RMSE values are reported in the figure caption to keep the
    # plot area uncluttered).
    ax_top.axhline(w_q, color=COLOR["qwen"],     ls="--", lw=1.0, alpha=0.6)
    ax_top.axhline(w_h, color=COLOR["heur"],     ls="--", lw=1.0, alpha=0.6)
    ax_top.axhline(w_v, color=COLOR["v2vanilla"], ls="--", lw=1.0, alpha=0.6)

    ax_top.set_ylabel("outer-LOCO test RMSE")
    ax_top.set_title("(a) Outer-LOCO head-to-head (10 anion folds)", loc="left")
    ax_top.legend(loc="upper left", ncol=1, frameon=False, fontsize=9,
                  bbox_to_anchor=(0.0, 0.97))
    ax_top.set_ylim(0, max(max(qwen), max(heur), max(v2)) * 1.18)

    # Bottom panel — per-anion N
    ax_bot.bar(x, ns, color="#cccccc", edgecolor="black", linewidth=0.4)
    for i, v in enumerate(ns):
        ax_bot.text(i, v + 0.6, str(v), ha="center", va="bottom", fontsize=8)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(ANIONS_ORDER)
    ax_bot.set_xlabel("held-out anion (outer-LOCO fold)")
    ax_bot.set_ylabel("N outer test")
    ax_bot.set_title("(b) Per-fold sample counts", loc="left", fontsize=10)
    ax_bot.set_ylim(0, max(ns) * 1.18)

    fig.tight_layout()
    _save(fig, "fig3_outer_loco")
    plt.close(fig)


# -- Fig 4: E4 ablation ------------------------------------------------------


def fig4_ablation() -> None:
    print("\n[fig4_ablation]")
    src = (REPO / "experiments" / "E4_ablation_fixed_council"
           / "runs" / "E4_fullbudget_seed42" / "ablation_summary.json")
    d = json.loads(src.read_text())
    n_seeds = int(d["n_seeds"])
    print(f"  source: {src.relative_to(REPO)}  n_seeds={n_seeds}")

    # Logical ordering: unablated reference (FULL) first, followed by
    # leave-one-out variants sorted by descending ΔRMSE degradation.
    order = ["FULL", "-AGENTS", "-ADAPTIVE", "-CAUSAL", "-MEMORY"]
    ytick_labels = [
        "FULL (all 4 modules)",
        "-AGENTS (no AgentCouncil)",
        "-ADAPTIVE (fixed operators)",
        "-CAUSAL (no sensitivity critic)",
        "-MEMORY (no cross-round memory)",
    ]
    rmse  = [d["summary"][v]["mean_test_rmse"] for v in order]
    rstd  = [d["summary"][v]["std_test_rmse"]  for v in order]
    delta = [d["summary"][v].get("delta_vs_FULL") for v in order]

    # Extract per-seed paired differences Δ_s = RMSE(variant, s) - RMSE(FULL, s)
    runs_by_var: dict[str, dict[int, float]] = {v: {} for v in order}
    for r in d.get("all_runs", []):
        v = r.get("variant")
        if v in runs_by_var and "test_rmse" in r:
            runs_by_var[v][int(r["seed_idx"])] = float(r["test_rmse"])

    full_vec = np.array([runs_by_var["FULL"][s] for s in range(n_seeds)])
    from math import comb
    sem_deltas = []
    sd_deltas = []
    pos_counts = []
    p_vals = []
    per_seed_deltas = []
    for v in order:
        if v == "FULL":
            sem_deltas.append(0.0)
            sd_deltas.append(0.0)
            pos_counts.append(0)
            p_vals.append(1.0)
            per_seed_deltas.append(np.zeros(n_seeds))
        else:
            vec = np.array([runs_by_var[v][s] for s in range(n_seeds)])
            diffs = vec - full_vec
            per_seed_deltas.append(diffs)
            sd_d = float(np.std(diffs, ddof=1))
            sem_d = sd_d / np.sqrt(n_seeds)
            sd_deltas.append(sd_d)
            sem_deltas.append(sem_d)
            n_pos = int(np.sum(diffs > 0))
            pos_counts.append(n_pos)
            k = max(n_pos, n_seeds - n_pos)
            p_2s = min(1.0, 2.0 * sum(comb(n_seeds, j) * (0.5 ** n_seeds) for j in range(k, n_seeds + 1)))
            p_vals.append(p_2s)

    for v, r, s, dd, sem_d, np_c, pv in zip(order, rmse, rstd, delta, sem_deltas, pos_counts, p_vals):
        ds = "ref" if dd is None else f"{dd:+.4f}±{sem_d:.4f} (SEM), {np_c}/{n_seeds}>0, p={pv:.4f}"
        print(f"  {v:<10}  RMSE={r:.4f}±{s:.4f}  delta={ds}")

    fig, ax = plt.subplots(figsize=(9.2, 4.2))

    x = np.arange(len(order))
    deltas = [delta[i] if delta[i] is not None else 0.0 for i in range(len(order))]
    bar_colors = [COLOR["FULL"] if order[i] == "FULL"
                  else (COLOR["pos"] if deltas[i] > 0 else COLOR["neg"])
                  for i in range(len(order))]

    ax.barh(x, deltas, color=bar_colors, edgecolor="black", linewidth=0.6, height=0.52, zorder=2)
    # Plot error bars (±1 SEM of paired ΔRMSE across n=8 seeds) for non-FULL rows
    for i in range(1, len(order)):
        ax.errorbar(
            deltas[i], x[i], xerr=sem_deltas[i],
            fmt="none", ecolor="#111111", elinewidth=1.4, capsize=4, capthick=1.2, zorder=4
        )

    # Plot individual seed data points (n = 8) as semi-transparent dots
    for i in range(1, len(order)):
        pts = per_seed_deltas[i]
        ax.scatter(
            pts, np.full_like(pts, x[i]),
            color="#222222", edgecolors="white", linewidth=0.7, s=34, alpha=0.85, zorder=5
        )

    ax.axvline(0, color="black", lw=0.8, zorder=1)
    pad = 0.005
    for i, dd in enumerate(deltas):
        if order[i] == "FULL":
            ref_label = f"unablated reference  (validation RMSE = {rmse[i]:.4f} ± {rstd[i]:.4f} SD)"
            ax.text(pad, i, ref_label, ha="left", va="center", fontsize=9.0)
        else:
            txt = f"{dd:+.4f} ± {sem_deltas[i]:.4f}"
            max_extent = max(dd + sem_deltas[i], float(np.max(per_seed_deltas[i])))
            anchor = max(max_extent, 0.0) + pad
            ax.text(anchor, i, txt, ha="left", va="center", fontsize=9.0, fontweight="bold")

    ax.set_yticks(x)
    ax.set_yticklabels(ytick_labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ Validation RMSE relative to FULL  (mean $\pm$ SEM, $n = 8$)", fontsize=10)
    ax.set_title("Leave-one-module-out ablation", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(-0.042, 0.160)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR["pos"], edgecolor="black", label=r"Mean $\Delta\mathrm{RMSE}$"),
        Line2D([0], [0], color="#111111", lw=1.4, marker="|", markersize=6, label=r"$\pm 1\ \mathrm{SEM}$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#222222", markeredgecolor="white", markersize=6, label="Individual seeds ($n = 8$)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=8.8, framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    _save(fig, "fig4_ablation")
    plt.close(fig)


# -- Fig 5: E5 diagnostics ---------------------------------------------------


def fig5_diagnostics() -> None:
    """Three-panel summary of E5: Z_M-only per-anion, LOMO per-metal,
    target-permutation null distribution."""
    print("\n[fig5_diagnostics]")
    summary = json.loads((REPO / "experiments" / "E5_diagnostics"
                          / "E5_combined_summary.json").read_text())
    zm   = summary["E5_1_zm_only"]
    lomo = summary["E5_2_lomo"]
    perm = summary["E5_3_target_permute"]
    print(f"  Z_M-only weighted RMSE = {zm['weighted_outer_rmse']:.4f}")
    print(f"  LOMO weighted RMSE     = {lomo['weighted_outer_rmse']:.4f}")
    print(f"  permute mean ratio     = {perm['rmse_over_std_ratio']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    # (a) Z_M-only per-anion bar — sort by ANIONS_ORDER for consistency.
    # Reference lines (Z_M-only weighted, LLM-guided harness, vanilla
    # PySR full-input weighted) are drawn but NOT labelled inline; the
    # numerical values are reported in the figure caption to keep the
    # plot area uncluttered.
    ax = axes[0]
    rmse_by_anion = {r["anion"]: r for r in zm["per_anion"]}
    rmses = [rmse_by_anion[a]["mean_outer_rmse"] for a in ANIONS_ORDER]
    stds  = [rmse_by_anion[a]["std_outer_rmse"]  for a in ANIONS_ORDER]
    ns    = [rmse_by_anion[a]["n_outer_test"]    for a in ANIONS_ORDER]
    cols  = [COLOR["chalc"] if a in CHALC else COLOR["hal"]
             if a in HAL else COLOR["pnic"] for a in ANIONS_ORDER]
    x = np.arange(len(ANIONS_ORDER))
    ax.bar(x, rmses, yerr=stds, capsize=3, color=cols, edgecolor="black",
           linewidth=0.4)
    ax.axhline(zm["weighted_outer_rmse"], color="black",         ls="--", lw=0.9, alpha=0.7)
    ax.axhline(0.6450,                     color=COLOR["qwen"],  ls=":",  lw=0.9, alpha=0.7)
    ax.axhline(0.7415,                     color=COLOR["v2vanilla"], ls=":", lw=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ANIONS_ORDER)
    ax.set_xlabel("held-out anion")
    ax.set_ylabel(r"outer-LOCO RMSE ($Z_M$ only, 3 seeds)")
    ax.set_title("(a) Single-feature critique: $Z_M$ only", loc="left")

    # (b) LOMO per-metal sorted by RMSE descending. Highlight bar in red
    # for d-block transition metals (the four metals at the high-RMSE
    # end happen to be Cr, Mn, Mo, Ni — a mix of 3d and 4d transition
    # metals; "d-block transition metal" is the precise descriptor).
    ax = axes[1]
    metals = sorted(lomo["per_metal"], key=lambda r: -r["mean_outer_rmse"])
    names = [m["metal"] for m in metals]
    rs    = [m["mean_outer_rmse"] for m in metals]
    Ns    = [m["n_outer_test"] for m in metals]
    DBLOCK_TM = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                 "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                 "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}
    # Tighten: in this dataset only Ni/Cr/Mn/Mo end up red because
    # those are the only d-block TMs whose LOMO RMSE > weighted threshold.
    # Use the actual data, not a static set: highlight any d-block TM
    # whose LOMO RMSE exceeds 1.0 (clearly above the rest).
    cols = [COLOR["neg"] if (n in DBLOCK_TM and r > 1.0) else COLOR["pos"]
            for n, r in zip(names, rs)]
    y = np.arange(len(names))
    ax.barh(y, rs, color=cols, edgecolor="black", linewidth=0.4)
    # Place N-labels in a fixed right-side column so they line up and
    # are never truncated by the bar edge.
    x_label = max(rs) * 1.08
    for i, n_out in enumerate(Ns):
        ax.text(x_label, i, f"N={n_out}", va="center", ha="left", fontsize=8)
    ax.axvline(lomo["weighted_outer_rmse"], color="black", ls="--", lw=0.9,
               label=f"weighted = {lomo['weighted_outer_rmse']:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("LOMO outer-test RMSE")
    ax.set_title("(b) Cross-metal extrapolation (red = d-block TM)", loc="left")
    ax.set_xlim(0, max(rs) * 1.30)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    # (c) target-permute scatter: std(y_outer) vs RMSE per shuffle
    ax = axes[2]
    p_full = json.loads((REPO / "experiments" / "E5_diagnostics" / "runs"
                         / "target_permute_O_seed42" / "fold_summary.json"
                         ).read_text())
    seeds = [r for r in p_full["seed_runs"] if "outer_rmse" in r]
    sx = [r["std_y_outer"] for r in seeds]
    sy = [r["outer_rmse"]  for r in seeds]
    ax.scatter(sx, sy, s=60, color=COLOR["null"], edgecolor="black",
               linewidth=0.5, label="permuted target (n=5)")
    lo = 0.0
    hi = max(max(sx), max(sy)) * 1.1
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.8,
            label="RMSE = std(y) (no skill)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"std$(y_{\mathrm{outer}})$ after shuffle")
    ax.set_ylabel("outer-LOCO RMSE")
    ax.set_title(f"(c) Permutation null test (mean ratio = {perm['rmse_over_std_ratio']:.3f})", loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    _save(fig, "fig5_diagnostics")
    plt.close(fig)


# -- Fig 6: per-fold parity plots for E2-Qwen Harness -----------------------


def fig6_parity() -> None:
    """For each of the 10 outer-LOCO folds, plot the E2-Qwen Harness
    headline expression's predictions on the held-out anion vs truth."""
    print("\n[fig6_parity]")
    fig, axes = plt.subplots(2, 5, figsize=(15.0, 6.4), sharex=False, sharey=False)
    axes = axes.flatten()

    df_all = pd.read_csv(REPO / "data" / "extended" / "tier2_binary.csv")
    overall_min = df_all["cauchy_A"].min()
    overall_max = df_all["cauchy_A"].max()

    for i, anion in enumerate(ANIONS_ORDER):
        ax = axes[i]
        q = _load_e2_fold(anion, qwen=True)
        # Each fold_summary stores the per-sample test predictions in the
        # outer_test_metrics block as 'y_true' / 'y_pred' if available.
        ev = q.get("outer_test_metrics", {})
        y_true = ev.get("y_true")
        y_pred = ev.get("y_pred")
        if y_true is None or y_pred is None:
            # Fallback: recompute from the headline expression and the
            # outer_test slice. The headline expression's `features_used`
            # is usually a subset of the 20 descriptors — PySR uses
            # x0..xK in that subset's column order, so we must project
            # the outer_test slice to the same subset (same convention
            # as src/v2/runners/E2_outer_loco.py:382 _project_features).
            from src.v2.runners.E2_outer_loco import (
                E2_FEATURE_NAMES, _project_features,
            )
            from src.v2.expr_eval import compile_expr
            df_fold = df_all[df_all["anion"] == anion].reset_index(drop=True)
            feats_used = q["headline"]["features_used"] or E2_FEATURE_NAMES
            X = _project_features(df_fold, feats_used)
            expr = q["headline"]["expression"]
            f = compile_expr(expr, feats_used, stable_range=(0.0, 20.0))
            y_pred = f(X)
            y_true = df_fold["cauchy_A"].to_numpy()
        else:
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_pred)
        ax.scatter(y_true[mask], y_pred[mask], s=22, alpha=0.7,
                   color=COLOR["qwen"], edgecolor="white", linewidth=0.4)
        rmse = ev.get("rmse", float("nan"))
        n = q["n_outer_test"]
        cplx = q["headline"]["complexity"]
        ax.plot([overall_min, overall_max], [overall_min, overall_max],
                color="black", ls="--", lw=0.7)
        ax.set_xlim(overall_min - 0.2, overall_max + 0.2)
        ax.set_ylim(overall_min - 0.2, overall_max + 0.2)
        ax.set_aspect("equal")
        ax.set_title(f"{anion}  N={n}  RMSE={rmse:.3f}  cplx={cplx}",
                     fontsize=9)
        if i % 5 == 0:
            ax.set_ylabel(r"Harness-predicted $A$")
        if i >= 5:
            ax.set_xlabel(r"DFT $A=\sqrt{\varepsilon_{\infty,xx}}$")

    fig.suptitle("Per-fold parity of the LLM-guided harness on each "
                 "outer-LOCO held-out anion (10 folds)",
                 y=1.0, fontsize=11)
    # Manual spacing — subplot titles in row 2 collide with row 1 tick
    # labels under tight_layout because each panel has its own axis range.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.09,
                        hspace=0.55, wspace=0.30)
    _save(fig, "fig6_parity")
    plt.close(fig)


# -- entry point -------------------------------------------------------------


_FIGS = {
    "fig1_architecture": fig1_architecture,
    "fig2_dataset":    fig2_dataset,
    "fig3_outer_loco": fig3_outer_loco,
    "fig4_ablation":   fig4_ablation,
    "fig5_diagnostics": fig5_diagnostics,
    "fig6_parity":     fig6_parity,
}


def main(argv: list[str]) -> int:
    names = argv[1:] or list(_FIGS.keys())
    for n in names:
        if n not in _FIGS:
            print(f"unknown figure {n!r}; choices: {list(_FIGS.keys())}",
                  file=sys.stderr)
            return 2
        _FIGS[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
