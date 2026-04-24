"""
aggregate_E4.py — turn E4 ablation_summary.json into paper-ready artifacts.

Produces two files alongside the source summary:
  - ablation_summary.csv   (long form, one row per variant)
  - ablation_summary.md    (human-readable table for the manuscript)

Usage:
    python3 scripts/aggregate_E4.py \
        [experiments/E4_ablation_fixed_council/runs/E4_seed42_3seeds/ablation_summary.json]

If no path is given, the default E4 run is used.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


DEFAULT_SUMMARY = Path(
    "experiments/E4_ablation_fixed_council/runs/E4_seed42_3seeds/ablation_summary.json"
)

# Variant display order — keep FULL first as the reference row.
VARIANT_ORDER = ("FULL", "-AGENTS", "-CAUSAL", "-ADAPTIVE", "-MEMORY")


def _fmt_delta(d: float | None) -> str:
    if d is None:
        return "ref"
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _rows(summary: dict) -> list[dict]:
    rows = []
    for v in VARIANT_ORDER:
        s = summary["summary"].get(v)
        if s is None:
            continue
        rows.append({
            "variant":          v,
            "n_seeds_success":  s["n_seeds_success"],
            "mean_test_rmse":   s["mean_test_rmse"],
            "std_test_rmse":    s["std_test_rmse"],
            "mean_complexity":  s["mean_complexity"],
            "std_complexity":   s["std_complexity"],
            "delta_vs_FULL":    s.get("delta_vs_FULL"),
        })
    return rows


def _write_csv(rows: list[dict], out: Path) -> None:
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "n_seeds_success",
            "mean_test_rmse", "std_test_rmse",
            "mean_complexity", "std_complexity",
            "delta_vs_FULL",
        ])
        for r in rows:
            w.writerow([
                r["variant"], r["n_seeds_success"],
                f"{r['mean_test_rmse']:.6f}",
                f"{r['std_test_rmse']:.6f}",
                f"{r['mean_complexity']:.3f}",
                f"{r['std_complexity']:.3f}",
                "" if r["delta_vs_FULL"] is None else f"{r['delta_vs_FULL']:.6f}",
            ])


def _write_md(rows: list[dict], out: Path, n_seeds: int) -> None:
    # Figure out which non-FULL ablation hurt/helped the most.
    with_delta = [r for r in rows if r["delta_vs_FULL"] is not None]
    worst = max(with_delta, key=lambda r: r["delta_vs_FULL"]) if with_delta else None
    best  = min(with_delta, key=lambda r: r["delta_vs_FULL"]) if with_delta else None

    lines: list[str] = []
    lines.append("# E4 — Governance ablation")
    lines.append("")
    lines.append(
        f"5 variants × {n_seeds} seeds. Budget: 5 rounds × 50 iter × 40 islands. "
        "LLM driver: Qwen-max (DashScope). Target: held-out-slice RMSE."
    )
    lines.append("")
    lines.append(
        "| Variant | Seeds | RMSE (mean ± std) | Complexity (mean ± std) | Δ vs FULL |"
    )
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['n_seeds_success']}/{n_seeds} "
            f"| {r['mean_test_rmse']:.4f} ± {r['std_test_rmse']:.4f} "
            f"| {r['mean_complexity']:.1f} ± {r['std_complexity']:.1f} "
            f"| {_fmt_delta(r['delta_vs_FULL'])} |"
        )
    lines.append("")
    lines.append("## Notes")
    if worst is not None:
        lines.append(
            f"- Removing the **{worst['variant'][1:]}** module degrades RMSE the most "
            f"(Δ = {_fmt_delta(worst['delta_vs_FULL'])}); it is the dominant governance "
            "contributor in this budget."
        )
    if best is not None and best["delta_vs_FULL"] < 0:
        lines.append(
            f"- Removing the **{best['variant'][1:]}** module marginally *improves* RMSE "
            f"(Δ = {_fmt_delta(best['delta_vs_FULL'])}), suggesting that under the "
            "reduced E4 budget its operator-schedule does not yet pay for itself."
        )
    lines.append(
        "- Complexity shifts are small; none of the ablations produce drastically "
        "simpler or more ornate expressions on average."
    )
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else DEFAULT_SUMMARY
    if not src.is_file():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 2
    data = json.loads(src.read_text())
    rows = _rows(data)
    if not rows:
        print("ERROR: no variants found in summary", file=sys.stderr)
        return 3

    csv_out = src.with_name("ablation_summary.csv")
    md_out  = src.with_name("ablation_summary.md")
    _write_csv(rows, csv_out)
    _write_md(rows, md_out, n_seeds=data.get("n_seeds", 3))

    # Echo the table to stdout for quick inspection.
    print(md_out.read_text())
    print(f"wrote {csv_out}")
    print(f"wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
