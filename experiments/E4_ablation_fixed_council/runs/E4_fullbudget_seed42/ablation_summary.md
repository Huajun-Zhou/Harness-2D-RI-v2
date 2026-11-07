# E4 — Governance ablation

5 variants × 8 seeds. Budget: 10 rounds × 100 iter × 40 islands. LLM driver: Qwen-max (DashScope). Target: held-out-slice RMSE.

| Variant | Seeds | RMSE (mean ± std) | Complexity (mean ± std) | Δ vs FULL |
|---|---|---|---|---|
| FULL | 8/8 | 0.5457 ± 0.0196 | 18.0 ± 4.9 | ref |
| -AGENTS | 8/8 | 0.6187 ± 0.0251 | 21.1 ± 2.2 | +0.0730 |
| -ADAPTIVE | 8/8 | 0.5591 ± 0.0171 | 18.2 ± 3.8 | +0.0134 |
| -CAUSAL | 8/8 | 0.5557 ± 0.0126 | 14.6 ± 3.3 | +0.0100 |
| -MEMORY | 8/8 | 0.5455 ± 0.0192 | 17.5 ± 4.1 | -0.0002 |

## Notes
- Removing the **AGENTS** module degrades RMSE the most (Δ = +0.0730); it is the dominant governance contributor in this budget.
- Removing the **MEMORY** module marginally *improves* RMSE (Δ = -0.0002), suggesting that under the reduced E4 budget its operator-schedule does not yet pay for itself.
- Complexity shifts are small; none of the ablations produce drastically simpler or more ornate expressions on average.
