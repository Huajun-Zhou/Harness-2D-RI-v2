# E4 — Governance ablation

5 variants × 3 seeds. Budget: 5 rounds × 50 iter × 40 islands. LLM driver: Qwen-max (DashScope). Target: held-out-slice RMSE.

| Variant | Seeds | RMSE (mean ± std) | Complexity (mean ± std) | Δ vs FULL |
|---|---|---|---|---|
| FULL | 3/3 | 0.5685 ± 0.0286 | 21.3 ± 3.2 | ref |
| -AGENTS | 3/3 | 0.6491 ± 0.0181 | 21.3 ± 5.5 | +0.0806 |
| -CAUSAL | 3/3 | 0.5863 ± 0.0301 | 17.7 ± 6.4 | +0.0178 |
| -ADAPTIVE | 3/3 | 0.5477 ± 0.0178 | 16.0 ± 5.3 | -0.0208 |
| -MEMORY | 3/3 | 0.5779 ± 0.0294 | 18.3 ± 2.1 | +0.0094 |

## Notes
- Removing the **AGENTS** module degrades RMSE the most (Δ = +0.0806); it is the dominant governance contributor in this budget.
- Removing the **ADAPTIVE** module marginally *improves* RMSE (Δ = -0.0208), suggesting that under the reduced E4 budget its operator-schedule does not yet pay for itself.
- Complexity shifts are small; none of the ablations produce drastically simpler or more ornate expressions on average.
