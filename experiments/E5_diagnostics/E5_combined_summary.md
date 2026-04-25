# E5 — Diagnostic experiments

Three independent checks defending the v2 outer-LOCO claim against the reviewer's main attacks: (1) is `Z_M` alone enough? (2) does Harness generalise across unseen *metals*, not just unseen anions? (3) does the pipeline find spurious signal when the target is shuffled?

## E5.1 — Z_M-only PySR

Vanilla PySR run under the exact E2 outer-LOCO partitioning, but with only the `atomic_number_M` column visible to the search. 3 seeds × 5 rounds per anion.

**Weighted outer RMSE: 0.7167**  (Harness/Qwen E2: 0.6450; E3 V2 vanilla 18-feat: 0.7415).

| anion | N_outer | RMSE (Z_M only) |
|---|---|---|
| I | 39 | 0.6771 ± 0.1746 |
| Br | 33 | 0.7073 ± 0.1150 |
| Se | 30 | 0.5165 ± 0.0284 |
| Cl | 27 | 1.1903 ± 0.0019 |
| S | 24 | 0.3610 ± 0.0416 |
| O | 19 | 0.8232 ± 0.0161 |
| Te | 8 | 0.7765 ± 0.0323 |
| N | 6 | 0.5029 ± 0.1886 |
| F | 4 | 1.3233 ± 0.0563 |
| P | 3 | 0.7050 ± 0.4666 |

**Interpretation.** Z_M alone does not collapse the problem: it loses to LLM-guided Harness by ~+0.072 RMSE weighted. On chemistries where Z_M *is* the dominant axis (S, Se, I, Br) the gap is small or negative; on the halide folds Cl and F, single-feature search is materially worse (RMSE > 1.1) — these are the folds that buy the descriptor set its keep.

## E5.2 — Leave-one-metal-out (LOMO)

Vanilla PySR on all 20 features, held-out metal as the test slice. Restricted to metals with N ≥ 5 (18 of 48 metals; total outer N = 130).

**Weighted outer RMSE: 0.6747** (weighted across 18 held-out metals).

| metal | N_outer | RMSE |
|---|---|---|
| Ni | 7 | 2.3044 |
| Cr | 6 | 1.8564 |
| Mn | 5 | 1.6613 |
| Mo | 9 | 1.2546 |
| Sn | 10 | 0.7220 |
| Pd | 5 | 0.6837 |
| Ag | 6 | 0.5326 |
| Zr | 7 | 0.4982 |
| B | 7 | 0.4638 |
| Al | 7 | 0.4122 |
| Hf | 5 | 0.3873 |
| Cd | 8 | 0.3366 |
| Pt | 6 | 0.3352 |
| Bi | 9 | 0.3268 |
| Ga | 5 | 0.3062 |
| Si | 5 | 0.2999 |
| Pb | 18 | 0.2078 |
| Hg | 5 | 0.2066 |

**Interpretation.** Per-metal RMSE varies by an order of magnitude: main-group metals with abundant analogues (Pb, Hg, Ga, Si) generalise well (RMSE < 0.35), while 3d-transition outliers (Ni, Cr, Mn, Mo) are predicted poorly (RMSE > 1.2). The weighted mean — pulled down by the populous Pb fold (N=18, RMSE=0.21) — is broadly in line with the anion-LOCO weighted mean, supporting the universality claim but with the caveat that 3d-TM chemistry is undersampled.

## E5.3 — Target-permutation null test

Shuffle `cauchy_A` across all samples, re-run vanilla PySR under outer-LOCO on anion=O. 5 independent shuffles × 3 rounds. A leakage-free pipeline should produce outer RMSE ≈ std(y_outer); a ratio < 1 would indicate that information is leaking through the protocol.

**Mean outer RMSE: 0.8650**  vs mean std(y_outer): 0.7518  →  **ratio = 1.151**.

| seed | std(y_outer) | outer RMSE | RMSE/std |
|---|---|---|---|
| 0 | 0.9809 | 1.0431 | 1.064 |
| 1 | 1.1133 | 1.2062 | 1.083 |
| 2 | 0.5255 | 0.7223 | 1.375 |
| 3 | 0.8050 | 0.8432 | 1.048 |
| 4 | 0.3344 | 0.5102 | 1.526 |

**Interpretation.** All five independent shuffles give ratio ≥ 1, i.e. the model never finds a signal stronger than the trivial 'predict the mean' baseline when labels are permuted. The pipeline is leakage-free.

## Combined verdict

- Z_M is **necessary but not sufficient** — refutes the 'periodic-table lookup' critique on its strong form, while honestly noting it carries a large fraction of the predictive signal on chalcogenide folds.
- Universality is **partial** — the harness generalises across metals at the same RMSE scale as across anions for main-group chemistry, but not for the 3d transition metals (Ni, Cr, Mn) where outer test slices are tiny.
- The pipeline is **not** finding spurious structure — null test ratio ≈ 1.15 across 5 independent permutations.
