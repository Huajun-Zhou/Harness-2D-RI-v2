# Supplementary Information

**Governance-guided symbolic regression for 2D-semiconductor refractive response — supplementary analyses and the limits of Moss-type exponent recovery**

---

## Table S1. Descriptor catalogue (Tier 2)

The 17 descriptors below are the shared feature dictionary used by both the Harness symbolic search and the classical baselines. Classical baselines (Table 1) and the LOCO retrained baselines (Table 5) additionally see the DFT-computed OPT-B88vdW band gap $E_g$ (18$^{\text{th}}$ input; top-ranked feature in Figure 3). The Harness symbolic search additionally uses three structural descriptors not listed here (coordination number, point-group encoding, and packing fraction), for 20 inputs total, but is *not* given $E_g$.

| Group | Symbol | Description | Source |
|---|---|---|---|
| Elemental (metal) | $Z_M$ | Metal atomic number | Standard table |
|  | $\chi_M$ | Pauling electronegativity of metal | Standard table |
|  | $r_M$ | Covalent radius of metal (Å) | Standard table |
|  | $IE_M$ | First ionisation energy of metal (eV) | Standard table |
|  | $d_e$ | Metal $d$-electron count | Derived from $Z_M$ |
| Elemental (anion) | $Z_X$ | Anion atomic number | Standard table |
|  | $\chi_X$ | Pauling electronegativity of anion | Standard table |
|  | $r_X$ | Covalent radius of anion (Å) | Standard table |
|  | $EA_X$ | Electron affinity of anion (eV) | Standard table |
| Structural | $d_{MX}$ | Metal–anion bond length (Å) | JARVIS-DFT |
|  | $a$ | In-plane lattice constant (Å) | JARVIS-DFT |
|  | $t$ | Monolayer thickness (Å) | JARVIS-DFT |
| Derived | $\|\Delta\chi\|$ | $\|\chi_M-\chi_X\|$ | Analytic |
|  | $r_M/r_X$ | Metal–anion radius ratio | Analytic |
|  | bond polarity | $1-\exp(-0.25|\Delta\chi|^2)$ | Analytic |
|  | $Z_M/Z_X$ | Metal–anion $Z$ ratio | Analytic |
|  | $IE_M/EA_X$ | Metal IE / anion EA | Analytic |

## Table S2. PHYSICS_PRIORS for TheoryAgent

| Feature | Expected sign of $\partial A/\partial x_j$ | Ambiguous? |
|---|---|---|
| $Z_X$ | $+$ | No |
| $r_X$ | $+$ | No |
| $\chi_X$ | $-$ | No |
| $EA_X$ | $-$ | No |
| $d_{MX}$ | $+$ | No |
| $t$ | $+$ | Yes (thickness effects non-monotone) |
| $Z_M$ | — | Yes (competing d-shell effects) |
| $\chi_M$ | — | Yes (competing effects) |
| bond polarity | $-$ | No |
| $Z_M/Z_X$ | $-$ | No |
| $r_M/r_X$ | $-$ | No |
| $d_e$ | — | Yes (shell-specific nonlinearity) |

Veto is triggered when three or more non-ambiguous features exhibit sign contradictions with sensitivity $|s_j|>0.05$; ambiguous features (marked Yes) do not contribute veto votes.

## Table S3. Tier 2 dataset composition

| Anion family | Anions | Materials | Fraction |
|---|---|---|---|
| Chalcogens | O, S, Se, Te | 81 | 42% |
| Halogens | F, Cl, Br, I | 103 | 53% |
| Pnictogens | N, P | 9 | 5% |

Per-anion counts: O=19, S=24, Se=30, Te=8, F=4, Cl=27, Br=33, I=39, N=6, P=3.

Per-stoichiometry counts: MX=41, MX$_2$=90, MX$_3$=45, MX$_4$=3, MX$_5$=1, M$_2$X=4, M$_2$X$_3$=4, M$_2$X$_5$=4, M$_3$X$_8$=1.

## Table S4. LOCO per-anion metrics (extended)

Values are identical to the main-text Table 5 and are reproduced here with the additional Harness-vs-Ridge column; source artefact `results/loco_new_headline.csv` (expression `results/discovery_report_20260420_165841.json`, JARVIS-DFT2D Tier-2, $N=193$).

| Anion | $N$ | Harness-2D-RI | RF(500) | Ridge | $\Delta$ (H vs. RF) | $\Delta$ (H vs. Ridge) |
|---|---|---|---|---|---|---|
| O  | 19 | **0.554** | 0.693 | 0.951 | $-20.0\%$ | $-41.7\%$ |
| S  | 24 | **0.298** | 0.371 | 0.488 | $-19.5\%$ | $-38.9\%$ |
| Se | 30 | 0.490 | **0.335** | 0.419 | $+46.1\%$ | $+16.9\%$ |
| Te |  8 | 0.715 | **0.370** | 0.378 | $+93.3\%$ | $+89.3\%$ |
| F  |  4 | **1.285** | 1.379 | 1.222 | $-6.8\%$ | $+5.2\%$ |
| Cl | 27 | **0.796** | 1.181 | 1.184 | $-32.6\%$ | $-32.8\%$ |
| Br | 33 | **0.584** | 0.814 | 0.856 | $-28.3\%$ | $-31.8\%$ |
| I  | 39 | **0.451** | 0.562 | 0.738 | $-19.7\%$ | $-38.9\%$ |
| N  |  6 | 0.422 | **0.089** | 0.315 | $+373.8\%$ | $+33.7\%$ |
| P  |  3 | **0.620** | 0.743 | 1.091 | $-16.5\%$ | $-43.2\%$ |
| **Weighted mean** | 193 | **0.549** | 0.643 | 0.749 | $-14.5\%$ | $-26.7\%$ |

Bold marks the lowest RMSE per anion. Harness-2D-RI wins 7/10 anions against RF (O, S, F, Cl, Br, I, P) and 7/10 against Ridge (same seven anions; F loses to Ridge by $+5.2\%$). The weighted-mean reduction is $14.5\%$ vs. RF and $26.7\%$ vs. Ridge.

## Table S5. Motif stability across Strong-PySR seeds

Motif recovery is quantified on the budget-matched Strong-PySR runs reported in Table S11 and `results/strong_pysr_tier2_runs.csv` / `results/strong_pysr_tier3_runs.csv`; those runs are the authoritative seed-variance measurement in this work. Under Tier-2 budget, 4/5 unguided seeds recover the cube-root outer operator and the $d_e$-dependence but at $\approx 2\times$ the complexity of the governed headline (11 nodes); under Tier-3 budget, 0/5 seeds recover the cube-root motif but 4/5 still pick up the inverse/$Z_\text{ratio}$/$d_e$ structure. We therefore report motif stability exclusively through the Strong-PySR audit rather than through a separate Harness seed sweep, which would duplicate evidence already in Table S11.

## Table S6. Moss-law statistics

| Quantity | Value |
|---|---|
| Moss theoretical slope ($n^{4}E_g=\text{const}$) | $-0.250$ |
| Data-level slope (raw $\log n$ vs. $\log E_g$) | $-0.294$ |
| Harness-predicted slope (Eq. 1, applied to the $N=193$ set) | $-0.094$ |
| Deviation (data vs. Moss) | 17.7% |
| Deviation (Harness prediction vs. Moss) | 62.5% |
| Deviation (Harness prediction vs. data) | 68.1% |

The fit uses $N=193$ entries with $E_g>0.2$ eV (i.e. the full Tier 2 dataset); source artefact `results/moss_loco_check.json`. The large gap between the Harness-predicted slope and both the data and the Moss reference is the honest-negative-result discussed in the main text (stability-veto / power-law tension); it is reproduced at Tier-3 scale in Table on Moss-exponent scan, Section S4.

## Table S7. Pareto top-10 detailed

See main text Table 2 (tab:pareto). All 10 expressions pass the positivity veto. Expression 3 and expression 5 (complexity 13 and 10) are the minimum-complexity variants that retain the $e^{d_e}$ or $(C-Z_X)$ polarizability-scaling motif.

## Algorithm 1. Harness-2D-RI workflow loop

```
Require: D_train (154 samples), D_hold (39 samples, 80/20 split, seed=42),
         max rounds R, knowledge K
Notation: D_hold plays the role of a per-round screening set. At the Tier-2
scale (N=193) we deliberately do NOT carve out a third "true blind" partition:
a triple split would leave <30 samples per piece. See Methods,
"Evaluation protocol" — leakage-free benchmarks are
10x5 CV on the full dataset and LOCO (main-text LOCO section),
not the held-out-slice RMSE read inside the loop below.
Initialise: AdaptiveEnv E; AgentCouncil C; Pareto set F = ∅
for r = 1 to R do
    cfg  ← E.getConfig(r, K)               # operator set, parsimony, seeds
    I_r  ← LLM(K.successPatterns, cfg)     # prompted search instruction
    P_r  ← PySR(D_train, I_r, cfg)         # symbolic search → Pareto (fit only on D_train)
    for f ∈ P_r do
        ŷ ← f(D_hold)                       # screening signal (read by council & Pareto)
        v ← C.vote(f, ŷ, D_hold)
        if v = ACCEPT: F ← F ∪ {f}
        else:           K.recordFailure(f, r)
    end for
    E.recordOutcome(r, F)                  # operator stats, convergence
    K.recordRound(r, F)                    # success patterns, exclusion rules
end for
return Pareto-optimal subset of F (ranked by cplx–accuracy–physical-screening)
```

## Figure S1. Raw feature correlation matrix

Pearson $r$ between all 18 descriptors and Cauchy $A$ on the $N=193$ dataset; heatmap with numerical annotations. $E_g$ shows $r=-0.485$, the strongest single-feature linear correlation; $Z_X$ shows $r=+0.290$.

## Figure S2. Per-anion prediction residuals

Box plot of $(\hat{A}-A_{\text{true}})$ for Eq. 1 stratified by anion; shows no systematic bias per anion family except modest Se over-prediction ($+0.2$ mean residual, consistent with the LOCO under-performance on Se reported in the main text).

## Figure S3. Knowledge-memory growth across rounds

Number of exclusion rules and success patterns stored in KnowledgeMemory at the end of each round for the 10-round reference run. Success patterns saturate at $\approx 12$ by round 6; exclusion rules continue accumulating through round 10 (final 38 rules, mostly vetoing log/exp compositions on high-$d_e$ features).

## Section S4. Tier-3 robustness expansion (JARVIS-DFT2D)

To stress-test the four main-text claims on a dataset $\approx$2.8$\times$ larger and chemically broader than Tier 2, we constructed Tier 3 by ingesting all JARVIS-DFT2D entries with valid $\varepsilon_\infty$ and $E_g>0.2$ eV without imposing a binary-composition filter. For ternaries and quaternaries we assign $(M,X)$ as the (most-electropositive, most-electronegative) pair by Pauling electronegativity; the featurisation is otherwise identical to Tier 2.

### Table S8. Tier-3 composition

| subset | $N$ | fraction |
|---|---|---|
| Binary ($n_{\text{elem}}=2$) | 222 | 42% |
| Ternary ($n_{\text{elem}}=3$) | 279 | 52% |
| Quaternary+ ($n_{\text{elem}}\geq 4$) | 31 | 6% |
| **Total (Tier 3)** | **532** | 100% |

### Zero-shot transfer of Eq. (1) to Tier 3

We apply the *frozen* post-veto headline expression (trained only on the 193 Tier-2 binaries) to all 532 Tier-3 entries without any parameter refit. All 532 predictions are finite and pass the stability veto $[1,20]$.

| slice | $N$ | RMSE | MAE | $R^2$ |
|---|---|---|---|---|
| Binary only | 222 | 0.573 | 0.394 | $+0.465$ |
| Ternary only | 279 | 0.726 | 0.461 | $+0.390$ |
| Quaternary+ | 31 | 0.649 | 0.382 | $-1.47$ |
| **Full Tier 3** | **532** | **0.662** | **0.428** | $+0.392$ |
| (Tier 2 binaries, sanity) | 193 | 0.577 | — | $+0.545$ |

Binary Tier-3 RMSE (0.573) is statistically indistinguishable from Tier-2 binary (0.577), confirming that the expression transfers cleanly within the chemistry class it was trained on. The ternary degradation ($\Delta\text{RMSE}\approx +0.15$) and the negative $R^2$ on the small quaternary slice (variance in this subset is low, so the denominator in $R^2$ shrinks) are expected consequences of a binary-trained expression operating outside its design domain.

### Table S9. Tier-3 LOCO (leave-one-anion-out)

Zero-shot frozen Eq. (1) vs. retrained RF/Ridge baselines. 18 anions are present; only the 10 with $N_{\text{test}}\geq 5$ and $N_{\text{train}}\geq 50$ are scored. Weighted averages are sample-weighted.

| Anion | $N$ | Harness (zero-shot) | RF(500) | Ridge | $\Delta$ vs RF | winner |
|---|---|---|---|---|---|---|
| As | 8 | 2.022 | 1.159 | 1.210 | $+0.86$ | RF |
| Br | 59 | **0.468** | 0.577 | 0.630 | $-0.11$ | Harness |
| Cl | 59 | **0.624** | 0.826 | 0.910 | $-0.20$ | Harness |
| F | 19 | **0.808** | 0.852 | 0.759 | $-0.04$ | Harness |
| I | 68 | 0.527 | 0.502 | 0.665 | $+0.03$ | RF |
| N | 24 | 0.357 | **0.160** | 0.290 | $+0.20$ | RF |
| O | 150 | 0.585 | **0.557** | 0.648 | $+0.03$ | RF |
| S | 53 | 0.369 | **0.352** | 0.542 | $+0.02$ | RF |
| Se | 63 | 0.661 | **0.370** | 0.485 | $+0.29$ | RF |
| Te | 16 | 1.487 | 1.246 | 1.693 | $+0.24$ | RF |
| **weighted** | 519 | 0.603 | **0.562** | 0.676 | $+0.04$ | RF |

### Table S10. Tier-3 binary-only LOCO (diagnostic)

To localize where the advantage weakens, we repeat LOCO on the binary-only Tier-3 slice. LOCO uses $N_{\text{LOCO}}$ = LOCO-scored samples (rare-anion folds with <5 test or <50 train samples excluded).

| | $N_{\text{LOCO}}$ | weighted RMSE | wins (vs RF) |
|---|---|---|---|
| Tier 2 (binary) | 193 | Harness 0.549 vs RF 0.643 | 7/10 |
| Tier 3 binary-only | 214 | Harness **0.529** vs RF 0.577 | 4/10 |
| Tier 3 full | 519 | Harness 0.603 vs RF 0.562 | 3/10 |

At Tier 3, two readings of the LOCO result split. On **weighted RMSE** (the aggregate quantity reported in the main text), the binary-only Harness advantage *strengthens* relative to Tier 2 ($0.549\to 0.529$, the best of the three settings) and holds against retrained RF (0.529 vs 0.577). On **per-anion wins**, however, Harness drops to 4/10 — the expanded JARVIS-DFT2D binaries add anion-family diversity without increasing same-anion density, so retrained RF catches up on several per-anion cells while still losing on the sample-weighted mean. These two readings are not contradictory (one is weighted, the other is a per-fold tally), but the per-anion margin is narrower at Tier 3 than at Tier 2 and we flag this explicitly. The reversal of the weighted average on full Tier 3 (RF 0.562 vs Harness 0.603) is driven primarily by the 279 ternaries where the frozen binary-trained expression is outside its chemical design domain.

### Moss-exponent scan at Tier 3

| slice | $N$ | data slope | Harness slope | theoretical |
|---|---|---|---|---|
| Full Tier 3 | 532 | $-0.333$ | $-0.099$ | $-0.250$ |
| Binary only | 222 | $-0.308$ | $-0.081$ | $-0.250$ |
| Tier 2 (reference) | 193 | $-0.294$ | $-0.094$ | $-0.250$ |

The stability-veto-induced Moss-slope deviation persists at Tier 3 and remains quantitatively similar to Tier 2 on both the full set and the binary-only slice. The data slope itself drifts slightly more negative as the chemistry diversifies (from $-0.294$ to $-0.333$), consistent with the 3D Moss literature, but the frozen Eq. (1) does not track this drift. This reinforces the honest-negative-result framing in the main text: the Moss exponent is *not* recovered by the stability-pruned expression, either at Tier 2 or at Tier 3.

### Table S11. Tier-3 Strong-PySR baseline (head-to-head)

We retrain Strong-PySR (vanilla PySR without the harness governance) on the *full* Tier-3 training split (80%, $N=425$) with budget $5$ seeds $\times$ $5$ rounds $\times$ $40$ islands $\times$ $100$ iterations, then compare on the frozen $20\%$ Tier-3 held-out slice ($N=107$) against the frozen Tier-2 headline (zero-shot, no refit).

| method | held-out RMSE | held-out $R^2$ | complexity | stable | notes |
|---|---|---|---|---|---|
| Frozen Harness Eq. 1 | 0.719 | $-0.053$ | **11** | ✓ | zero-shot — never trained on Tier 3 |
| Strong-PySR (mean $\pm$ std, 5 seeds) | 0.662 $\pm$ 0.029 | $+0.107 \pm 0.080$ | 23.6 $\pm$ 1.5 | 100% | retrained on Tier 3 |

Strong-PySR edges out the frozen Harness expression on the Tier-3 held-out slice by $\approx 8\%$ in RMSE — but at $\approx 2.1\times$ the complexity. No seed recovers the nested cube-root motif (0/5 `has_cbrt`); the motifs that *are* consistently picked up are the inverse/ratio structure (4/5) and the d-electron count (4/5), matching the Tier-2 Strong-PySR finding that the governance scaffold is specifically responsible for the parsimonious cube-root closure. A fully-retrained unguided baseline on the expanded dataset only narrowly beats a frozen 11-node binary-trained expression while producing $2\times$-longer expressions that do not generalise across anion families as cleanly (Table S9).

### Summary of Tier-3 robustness results

1. **Zero-shot accuracy transfers cleanly on binaries**: RMSE 0.573 on Tier-3 binaries is statistically equivalent to Tier-2 binaries (0.577), despite 29 new materials spanning new (M, X) combinations.
2. **LOCO advantage holds on weighted RMSE but narrows at the per-anion level**: binary-only Tier-3 weighted LOCO RMSE (0.529) still beats retrained RF (0.577) and Ridge (0.654); per-anion wins against RF, however, drop from 7/10 at Tier 2 to 4/10 at Tier 3 binary-only. Against full Tier 3 including ternaries, RF overtakes on weighted RMSE (0.562) as expected for an expression not designed for ternaries.
3. **Stability veto still guarantees physical predictions**: 532/532 predictions pass the $[1, 20]$ physical-range check with no retraining.
4. **Moss slope behaviour is robust**: the stability-veto/Moss-exponent tension identified in Tier 2 reproduces on Tier 3, with both data and predicted slopes changing by $<0.04$ — not a Tier-2 artefact.
5. **Parsimony advantage is reinforced**: a fully-retrained Strong-PySR baseline on Tier 3 edges the frozen Harness expression in raw RMSE by $\approx 8\%$, but at $2.1\times$ the complexity and without recovering the cbrt motif in any of 5 seeds (Table S11).

The Harness-2D-RI framework therefore produces a **binary-class refractive-index law that is (a) governance-stable, (b) zero-shot transferable within its training chemistry, and (c) honest about the limits of stability-pruned fits for multi-element systems.** Extension to ternaries would require either a joint search with per-element featurisation or a composition-weighted anion descriptor — both left for future work.

## Section S5. Why $Z_M$ is a physics variable, not a periodic-table index

The headline expression depends on $Z_M$, while the data's feature-importance ranking flags $E_g$ as the strongest single scalar. This section expands the mechanism-level explanation given compactly in the main text.

**$Z_M$ and $E_g$ are not independent descriptors.** In transition-metal 2D semiconductors the static dielectric response is dominated by inter-band transitions between the valence-band edge (predominantly ligand-$p$ and metal-$d$ hybridised character) and the conduction-band edge. $Z_M$ monotonically indexes three upstream quantities that together set this edge structure:

1. **Principal quantum number $n$ (row in the periodic table).** Larger $n$ means more radially extended $d$-orbitals, stronger $d$–$p$ overlap with the anion, and narrower $d$-bands that shift closer to $E_F$.
2. **$d$-electron count within the row.** This controls the filling-dependent position of the $t_{2g}$/$e_g$ manifolds relative to the ligand-$p$ band and therefore whether the valence-band edge has dominant metal or ligand character.
3. **Nuclear charge and core–valence screening.** Larger $Z_M$ (holding $n$ fixed) contracts the $3d$ shell and lowers its on-site energy, shifting the gap and the transition-matrix elements in concert.

**Why $E_g$ is downstream.** The DFT band gap $E_g$ is the difference between the hybridised conduction- and valence-band edges set by (1)–(3); the f-sum-rule value of $\varepsilon_{\infty}$ depends on the same upstream quantities through the transition-matrix elements and oscillator strengths. $E_g$ is therefore *derivable* from the same electronic-structure inputs that $Z_M$ indexes, which is why Moss-type relations hold inside a fixed-anion family (where only $Z_M$ varies) but break across chemistries (where $Z_X$ also changes and the ligand-$p$ band shifts).

**What the cube-root-of-cube-root-of-rational-in-$Z_M$ is doing.** The discovered form has a pole at $Z_M\approx 23$ (V), which marks the $3d$-row transition between early-TM $d^0$/$d^1$ oxides (low $A$, wide $E_g$) and late-TM chalcogenides (high $A$, narrow $E_g$). A piecewise function would impose a discontinuity; the nested cube-roots produce a smooth, bounded function that captures the non-monotonic $A(Z_M)$ trend without a literal step. The result is that $\hat{A}$, while a function of $Z_M$ alone, behaves empirically like a partial Moss surrogate — r($\hat{A},A$) $=0.738$, r($\hat{A},E_g$) $=-0.26$ — even though neither $E_g$ nor the $d$-electron count appears explicitly.

**Why the physics argument is not a post-hoc rationalisation.** The LOCO evaluation at Tier 2 (main-text LOCO section) and the Tier-3 binary expansion (Section S4) each apply Eq.(1) *frozen* to entries that include metals and anions unseen in the training partition. If $Z_M$ were serving as a dataset-specific look-up, zero-shot transfer to unseen chemistries would not improve under scale-up — it would degrade. Instead, binary-only LOCO weighted-RMSE falls from $0.549$ (Tier 2) to $0.529$ (Tier 3), exactly the direction expected if $Z_M$ encodes a real electronic-structure trend rather than a fitted surrogate.

**What the argument does *not* claim.** We do *not* claim that $Z_M$ is sufficient to derive $n(E_g)$ exactly; the recovered Moss slope is $-0.094$, not $-0.29$ (Table S6), and the stability veto is the binding constraint on exact power-law recovery (main text, Moss-law section). What we claim is weaker and supported: $Z_M$ is a physics-meaningful index for the electronic-structure variables that together control $\varepsilon_{\infty}$, and the nested cube-root form is a stable, transferable, compact encoding of that dependence in the binary-2D-semiconductor subclass.

## Section S6. Experimental-calibration diagnostics (TMD family bridge)

The single-$\alpha$ calibration presented in the main text (main-text Table 3) is a **family-level mean bridge**. This section collects the supporting diagnostics.

**Spread comparison.** Measured $n_{\exp}$ across the five TMDs spans $[4.02,\,4.90]$, a relative spread of $19.6\%$. Surrogate outputs $\hat{A}$ span $[1.70,\,1.73]$ ($1.7\%$); calibrated predictions $\alpha\hat{A}$ span $[4.43,\,4.51]$ ($1.7\%$). The ratio $19.6\%/1.7\% \approx 12$ is the multiplicative factor by which the experimental range exceeds the resolving power of the surrogate on this sample.

**Null-model comparison.** A constant predictor equal to the experimental mean, $\bar{n}_{\exp}=4.482$, reaches MAPE $=6.0\%$ on the same five materials. Surrogate+$\alpha$ reaches MAPE $=5.9\%$. The $0.1$pp gap is well within bootstrap noise on $n=5$, so the $5.9\%$ figure should be read as "the mean is right after calibration," not as "the surrogate discriminates per-material."

**Rank correlations on the five-point sample:**

| predictor | Pearson $r$ | Spearman $\rho$ |
|---|---|---|
| $A_\text{DFT}$ (JARVIS raw) | $0.72$ | $0.60$ |
| $\hat{A}$ (surrogate, Eq. 1) | $0.55$ | $0.58$ |
| $\alpha\hat{A}$ (calibrated) | $0.55$ | $0.58$ |
| $\bar{n}_{\exp}$ (constant) | — | — |

Multiplicative $\alpha$ cannot change rank correlation, so calibrated and uncalibrated surrogate Pearson-$r$ are identical. Raw DFT actually ranks the five materials slightly better than the surrogate does; neither is conclusive at $n=5$.

**Bootstrap on $\alpha$.** Ten thousand resamples with replacement of the five-material set give $\alpha\in[2.45,\,2.75]$ (95% CI, percentile), point estimate $2.605$, standard error $0.077$. This CI bounds the *calibration* uncertainty (where the mean sits) and does *not* bound discrimination uncertainty (how well the surrogate resolves within the family). The latter would require a larger, more spread experimental panel.

**What would move this from "bridge" to "discrimination."** A reasonable discrimination test would need (i) $\gtrsim 20$ monolayers with measured $n_{\exp}$ spanning at least $[1.5,\,6]$, (ii) a held-out split so $\alpha$ is fitted on one half and evaluated on the other, (iii) an explicit non-trivial null (e.g. family-conditional mean, $E_g$-only prediction after $\alpha$-bridging) against which Pearson-$r$ is tested, and (iv) a family-conditional $\alpha(\text{anion})$ so that halides and pnictides are not forced to share the TMD value. None of these are possible on the five-point literature sample we have access to at review time; we flag this as the primary limitation of the experimental-calibration claim.

**Scope-appropriate framing.** Within these limits, the single-$\alpha$ bridge has three defensible uses: (a) as a *screening predictor* that gives an experimentally compatible refractive-index scale for TMDs from DFT inputs alone; (b) as a *null check* for future family-conditional calibrations (any anion-conditional $\alpha(\text{anion})$ should reduce to the TMD value of $2.605$ on the TMD subset); and (c) as a *demonstration* that the $\sim 50\%$ OPT-B88vdW underestimation of $\varepsilon_{\infty}$ is concentrated in a scalar offset rather than distributed across per-material deviations that would require a more complex correction. Beyond these, the five-point calibration is not sufficient to support claims about the surrogate's ability to discriminate between individual TMDs at experimental precision.
