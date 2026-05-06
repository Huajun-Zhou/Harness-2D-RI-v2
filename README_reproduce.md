# Reproducing the manuscript results

This file lists the exact commands needed to reproduce **Table 1**, **Figures 2–5**, **Figure S1**, and **Tables S2–S6** reported in the accompanying paper.

## Repository layout

| Path | Contents |
|---|---|
| `data/extended/tier2_binary.csv` | Tier-2 dataset, $N=193$ binary 2D semiconductors (the only data file the manuscript depends on). |
| `experiments/E1_full_harness_fixed_council/runs/` | E1 reference run on Tier-2 fixed split (precursor; cited only via Supplementary §S5). |
| `experiments/E2_outer_loco/runs/` | E2 outer-LOCO runs: ten `fold_<anion>_seed42/` directories for the no-LLM heuristic stub and ten `fold_<anion>_seed42_qwen/` directories for the LLM-guided harness. Each contains `fold_summary.json` and `manifest.json`. |
| `experiments/E3_matched_pysr_baselines/runs/` | E3 budget-matched vanilla PySR runs, ten folds × five seeds × four selection policies. |
| `experiments/E4_ablation_fixed_council/runs/E4_seed42_3seeds/ablation_summary.json` | E4 leave-one-module ablation, three seeds × five variants. |
| `experiments/E5_diagnostics/runs/zm_only_seed42/` | E5.1 single-feature diagnostic. |
| `experiments/E5_diagnostics/runs/lomo_seed42/` | E5.2 leave-one-metal-out diagnostic. |
| `experiments/E5_diagnostics/runs/target_permute_O_seed42/` | E5.3 permutation null test. |
| `experiments/E5_diagnostics/runs/threshold_scan_I_seed42/` | Supplementary §S2.2 threshold sensitivity scan. |
| `experiments/E5_diagnostics/runs/eq1_frozen_eval/` | Supplementary §S5 precursor counter-example. |
| `experiments/E5_diagnostics/E5_combined_summary.json` | Aggregator output combining E5.1, E5.2, E5.3. |
| `configs/harness_config.yaml` | Default harness configuration consumed by the LLM-guided runs. |
| `scripts/make_v2_figures.py` | Generates Figures 1–5 and Figure S1 from artifacts. |
| `scripts/aggregate_E2.py` … `aggregate_E5.py` | Per-experiment aggregators (run after each campaign). |
| `scripts/threshold_sensitivity_scan.py` | Produces the §S2.2 scan artifact. |
| `scripts/eval_eq1_outer_loco.py` | Produces the §S5 precursor frozen-evaluation artifact. |
| `harness_runner.py` | Top-level dispatcher used by all five experiment campaigns. |

## Environment

The pinned Python environment used to produce the manuscript artifacts is `requirements.txt`. Install with:

```
pip install -r requirements.txt
```

Key versions: Python 3.14, PySR v1.5.10 (Julia backend), NumPy 2.4.2, pandas 3.0.0. PySR's Julia backend builds on first run (~2–5 minutes).

The LLM-guided E2 runs additionally require an LLM API key. The harness reads the provider from environment variables; the runs reported in the manuscript use Qwen-max via DashScope:

```
export LLM_PROVIDER=qwen
export DASHSCOPE_API_KEY=<your key>
export LLM_TIMEOUT_S=90
export LLM_RETRIES=1
```

The `heuristic-stub` control needs no API key; setting `LLM_PROVIDER=heuristic` (or omitting the API key) routes the proposal through a feature-correlation ranking instead.

## Provenance and integrity

Each run artifact is written together with `manifest.json` recording:

- the git commit (current head: `9dfc415`),
- SHA-256 hashes of the dataset files read,
- the package versions present at run time,
- a hash of the configuration consumed.

The manifest is written *before* any computation and finalised on successful completion, so any artifact in the repository whose manifest lacks an end-time is from an incomplete run and should not be trusted.

## Reproducing main-text claims

### Table 1 (sample-weighted outer-LOCO RMSE)

The five rows of Table 1 are sample-weighted across the ten anion folds. Re-aggregate from existing artifacts with:

```
python3 scripts/aggregate_E2.py
python3 scripts/aggregate_E3.py
```

Each script prints the weighted RMSE of every method to stdout and writes the per-anion table referenced as Supplementary Table S4.

To regenerate the per-fold artifacts from scratch (one full E2 + E3 pass), run:

```
# E2 — outer-LOCO with LLM-guided harness (Qwen-max) on all ten anions
bash scripts/run_E2_qwen_all_folds.sh

# E2 — outer-LOCO with heuristic-stub control
for anion in O S Se Te F Cl Br I N P; do
  python3 harness_runner.py --experiment E2_outer_loco \
      --config experiments/E2_outer_loco/configs/fold_${anion}.yaml \
      --run-seed 42 --run-id fold_${anion}_seed42
done

# E3 — budget-matched vanilla PySR baselines, four selection policies, five seeds
for anion in O S Se Te F Cl Br I N P; do
  python3 harness_runner.py --experiment E3_matched_pysr_baselines \
      --config experiments/E3_matched_pysr_baselines/configs/fold_${anion}.yaml \
      --run-seed 42 --run-id fold_${anion}_seed42
done
```

Total compute on a single laptop CPU: ~12 hours wall-clock for E2-Qwen, ~4 hours for E2-heuristic, ~2 hours for E3 (vanilla PySR is fast).

### Figure 1 (architecture schematic)

```
PYTHONPATH=. python3 scripts/make_v2_figures.py fig1_architecture
```

Output: `figures/fig1_architecture.{pdf,png}`. The figure is a static layout, no data is read from disk.

### Figure 2 (Tier-2 dataset overview)

```
PYTHONPATH=. python3 scripts/make_v2_figures.py fig2_dataset
```

Reads `data/extended/tier2_binary.csv`. Output: `figures/fig2_dataset.{pdf,png}`.

### Figure 3 (outer-LOCO head-to-head, Table 1 visual)

```
PYTHONPATH=. python3 scripts/make_v2_figures.py fig3_outer_loco
```

Reads ten `experiments/E2_outer_loco/runs/fold_<anion>_seed42_qwen/fold_summary.json` and ten `experiments/E2_outer_loco/runs/fold_<anion>_seed42/fold_summary.json` (heuristic), and ten `experiments/E3_matched_pysr_baselines/runs/fold_<anion>_seed42/fold_summary.json` (V2$_\text{screen}$ baseline). The script prints the per-condition weighted RMSE before saving, allowing direct comparison with Table 1.

### Figure 4 (leave-one-module ablation)

```
PYTHONPATH=. python3 scripts/make_v2_figures.py fig4_ablation
```

Reads `experiments/E4_ablation_fixed_council/runs/E4_seed42_3seeds/ablation_summary.json`. To regenerate that artifact:

```
python3 harness_runner.py --experiment E4_ablation_fixed_council \
    --config experiments/E4_ablation_fixed_council/configs/E4_default.yaml \
    --run-seed 42 --run-id E4_seed42_3seeds
python3 scripts/aggregate_E4.py
```

### Figure 5 (E5 diagnostics, three panels)

```
PYTHONPATH=. python3 scripts/make_v2_figures.py fig5_diagnostics
```

Reads `experiments/E5_diagnostics/E5_combined_summary.json`. To regenerate that artifact:

```
python3 harness_runner.py --experiment E5_diagnostics --config experiments/E5_diagnostics/configs/zm_only.yaml       --run-seed 42 --run-id zm_only_seed42
python3 harness_runner.py --experiment E5_diagnostics --config experiments/E5_diagnostics/configs/lomo.yaml          --run-seed 42 --run-id lomo_seed42
python3 harness_runner.py --experiment E5_diagnostics --config experiments/E5_diagnostics/configs/target_permute.yaml --run-seed 42 --run-id target_permute_O_seed42
python3 scripts/aggregate_E5.py
```

### Figure S1 (per-fold parity plots)

```
PYTHONPATH=. python3 scripts/make_v2_figures.py fig6_parity
```

Reads the same E2-Qwen artifacts as Figure 3 plus `data/extended/tier2_binary.csv` for the held-out predictions. Output is currently named `fig6_parity.{pdf,png}` for legacy reasons; embed it as Figure S1 in the supplementary file.

### Supplementary Table S2 (ablation summary)

The five-row table is generated in the same step as Figure 4; values are read from `experiments/E4_ablation_fixed_council/runs/E4_seed42_3seeds/ablation_summary.json`.

### Supplementary §S2.2 (threshold sensitivity scan)

```
PYTHONPATH=. python3 scripts/threshold_sensitivity_scan.py
```

Output: `experiments/E5_diagnostics/runs/threshold_scan_I_seed42/fold_summary.json`. Contains the five-condition table reported in the supplementary as Supplementary Table S2.2.

### Supplementary Table S6 (precursor counter-example)

```
PYTHONPATH=. python3 scripts/eval_eq1_outer_loco.py
```

Output: `experiments/E5_diagnostics/runs/eq1_frozen_eval/fold_summary.json`. Contains the per-anion frozen-evaluation RMSE referenced in §S5.

## Quick verification — every reported number traceable to an artifact

The 21 numerical claims of the manuscript are verified against the on-disk artifacts by:

```
PYTHONPATH=. python3 scripts/verify_v3_numbers.py     # included in the release
```

The script returns exit code 0 with `21/21 OK, 0 FAILED` on a clean checkout at commit `9dfc415`.

## Notes for reviewers

* Run seed `42` is used throughout. Per-fold seeds (LLM, PySR, inner-split) are derived deterministically by SHA-256 hashing of `(run_seed, anion, ...)`. Re-running with `--run-seed 42` reproduces the artifact files bit-for-bit on the same machine; PySR's parallel search has small platform-level non-determinism, so a different machine will reproduce the headline numbers to within the per-seed standard deviations reported in the ablation (§4.3).
* PySR's search is occasionally non-terminating on degenerate input. The diagnostics (§4.4) protect against this with a per-round wall-clock cap (60 seconds for E5.2 LOMO, 90 seconds elsewhere) and a single retry; the artifact log files record any timeout-driven round skips.
* All anonymised URLs in the Data availability and Code availability sections are placeholders; the actual repository URL will be substituted at acceptance.
