# Harness-2D-RI

**LLM-guided harness engineering for small-data symbolic regression, applied
to the refractive response of two-dimensional semiconductors.**

This repository is the code release accompanying the paper *LLM-guided
harness engineering stabilizes symbolic regression for optical modeling in
2D semiconductors*. It contains the harness implementation, the four-agent
governance council, the configuration files for all reported experiments
(E1–E5), the per-fold artefacts that the figures and tables of the paper
were rendered from, and the analysis scripts.

## What this repository contains

A four-layer governance scaffold wrapped around an unmodified PySR symbolic
regression engine:

| Layer | Module | Purpose |
| --- | --- | --- |
| Data & featurisation | `src/data_pipeline.py`, `src/real_data_loader.py` | Load JARVIS-DFT2D entries and compute the 20-feature descriptor table |
| Symbolic search | `src/llm_sr_engine.py` (Python) + PySR (Julia) | LLM-proposed feature/operator/skeleton instructions feeding a population-based PySR search |
| Governance | `agents/` (Data, Theory, Validator, Meta) + `src/causal_diagnostic_engine.py` + `src/knowledge_memory.py` + `src/adaptive_search_environment.py` | The four governance modules ablated in the paper |
| Orchestration | `src/harness_orchestrator.py`, `harness_runner.py` | Per-round loop, single-VETO gate, cross-round memory injection |

The empirical claim is **not** "we discovered a refractive-index law for 2D
materials". It is: *under matched compute and a leakage-free outer-LOCO
protocol, an LLM-guided governed search produces smaller, more stable, and
more transferable expressions than budget-matched vanilla PySR.*
Sample-weighted across ten anion folds, the harness attains an outer-LOCO
RMSE of **0.6450** versus **0.7415** for the matched vanilla baseline
(Δ = −0.097, 13.1% relative; harness wins on 7 of 10 folds).

## Repository layout

```
Harness-2DTMDs/
├── agents/                4-agent council (Data, Theory, Validator, Meta)
├── configs/               Default harness config consumed by all experiments
├── data/extended/         Tier-2 dataset (tier2_binary.csv, N = 193)
├── experiments/           E1–E5 campaigns; see experiments/README.md
│   ├── E1_full_harness_fixed_council/
│   ├── E2_outer_loco/
│   ├── E3_matched_pysr_baselines/
│   ├── E4_ablation_fixed_council/
│   └── E5_diagnostics/
├── legacy_v1/             Read-only archive of the v1 precursor analysis
├── scripts/               Aggregators, figure renderers, configuration generators
├── src/                   Core modules
│   └── v2/                Run infrastructure: seed derivation, manifests, LLM provider
├── harness_runner.py      Single CLI entry point for all experiments
├── main.py                Reference one-fold harness invocation
├── requirements.txt       Pinned dependency versions
├── README.md              (this file)
└── README_reproduce.md    Step-by-step commands to reproduce every figure and table
```

## Requirements

- Python 3.14
- Julia 1.10+ with PySR v1.5.10 (built automatically on first PySR call;
  initial build takes 2–5 minutes)
- An LLM API key for the harness in its LLM-guided mode:
  - Anthropic Claude (default; set `ANTHROPIC_API_KEY`), or
  - Alibaba Qwen via DashScope (set `LLM_PROVIDER=qwen` and
    `DASHSCOPE_API_KEY`); the LLM-guided runs reported in the paper used
    Qwen-max.
- The heuristic-stub control and all vanilla-PySR baselines (E3) require
  no API key.

```bash
pip install -r requirements.txt
```

### LLM API keys

The harness loads provider keys exclusively from environment variables; no
key is ever read from disk, written to a manifest, logged to stdout, or
serialised into any artefact under `experiments/E*/runs/`. To run the
LLM-guided harness, set the keys in your shell session (or a local
`.env` file that is already covered by `.gitignore`):

```bash
# Option A — Anthropic Claude (default provider)
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=<your key>

# Option B — Alibaba Qwen via DashScope (the provider used in the paper)
export LLM_PROVIDER=qwen
export DASHSCOPE_API_KEY=<your key>
export LLM_TIMEOUT_S=90
export LLM_RETRIES=1
```

Never commit a key. The `.gitignore` blocks `.env`, `.env.*`, `**/api_keys*`,
and `**/secrets*` by default; the run-manifest writer in
`src/v2/manifest.py` explicitly redacts the API-key environment variables
from the recorded environment snapshot.

## Quick start

```bash
# 1. Verify the four-agent council is wired up correctly
python3 scripts/smoke_test_council.py
#   Expect: "RESULT: PASS — 4-agent council voted cleanly on round-1 pre-search."

# 2. Run one outer-LOCO fold (here: the fluorine fold) with the LLM-guided harness
export LLM_PROVIDER=qwen
export DASHSCOPE_API_KEY=<your key>
python3 harness_runner.py \
    --experiment E2_outer_loco \
    --config experiments/E2_outer_loco/configs/fold_F.yaml \
    --run-seed 42 \
    --run-id fold_F_seed42_qwen
```

A complete recipe for reproducing every number, table, and figure in the
paper is in [`README_reproduce.md`](README_reproduce.md).

## Reproducibility contract

Every result reported in the paper is derived from an artefact directory at
`experiments/E*/runs/<id>/` containing:

- `manifest.json` — run seed, git commit hash, configuration hash, dataset
  hashes, package versions, and start/end timestamps (written by
  `src/v2/manifest.py`); a manifest without an end timestamp marks an
  incomplete run and is excluded from aggregation.
- `fold_summary.json` — the headline per-fold metrics and the headline
  expression.
- The raw Pareto fronts and per-round logs that the figures consume.

Per-fold seeds for the LLM call, the PySR search, and the inner train/
screen split are derived deterministically by SHA-256 hashing of
`(run_seed, anion, component)`, so re-running with `--run-seed 42`
reproduces the artefact files bit-for-bit on the same machine. PySR's
parallel search has small platform-level non-determinism, so a different
machine reproduces the headline numbers to within the per-seed standard
deviations reported in the ablation (paper §4.3).

## v1 precursor

`legacy_v1/` archives the precursor analysis that motivated the redesign.
Three substantive issues identified in that earlier evidence drove the
present work:

1. The v1 "LOCO" evaluation applied a frozen expression — already trained
   on all anions — anion-stratified; it was not a leakage-free discovery
   test.
2. The Strong-PySR baseline selected on training RMSE while the harness
   selected on a screening slice, so the head-to-head compared selection
   policies, not governance.
3. The four-agent council was advertised but only three agents voted at
   runtime; `DataAgent._vote_feature_selection` raised `NameError: name
   'context' is not defined` on every round (fixed in the present code).

The current paper addresses all three by enforcing a true outer-LOCO
protocol, by introducing budget-matched vanilla-PySR baselines under
identical inputs and selection rules, and by reframing the contribution
around the harness scaffold rather than any specific output expression.

## Data availability

All data needed to reproduce the paper are tracked in this repository so
that a fresh `git clone` is sufficient — no external download is required.

- **`data/extended/tier2_binary.csv`** (N = 193) is the Tier-2 dataset that
  every result in the paper (Table 1, Figs 2–5, Table S2–S6) is computed
  from. This is the only file the manuscript depends on.
- **`data/extended/jarvis_dft2d_raw.json`** (~24 MB) and
  **`data/extended/c2db_raw.json`** (~13 MB) are the raw upstream snapshots
  from JARVIS-DFT2D (NIST) and C2DB. They are kept in-repo so that the
  Tier-2 build pipeline (`scripts/build_extended_dataset.py`) is end-to-end
  reproducible from the same commit, without depending on the upstream
  databases continuing to be reachable or unchanged.
- **`data/extended/tier1_extended.json`**, **`tier1_tmd.json`**,
  **`tier3_all.json`**, **`tier2_binary.json`** are intermediate or
  precursor tables retained for provenance; they are not consumed by any
  result in the paper but document the construction pipeline.

GitHub displays a soft-warning banner on files above 50 MB; none of the
files in this repository exceed that threshold.

## Citation

If you use this code or the Tier-2 dataset, please cite the accompanying
paper. A BibTeX entry will be added on acceptance.

## License

Code: MIT. Data derived from JARVIS-DFT (public domain, courtesy NIST) and
C2DB (licensed under their respective terms).
