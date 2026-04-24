# Harness-2D-RI

**Harness engineering for small-data symbolic regression, applied to the
refractive response of 2D semiconductors.**

Working title (v2, in progress):
*Harness engineering stabilises symbolic regression for small-data optical
modelling of 2D semiconductors.*

## What this repository contains

A four-layer "harness" wrapped around a PySR symbolic-regression engine:

| Layer | Module | Purpose |
| --- | --- | --- |
| Data & featurisation | `src/data_pipeline.py`, `src/real_data_loader.py` | Load JARVIS-DFT2D entries, compute 17/20-feature descriptors |
| Symbolic search | `src/llm_sr_engine.py` (Python) + PySR (Julia) | Multi-island genetic programming with LLM-proposed configs |
| Governance | `agents/` (Data, Theory, Validator, Meta) + `src/causal_diagnostic_engine.py` + `src/knowledge_memory.py` + `src/adaptive_search_environment.py` | The four governance modules the paper ablates |
| Orchestration | `src/harness_orchestrator.py`, `harness_runner.py` | Round loop, veto gates, cross-round memory injection |

The claim the v2 paper will defend is **not** "we found a refractive-index
law for 2D materials". The claim is: *under matched compute and nested
evaluation, a governed symbolic search produces smaller, more stable, and
more transferable expressions than an unguided one.*

## Repository layout (v2)

```
Harness-2DTMDs/
├── agents/                 4-agent council (DataAgent, TheoryAgent, ValidatorAgent, MetaAgent)
├── configs/                Harness + experiment configs
├── data/
│   ├── raw/                JARVIS-DFT2D snapshots
│   └── extended/           Processed feature tables (tier2_binary.csv, tier3_binary_plus_ternary.csv)
├── experiments/            v2 campaigns (E1–E5). See experiments/README.md
├── legacy_v1/              Read-only archive of the v1 manuscript's artefacts
├── newpaper/               Manuscript and SI sources (in flight for v2 rewrite)
├── scripts/                Reusable dataset / featurisation / baseline utilities
├── src/                    Core modules
│   └── v2/                 v2-specific infra: seed derivation, run manifests
├── harness_runner.py       Single CLI entry point for all v2 experiments
├── requirements.txt        Pinned dependency versions
└── README.md               (this file)
```

## Getting started

```bash
# 1. Python 3.14, Julia 1.10+, with PySR preinstalled
pip install -r requirements.txt

# 2. Verify the 4-agent council wires up correctly (no DataAgent exception)
python3 scripts/smoke_test_council.py
# Expect: "RESULT: PASS — 4-agent council voted cleanly on round-1 pre-search."

# 3. Phase 2 experiments will be driven by
python3 harness_runner.py \
    --experiment E2_outer_loco \
    --config experiments/E2_outer_loco/configs/fold_O.yaml \
    --run-seed 42
```

## v1 vs v2

v1 (see `legacy_v1/`) framed the contribution as "we found Eq. 1, a concise
refractive-index law". Audit rounds 2 and 3 exposed three substantive
problems with the v1 evidence:

1. The "LOCO" evaluation applied a frozen expression — already trained on
   all anions — anion-stratified; it was not a leakage-free discovery test.
2. The Strong-PySR baseline selected on training RMSE while Harness selected
   on a screening slice, so the head-to-head compared selection policies,
   not governance.
3. The 4-agent council was advertised but only 3 agents voted at runtime;
   `DataAgent._vote_feature_selection` raised `NameError: name 'context'
   is not defined` on every round (now fixed in Phase 1).

v2 addresses all three with re-runs under the reviewer's asks (see
`experiments/README.md`) and re-frames the contribution around the harness
rather than the specific output expression.

## Reproducibility contract

Every result that appears in the v2 paper must come from a directory in
`experiments/E*/runs/<id>/` that contains:

- `manifest.json` — seed, git commit hash, config hash, data hashes, package
  versions, timestamps (written by `src/v2/manifest.py`)
- the raw outputs the figure / table was rendered from

Numbers quoted from `legacy_v1/` are labelled as "v1 run YYYYMMDD_HHMMSS"
and are reported as historical, not primary evidence.

## License

Code: MIT. Data derived from JARVIS-DFT (public domain, courtesy NIST).
