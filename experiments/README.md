# experiments/ — v2 experiment campaigns

Each subdirectory is one **experiment** (capital "E") corresponding to a
numbered claim in the v2 paper. An experiment has:

  - `configs/` — one YAML config per variant (versioned, committed)
  - `runs/` — one subdir per run; contains `manifest.json`, raw outputs, logs
  - `figures/` — final figures that consume `runs/*` data

Never hand-edit anything in `runs/`. If a number needs to change, rerun and
re-render the figure.

## The five experiments

### E1 — Full Harness on fixed council (headline reference run)
Single run of the full 4-agent Harness on Tier-2 (N=193, 80/20 seed=42) with
the **DataAgent bug fixed**. Produces:
  - the v2 reference expression (Eq. 1 in v2, may or may not equal v1 Eq. 1)
  - per-round veto/approve tallies showing all 4 agents active
  - full Pareto front
This run is the "what does the fixed system produce" anchor. Everything else
compares back to it.

### E2 — Outer-LOCO Harness discovery (the reviewer's main ask)
Ten independent Harness discovery runs. For each anion a∈{O,S,Se,Te,F,Cl,Br,I,N,P},
the outer training pool is all binaries with anion ≠ a; a new Harness
search is run on this pool with an internal train/screening split, and the
resulting frozen expression is evaluated on the held-out anion. This is a
**true leakage-free LOCO** — the harness never sees any material from the
test anion during discovery.

Reports: per-fold RMSE, complexity, stability rate, and which expressions
generalise vs. which anions are genuinely out-of-distribution.

### E3 — Matched PySR baselines (fair-selection controls)
Four PySR variants run on the SAME outer splits as E2, all with the same
10-round × 40-island × 100-iter budget and five seeds each:
  1. `vanilla-train-selected` — v1's selection policy (lowest train RMSE)
  2. `vanilla-screening-selected` — same screening split Harness uses
  3. `vanilla-complexity-capped` — same as #2 but maxsize ≤ 11
  4. `stability-veto-only` — #2 + the stability post-filter (isolates the veto)

The Full Harness advantage must survive vs. #2 (otherwise it's a selection
effect, not a governance effect). The gap to #4 isolates causal critique +
knowledge memory + adaptive operators.

### E4 — Leave-one-module-out ablation on the fixed council
Five variants on the E1 split, 3 seeds each:
  - FULL (all four governance modules)
  - −AgentCouncil (PySR + LLM only)
  - −Causal diagnostics
  - −Knowledge memory
  - −Adaptive operators

v1 had this table but ran on the broken 3-agent council, so the
"agent council is largest contributor (+0.097)" number needs to be
re-measured on the 4-agent version.

### E5 — Diagnostics defending against "it's a Z_M lookup" criticism
  - `Zm_only_pysr` — PySR with only Z_M; if it can't find Eq. 1, governance
    matters, not the variable
  - `leave_one_metal_out` — remove one metal entirely; test whether the
    expression extrapolates across electronic-structure rows
  - `target_permutation` — shuffle target within anion family; a model
    that still transfers is a table-lookup

---

## How to run an experiment (Phase 2+)

    # example: one E2 fold
    python3 -m harness_runner \
        --experiment E2_outer_loco \
        --config experiments/E2_outer_loco/configs/fold_O.yaml \
        --run-seed 42 \
        --out-dir experiments/E2_outer_loco/runs/

Every invocation writes a timestamped run directory containing
`manifest.json` (seed, git commit, data hashes, package versions) plus the
run's own outputs. Figures are regenerated from `runs/` by scripts in
`experiments/E*/figures/`.
