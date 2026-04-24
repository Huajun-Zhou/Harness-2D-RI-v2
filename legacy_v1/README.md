# legacy_v1/ — v1 paper artefacts (archived 2026-04-24)

This directory is the frozen snapshot of everything produced for v1 of the
manuscript (the "we found Eq. 1" narrative). Nothing here should be edited or
rerun; it exists so that the v2 paper can reference v1 numbers and so we can
prove what was known at submission of v1.

**v1 narrative** — "Governance-guided symbolic regression discovers an 11-node
closed-form refractive-index law for 2D binary semiconductors."

**v2 narrative** (in progress) — "Harness engineering stabilises symbolic
regression for small-data optical modelling of 2D semiconductors." Eq. 1 is
a representative output of the harness, not the contribution itself.

## Contents

| Path | Purpose |
| --- | --- |
| `outputs_v1/` | 103 per-round PySR run directories from 2026-04-20. All contributed to v1 Table 2 / Table 4 / Table 6. |
| `results_v1/` | v1 CSV/JSON result artefacts. The source of every number in v1 tables. |
| `scripts_v1/` | v1 analysis scripts. **Two of these had a variable-mapping bug** — see below — which is the forensic reason the v2 paper no longer makes the `17-node pre-veto` claim. |
| `run_strong_pysr.py` | v1 Strong-PySR entry point — used `train-RMSE` selection only. v2 replaces this with four matched-selection policies. |

## Known bugs in v1 artefacts (carried forward from audit round 2)

1. **Variable-mapping bug in `scripts_v1/test_stability_veto.py` and
   `scripts_v1/screen_pareto_stability.py`.** Both scripts assume
   `x2=Z_X, x4=d_e`, but the actual PySR runs used `x2=d_e, x4=bond_polarity`
   (see `discovery_report_20260420_165841.json::features_used`). Consequence:
   the `b2_pareto_stability.csv` rmse_chalcogen/halogen/pnictogen columns are
   nonsense, and the "17-node pre-veto expression is unstable on 12% of the
   grid" narrative used in v1 §Moss-law is **unsupported**. v2 drops this
   narrative (see audit round 2 notes).

2. **Silent DataAgent abstain.** v1 ran with `_vote_feature_selection`
   referencing a `context` variable that wasn't in scope, raising NameError
   on every round. MetaAgent's exception handler converted every DataAgent
   vote to ABSTAIN. Net effect: v1 council had 3 active agents
   (theory-prior, validator, meta-resolver), not 4. v1 Methods disclosed
   this; v2 fixes the bug in `agents/data_agent.py` and reruns the
   ablation on a genuine 4-agent council.

## Policy

- **Read-only.** Do not edit files in this directory.
- Numbers referenced from v2 must either match a re-run or cite v1 explicitly
  as "v1 run, timestamp YYYYMMDD_HHMMSS".
- If v2 proves a v1 claim wrong, amend the v2 paper, not v1 (v1 is a historical
  record).
