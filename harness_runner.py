"""
harness_runner.py — v2 single entry point for all experiments.

This script replaces v1's `main.py` and `run_strong_pysr.py`. It dispatches on
`--experiment` to the appropriate runner in `src/v2/runners/` (added in Phase 2).

Every invocation:
  1. Resolves an output directory under ``experiments/<experiment>/runs/``
  2. Builds a `RunManifest` capturing seed, git commit, data hashes, package
     versions, and config hash
  3. Writes the manifest to ``<out>/manifest.json`` BEFORE any computation
  4. Runs the experiment-specific logic
  5. Stamps the manifest end-time on successful completion

This is the contract for every number that will appear in the v2 paper.
If a result directory doesn't have a manifest, it doesn't go in the paper.

Phase 1 status
--------------
Only the CLI + manifest wiring is implemented here. Experiment dispatch
raises NotImplementedError for now; Phase 2 fills in the five runners.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.v2.manifest import RunManifest


SUPPORTED_EXPERIMENTS = (
    "E1_full_harness_fixed_council",
    "E2_outer_loco",
    "E3_matched_pysr_baselines",
    "E4_ablation_fixed_council",
    "E5_diagnostics",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified runner for Harness-2D-RI v2 experiments.",
    )
    p.add_argument(
        "--experiment", required=True, choices=SUPPORTED_EXPERIMENTS,
        help="Which experiment campaign to run.",
    )
    p.add_argument(
        "--config", required=True, type=Path,
        help="Path to the YAML config for this variant of the experiment.",
    )
    p.add_argument(
        "--run-seed", required=True, type=int,
        help="Root seed for this run. All downstream seeds derive from this.",
    )
    p.add_argument(
        "--run-id", type=str, default=None,
        help="Optional human-readable run ID; defaults to a UTC timestamp.",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("experiments"),
        help="Parent directory for runs/. The actual run dir will be "
             "<out-dir>/<experiment>/runs/<run-id>/ (or similar).",
    )
    p.add_argument(
        "--notes", type=str, default="",
        help="Free-form note recorded in the manifest.",
    )
    return p.parse_args()


def _resolve_run_dir(out_dir: Path, experiment: str, run_id: str) -> Path:
    run_dir = out_dir / experiment / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dispatch(experiment: str, config_path: Path, run_seed: int,
              run_dir: Path) -> dict:
    """Route to the experiment-specific runner. Phase 2 fills these in."""
    if experiment == "E1_full_harness_fixed_council":
        from src.v2.runners import E1_full_harness
        return E1_full_harness.run(config_path, run_seed, run_dir)
    # E2–E5 will be wired in subsequent Phase 2 sub-tasks.
    raise NotImplementedError(
        f"Runner for {experiment!r} is not implemented yet (Phase 2 TODO)."
    )


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("harness_runner")

    run_id = args.run_id or f"{args.experiment}_seed{args.run_seed}_{_default_run_id()}"
    run_dir = _resolve_run_dir(args.out_dir, args.experiment, run_id)
    log.info("Run dir: %s", run_dir)

    # List the data files likely touched by each experiment so the manifest
    # captures their hashes. Experiments can add more in Phase 2.
    default_data_files = ["data/extended/tier2_binary.csv"]
    if args.experiment == "E1_full_harness_fixed_council" or \
       args.experiment == "E2_outer_loco" or \
       args.experiment == "E4_ablation_fixed_council":
        data_files = default_data_files
    else:
        data_files = default_data_files + ["data/extended/tier3_binary_plus_ternary.csv"]

    manifest = RunManifest(
        run_id=run_id,
        experiment=args.experiment,
        run_seed=args.run_seed,
        config_path=str(args.config.resolve()),
        data_files=data_files,
        notes=args.notes,
    ).build()

    # Write manifest BEFORE compute so we have a record even if the run crashes.
    manifest_path = manifest.write(run_dir)
    log.info("Wrote pre-flight manifest: %s", manifest_path)

    try:
        outputs = _dispatch(
            experiment=args.experiment,
            config_path=args.config,
            run_seed=args.run_seed,
            run_dir=run_dir,
        )
        log.info("Experiment finished OK; outputs: %s", list(outputs))
    except NotImplementedError as e:
        log.warning("%s", e)
        log.warning("Phase 1 skeleton — exiting after manifest write.")
        return 0

    # Stamp end-time on successful completion.
    manifest.finalise().write(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
