from __future__ import annotations

import argparse
from datetime import datetime

from am_mvt.modelling.experiment_training import run_experiment_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train grouped-CV process-only and reduced-testing AM models, "
            "including censor-aware AFT and Basquin residual fatigue routes."
        )
    )
    parser.add_argument(
        "--run-name",
        default=f"balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="New directory name under outputs/experiments/.",
    )
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=["fast", "balanced", "standard"],
        help="CPU-fast, five-fold balanced, or full comparison profile.",
    )
    parser.add_argument(
        "--mode",
        default="process_only",
        choices=["process_only", "reduced_testing", "all"],
        help="Prediction mode selection.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Override grouped CV folds (fast=3, standard=5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Step 06: physics-anchored and censor-aware model training")
    print(f"Run name: {args.run_name}")
    print(f"Profile: {args.profile}")
    print(f"Mode: {args.mode}")
    print("Existing project metrics and models will not be overwritten.")
    run_dir = run_experiment_suite(
        run_name=args.run_name,
        profile=args.profile,
        n_splits=args.cv_folds,
        mode=args.mode,
    )
    print("\nStep 06 complete.")
    print(f"Experiment outputs: {run_dir}")
    print(f"Summary: {run_dir / 'tables' / 'experiment_summary.csv'}")
    print(f"Registry: {run_dir / 'model_registry.json'}")


if __name__ == "__main__":
    main()
