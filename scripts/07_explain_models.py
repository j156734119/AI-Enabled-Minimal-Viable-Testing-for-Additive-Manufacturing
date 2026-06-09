"""Step 07: explain trained models and quantify evidence coverage."""

from __future__ import annotations

import argparse

from am_mvt.modelling.model_explanation import run_model_explanation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate permutation importance, grouped errors, coverage, "
            "sensitivity, and relationship-evidence outputs."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/experiments/balanced_v2",
    )
    parser.add_argument(
        "--mode",
        choices=["process_only", "reduced_testing"],
        default="process_only",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Permutation repeats per feature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_model_explanation(
        args.run_dir,
        mode=args.mode,
        repeats=args.repeats,
    )
    print("Step 07 complete: model explanation and relationship evidence generated.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
