"""Step 08: generate an evidence- and coverage-aware testing matrix."""

from __future__ import annotations

import argparse

from am_mvt.optimisation.testing_matrix import generate_testing_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the reduced but representative testing matrix."
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/experiments/cpu_fast_v1",
    )
    parser.add_argument(
        "--output",
        default="outputs/tables/reduced_testing_matrix.csv",
    )
    parser.add_argument(
        "--scenario-input",
        default="examples/prediction_scenarios_template.csv",
    )
    parser.add_argument(
        "--scenario-output",
        default=(
            "outputs/experiments/cpu_fast_v1/"
            "example_scenario_predictions.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_testing_matrix(
        args.run_dir,
        output_path=args.output,
        scenario_input=args.scenario_input,
        scenario_output=args.scenario_output,
    )
    print("Step 08 complete: reduced testing recommendations generated.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
