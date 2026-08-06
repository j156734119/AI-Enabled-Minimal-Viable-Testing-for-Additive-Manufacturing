"""Step 08: generate an evidence- and coverage-aware testing matrix."""

from __future__ import annotations

import argparse

from am_mvt.optimisation.actionable_matrix import (
    ActionableMatrixConfig,
    generate_actionable_testing_matrix,
)
from am_mvt.optimisation.testing_matrix import generate_testing_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the reduced but representative testing matrix."
    )
    parser.add_argument(
        "--run-dir",
        default="outputs/experiments/balanced_v2",
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
        default=None,
        help="Defaults to <run-dir>/example_scenario_predictions.csv.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Generate the legacy Step 08 output for dissertation comparison.",
    )
    parser.add_argument("--static-budgets", nargs="+", type=int, default=None)
    parser.add_argument("--fatigue-budgets", nargs="+", type=int, default=None)
    parser.add_argument("--static-replicates", type=int, default=3)
    parser.add_argument("--fatigue-stress-levels", type=int, default=5)
    parser.add_argument("--fatigue-replicates-per-level", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.legacy:
        outputs = generate_testing_matrix(
            args.run_dir,
            output_path=args.output,
            scenario_input=args.scenario_input,
            scenario_output=args.scenario_output,
        )
        label = "legacy comparison matrix"
    else:
        config = ActionableMatrixConfig(
            static_budgets=tuple(args.static_budgets or (24, 36, 48)),
            fatigue_budgets=tuple(args.fatigue_budgets or (30, 45, 60)),
            static_replicates=args.static_replicates,
            fatigue_stress_levels=args.fatigue_stress_levels,
            fatigue_replicates_per_level=args.fatigue_replicates_per_level,
        )
        outputs = generate_actionable_testing_matrix(
            args.run_dir,
            config=config,
        )
        label = "domain readiness and evidence-validation matrices"
    print(f"Step 08 complete: {label} generated.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
