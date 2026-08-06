from __future__ import annotations

import argparse

from am_mvt.modelling.compose_runs import compose_experiment_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose disjoint target-specific experiment runs.",
    )
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--supplement-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument(
        "--replace-conflicts",
        action="store_true",
        help="Replace matching model/target/mode/route entries from the base run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = compose_experiment_runs(
        args.base_run,
        args.supplement_run,
        args.output_run,
        replace_conflicts=args.replace_conflicts,
    )
    print(f"Composed experiment run: {output}")


if __name__ == "__main__":
    main()
