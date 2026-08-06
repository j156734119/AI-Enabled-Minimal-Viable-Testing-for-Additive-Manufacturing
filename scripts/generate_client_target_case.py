from __future__ import annotations

import argparse

from am_mvt.optimisation.client_target_matrix import generate_client_target_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a client-targeted pilot testing matrix.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--static-budget", type=int, default=24)
    parser.add_argument("--fatigue-budget", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_client_target_matrix(
        args.run_dir,
        args.target_file,
        static_budget=args.static_budget,
        fatigue_budget=args.fatigue_budget,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
