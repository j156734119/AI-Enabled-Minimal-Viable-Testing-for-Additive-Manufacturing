from __future__ import annotations

import argparse

from am_mvt.modelling.experiment_inference import predict_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-predict AM experiment scenarios from a Step 06 run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["all", "process_only", "reduced_testing"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = predict_scenarios(
        run_dir=args.run_dir,
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
    )
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()

