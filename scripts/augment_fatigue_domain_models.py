from __future__ import annotations

import argparse

from am_mvt.modelling.experiment_training import augment_aft_domain_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add exact and alloy-family AFT models to an existing run.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", default="process_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = augment_aft_domain_models(args.run_dir, args.mode)
    print(f"Domain AFT models added: {count}")


if __name__ == "__main__":
    main()
