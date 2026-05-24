"""
Step 06: Train baseline models.

This script will later train regression and classification models using the
processed modelling dataset.
"""

from am_mvt.config import get_path


def main() -> None:
    dataset_path = get_path("data", "processed", "modelling_dataset.csv")
    model_dir = get_path("outputs", "models")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Step 06 placeholder: model training will be implemented later.")
    print(f"Expected dataset path: {dataset_path}")
    print(f"Model output folder: {model_dir}")


if __name__ == "__main__":
    main()