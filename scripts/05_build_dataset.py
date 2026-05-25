from __future__ import annotations

from am_mvt.cleaning.build_master_dataset import save_master_dataset
from am_mvt.config import get_path


def main() -> None:
    master_path, modelling_path, report_path, master_df = save_master_dataset()

    print("Step 05 complete: project-focused master dataset built.")
    print(f"Master dataset: {master_path}")
    print(f"Compatibility modelling dataset: {modelling_path}")
    print(f"Build report: {report_path}")
    print(f"Rows: {len(master_df)}")
    print(f"Columns: {len(master_df.columns)}")

    if "source_id" in master_df.columns:
        print("\nRows by source:")
        print(master_df["source_id"].value_counts(dropna=False))

    if "task_type" in master_df.columns:
        print("\nRows by task type:")
        print(master_df["task_type"].value_counts(dropna=False))

    selected_cols = [
        "alloy",
        "alloy_family",
        "am_process",
        "laser_power_W",
        "scan_speed_mm_s",
        "hatch_spacing_um",
        "layer_thickness_um",
        "ved_J_mm3",
        "yield_strength_MPa",
        "uts_MPa",
        "elongation_percent",
        "fatigue_life_cycles",
        "stress_amplitude_MPa",
        "max_stress_MPa",
        "r_ratio",
    ]

    existing_cols = [col for col in selected_cols if col in master_df.columns]

    print("\nNon-missing counts for key project fields:")
    print(master_df[existing_cols].notna().sum())

    validation_path = get_path(
        "data",
        "processed",
        "master_dataset_quick_summary.csv",
    )

    summary = master_df[existing_cols].notna().sum().reset_index()
    summary.columns = ["field", "non_missing_count"]
    summary.to_csv(validation_path, index=False, encoding="utf-8-sig")

    print(f"\nQuick summary saved to: {validation_path}")


if __name__ == "__main__":
    main()