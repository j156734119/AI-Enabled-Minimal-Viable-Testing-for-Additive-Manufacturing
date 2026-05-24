"""
Step 05: Build structured dataset.

This script will later combine validated extraction outputs into structured
CSV files for modelling.
"""

from pathlib import Path

from am_mvt.config import get_path


def create_empty_csv_if_missing(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text(header + "\n", encoding="utf-8")


def main() -> None:
    processed_dir = get_path("data", "processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    create_empty_csv_if_missing(
        processed_dir / "sources.csv",
        "source_id,doi,title,journal,year,source_type,url,licence,notes",
    )

    create_empty_csv_if_missing(
        processed_dir / "build_conditions.csv",
        (
            "build_id,source_id,alloy,alloy_family,am_process,"
            "build_orientation,surface_condition,heat_treatment,"
            "laser_power_W,scan_speed_mm_s,hatch_spacing_um,"
            "layer_thickness_um,ved_J_mm3,defect_type,"
            "porosity_percent,residual_stress_indicator"
        ),
    )

    create_empty_csv_if_missing(
        processed_dir / "mechanical_tests.csv",
        (
            "test_id,build_id,test_type,yield_strength_MPa,uts_MPa,"
            "elongation_percent,fatigue_life_cycles,stress_amplitude_MPa,"
            "r_ratio,runout,failure_mode"
        ),
    )

    create_empty_csv_if_missing(
        processed_dir / "extraction_audit.csv",
        (
            "record_id,source_id,field_name,extracted_value,evidence_text,"
            "extraction_confidence,needs_human_check,human_checked,"
            "corrected_value,comment"
        ),
    )

    create_empty_csv_if_missing(
        processed_dir / "modelling_dataset.csv",
        (
            "source_id,build_id,test_id,alloy,alloy_family,am_process,"
            "build_orientation,surface_condition,heat_treatment,"
            "laser_power_W,scan_speed_mm_s,hatch_spacing_um,"
            "layer_thickness_um,ved_J_mm3,defect_type,porosity_percent,"
            "test_type,yield_strength_MPa,uts_MPa,elongation_percent,"
            "fatigue_life_cycles,runout,failure_mode"
        ),
    )

    print("Step 05 complete: processed dataset CSV files prepared.")
    print(f"Processed data folder: {processed_dir}")


if __name__ == "__main__":
    main()