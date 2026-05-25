from __future__ import annotations

from pathlib import Path

import pandas as pd

from am_mvt.cleaning.schema_mapping import STANDARD_COLUMNS


def validate_modelling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict[str, object]] = []

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            issues.append(
                {
                    "issue_type": "missing_column",
                    "column": col,
                    "row_index": "",
                    "value": "",
                    "message": f"Required standard column is missing: {col}",
                }
            )

    numeric_ranges = {
        "laser_power_W": (0, 2000),
        "scan_speed_mm_s": (0, 20000),
        "hatch_spacing_um": (0, 1000),
        "layer_thickness_um": (0, 500),
        "ved_J_mm3": (0, 1000),
        "porosity_percent": (0, 100),
        "relative_density_percent": (0, 100),
        "yield_strength_MPa": (0, 3000),
        "uts_MPa": (0, 4000),
        "elongation_percent": (0, 100),
        "fatigue_life_cycles": (0, 1e12),
        "stress_amplitude_MPa": (0, 3000),
    }

    for col, (lower, upper) in numeric_ranges.items():
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid = numeric.notna() & ((numeric < lower) | (numeric > upper))

        for row_index, value in numeric[invalid].items():
            issues.append(
                {
                    "issue_type": "out_of_expected_range",
                    "column": col,
                    "row_index": row_index,
                    "value": value,
                    "message": f"{col} is outside expected range {lower}–{upper}.",
                }
            )

    if "source_id" in df.columns:
        missing_source = df["source_id"].isna()
        for row_index in df.index[missing_source]:
            issues.append(
                {
                    "issue_type": "missing_source_id",
                    "column": "source_id",
                    "row_index": row_index,
                    "value": "",
                    "message": "Each record should have a source_id.",
                }
            )

    return pd.DataFrame(issues)


def save_validation_report(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = validate_modelling_dataset(df)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path