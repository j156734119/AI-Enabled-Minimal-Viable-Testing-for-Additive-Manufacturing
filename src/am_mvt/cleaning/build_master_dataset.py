from __future__ import annotations

import pandas as pd

from am_mvt.cleaning.calculate_features import add_engineered_features
from am_mvt.cleaning.normalise_alloys import apply_alloy_normalisation
from am_mvt.cleaning.normalise_defects import apply_defect_normalisation
from am_mvt.cleaning.project_schema import MASTER_COLUMNS
from am_mvt.cleaning.unit_conversion import standardise_units
from am_mvt.config import get_path
from am_mvt.ingestion.load_fatigue_database import load_fatigue_database_master_rows
from am_mvt.ingestion.load_materials_design_dataset import (
    load_materials_design_master_rows,
)


MINIMUM_USEFUL_COLUMNS = [
    "alloy",
    "alloy_family",
    "am_process",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "relative_density_percent",
    "porosity_percent",
    "yield_strength_MPa",
    "uts_MPa",
    "elongation_percent",
    "fatigue_life_cycles",
    "stress_amplitude_MPa",
    "max_stress_MPa",
    "delta_K_MPa_sqrt_m",
    "da_dN_m_per_cycle",
]


def keep_project_relevant_rows(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    available_cols = [col for col in MINIMUM_USEFUL_COLUMNS if col in result.columns]

    if not available_cols:
        return result

    useful_count = result[available_cols].notna().sum(axis=1)
    result = result.loc[useful_count > 0].copy()

    return result


def build_master_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    fatigue_df, fatigue_report = load_fatigue_database_master_rows()
    materials_df, materials_report = load_materials_design_master_rows()

    combined = pd.concat(
        [fatigue_df, materials_df],
        ignore_index=True,
        sort=False,
    )

    for col in MASTER_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined = combined[MASTER_COLUMNS]
    combined = standardise_units(combined)
    combined = apply_alloy_normalisation(combined)
    combined = apply_defect_normalisation(combined)
    combined = add_engineered_features(combined)
    combined = keep_project_relevant_rows(combined)

    report_df = pd.concat(
        [fatigue_report, materials_report],
        ignore_index=True,
        sort=False,
    )

    return combined, report_df


def save_master_dataset() -> tuple:
    processed_dir = get_path("data", "processed")
    interim_dir = get_path("data", "interim")
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    master_df, report_df = build_master_dataset()

    master_path = processed_dir / "master_modelling_dataset.csv"

    # Compatibility path for the current training script.
    modelling_path = processed_dir / "modelling_dataset.csv"

    report_path = interim_dir / "master_dataset_build_report.csv"

    master_df.to_csv(master_path, index=False, encoding="utf-8-sig")
    master_df.to_csv(modelling_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    return master_path, modelling_path, report_path, master_df