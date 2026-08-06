from __future__ import annotations

import pandas as pd


NUMERIC_COLUMNS = [
    "source_year",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "layer_rotation_degree",
    "build_plate_temperature_C",
    "porosity_percent",
    "relative_density_percent",
    "surface_roughness_Ra_um",
    "surface_roughness_Rz_um",
    "residual_stress_MPa",
    "critical_section_size_mm",
    "stress_concentration_factor",
    "test_temperature_C",
    "yield_strength_MPa",
    "uts_MPa",
    "elongation_percent",
    "youngs_modulus_GPa",
    "hardness_HV",
    "stress_amplitude_MPa",
    "max_stress_MPa",
    "strain_amplitude",
    "delta_K_MPa_sqrt_m",
    "da_dN_m_per_cycle",
    "r_ratio",
    "frequency_Hz",
    "fatigue_life_cycles",
    "fatigue_life_h",
]


def to_numeric_safe(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in result.columns:
            result[col] = to_numeric_safe(result[col])

    return result


def normalise_boolean_runout(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "runout" not in result.columns:
        return result

    def convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA

        text = str(value).strip().lower()

        if text in {
            "true",
            "yes",
            "y",
            "1",
            "runout",
            "run-out",
            "run out",
            "run-out samples",
            "run out samples",
            "survived",
        }:
            return True

        if text in {"false", "no", "n", "0", "failure", "failed"}:
            return False

        return pd.NA

    result["runout"] = result["runout"].map(convert)

    return result


def standardise_units(df: pd.DataFrame) -> pd.DataFrame:
    result = convert_numeric_columns(df)
    result = normalise_boolean_runout(result)

    return result
