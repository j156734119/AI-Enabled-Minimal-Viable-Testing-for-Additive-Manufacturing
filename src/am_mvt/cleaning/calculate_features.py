from __future__ import annotations

import numpy as np
import pandas as pd


def _to_float_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def calculate_ved(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "ved_J_mm3" not in result.columns:
        result["ved_J_mm3"] = np.nan

    laser_power = _to_float_series(result, "laser_power_W")
    scan_speed = _to_float_series(result, "scan_speed_mm_s")
    hatch_spacing_um = _to_float_series(result, "hatch_spacing_um")
    layer_thickness_um = _to_float_series(result, "layer_thickness_um")
    existing_ved = _to_float_series(result, "ved_J_mm3")

    hatch_spacing_mm = hatch_spacing_um / 1000.0
    layer_thickness_mm = layer_thickness_um / 1000.0

    denominator = scan_speed * hatch_spacing_mm * layer_thickness_mm

    calculated_ved = pd.Series(np.nan, index=result.index, dtype="float64")
    valid_mask = (
        laser_power.notna()
        & denominator.notna()
        & np.isfinite(denominator)
        & (denominator > 0)
    )

    calculated_ved.loc[valid_mask] = laser_power.loc[valid_mask] / denominator.loc[
        valid_mask
    ]

    result["ved_J_mm3"] = existing_ved.fillna(calculated_ved)

    return result


def calculate_porosity_from_relative_density(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "porosity_percent" not in result.columns:
        result["porosity_percent"] = np.nan

    if "relative_density_percent" not in result.columns:
        return result

    porosity = _to_float_series(result, "porosity_percent")
    relative_density = _to_float_series(result, "relative_density_percent")

    calculated_porosity = pd.Series(np.nan, index=result.index, dtype="float64")
    valid_mask = (
        relative_density.notna()
        & np.isfinite(relative_density)
        & (relative_density >= 0)
        & (relative_density <= 100)
    )

    calculated_porosity.loc[valid_mask] = 100.0 - relative_density.loc[valid_mask]
    result["porosity_percent"] = porosity.fillna(calculated_porosity)

    return result


def calculate_fatigue_cycles_from_hours(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "fatigue_life_cycles" not in result.columns:
        result["fatigue_life_cycles"] = np.nan

    if "fatigue_life_h" not in result.columns or "frequency_Hz" not in result.columns:
        return result

    existing_cycles = _to_float_series(result, "fatigue_life_cycles")
    life_h = _to_float_series(result, "fatigue_life_h")
    frequency = _to_float_series(result, "frequency_Hz")

    calculated_cycles = pd.Series(np.nan, index=result.index, dtype="float64")
    valid_mask = (
        life_h.notna()
        & frequency.notna()
        & np.isfinite(life_h)
        & np.isfinite(frequency)
        & (life_h > 0)
        & (frequency > 0)
    )

    calculated_cycles.loc[valid_mask] = (
        life_h.loc[valid_mask] * 3600.0 * frequency.loc[valid_mask]
    )

    result["fatigue_life_cycles"] = existing_cycles.fillna(calculated_cycles)

    return result


def calculate_log_fatigue_life(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "fatigue_life_cycles" not in result.columns:
        result["log10_fatigue_life_cycles"] = np.nan
        return result

    fatigue_life = _to_float_series(result, "fatigue_life_cycles")

    log_values = pd.Series(np.nan, index=result.index, dtype="float64")
    valid_mask = fatigue_life.notna() & np.isfinite(fatigue_life) & (
        fatigue_life > 0
    )

    log_values.loc[valid_mask] = np.log10(fatigue_life.loc[valid_mask])

    result["log10_fatigue_life_cycles"] = log_values

    return result


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = calculate_ved(result)
    result = calculate_porosity_from_relative_density(result)
    result = calculate_fatigue_cycles_from_hours(result)
    result = calculate_log_fatigue_life(result)

    return result