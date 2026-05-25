from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


PROVENANCE_COLUMNS = [
    "source_id",
    "source_name",
    "source_file",
    "source_sheet",
    "dataset_id",
    "record_id",
    "doi",
    "source_title",
    "source_year",
    "source_url",
]

CONTROL_COLUMNS = [
    "task_type",
    "extraction_method",
    "needs_human_check",
]

INPUT_COLUMNS = [
    "alloy",
    "alloy_family",
    "am_process",
    "machine_model",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "build_orientation",
    "test_direction",
    "scan_strategy",
    "layer_rotation_degree",
    "build_plate_temperature_C",
    "surface_condition",
    "heat_treatment",
    "post_processing",
    "porosity_percent",
    "relative_density_percent",
    "density_measurement_method",
    "defect_type",
    "residual_stress_indicator",
]

OUTPUT_COLUMNS = [
    "test_type",
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
    "log10_fatigue_life_cycles",
    "runout",
    "failure_mode",
    "fracture_origin",
]

MASTER_COLUMNS = PROVENANCE_COLUMNS + CONTROL_COLUMNS + INPUT_COLUMNS + OUTPUT_COLUMNS


COLUMN_ALIASES: dict[str, list[str]] = {
    "source_id": ["source id", "source_id"],
    "source_name": ["source name", "source_name"],
    "source_file": ["source file", "source_file"],
    "source_sheet": ["source sheet", "source_sheet"],
    "record_id": ["record id", "record_id"],
    "dataset_id": [
        "dataset id",
        "dataset_id",
        "data set id",
        "datasetid",
        "id",
    ],
    "doi": [
        "doi",
        "source doi",
    ],
    "source_title": [
        "title",
        "paper title",
        "publication title",
        "source title",
    ],
    "source_year": [
        "year",
        "year of publication",
        "publication year",
    ],
    "source_url": [
        "link to paper",
        "url",
        "source url",
    ],
    "alloy": [
        "material",
        "alloy",
        "name of the material",
        "material name",
        "alloy name",
    ],
    "alloy_family": [
        "alloy family",
        "material family",
    ],
    "am_process": [
        "types of am",
        "type of am",
        "am process",
        "process",
        "method",
        "manufacturing process",
        "am method",
    ],
    "machine_model": [
        "am machine",
        "machine",
        "model",
        "machine model",
        "machine name",
    ],
    "laser_power_W": [
        "power",
        "power w",
        "power (w)",
        "laser power",
        "laser power w",
        "laser power (w)",
    ],
    "scan_speed_mm_s": [
        "scan speed",
        "scan speed mm/s",
        "scan speed (mm/s)",
        "scanning speed",
        "laser speed",
        "laser speed mm/s",
        "laser speed (mm/s)",
    ],
    "hatch_spacing_um": [
        "hatch space",
        "hatch space um",
        "hatch space (um)",
        "hatch space μm",
        "hatch space (μm)",
        "hatch spacing",
        "hatch spacing um",
        "hatch spacing (um)",
        "hatch spacing μm",
        "hatch spacing (μm)",
    ],
    "layer_thickness_um": [
        "layer thickness",
        "layer thickness um",
        "layer thickness (um)",
        "layer thickness μm",
        "layer thickness (μm)",
    ],
    "ved_J_mm3": [
        "volumetric energy density",
        "volumetric energy density j/mm3",
        "volumetric energy density (j/mm3)",
        "energy density",
        "ved",
    ],
    "build_orientation": [
        "direction of specimen",
        "direciton of specimen",
        "specimen orientation",
        "build orientation",
        "orientation",
    ],
    "test_direction": [
        "test direction",
        "testing direction",
    ],
    "scan_strategy": [
        "scan pattern",
        "scan strategy",
        "scanning strategy",
    ],
    "layer_rotation_degree": [
        "layer scan rotation",
        "layer scan rotation degree",
        "layer scan rotation °",
        "layer scan rotation (°)",
        "layer rotation degree",
        "layer rotation",
        "layer rotation (degree)",
    ],
    "build_plate_temperature_C": [
        "preheat temperature",
        "preheat temperature c",
        "preheat temperature °c",
        "preheat temperature (°c)",
        "build plate temperature",
        "build plate temperature c",
        "build plate temperature °c",
        "build plate temperature (°c)",
    ],
    "surface_condition": [
        "surface condition",
        "surface treatment",
        "surface finish",
    ],
    "heat_treatment": [
        "heat treatment",
        "treated",
        "treated hip/y/n",
        "treated (hip/y/n)",
    ],
    "post_processing": [
        "processing sequence and parameters",
        "post processing",
        "post-processing",
    ],
    "porosity_percent": [
        "porosity",
        "porosity percent",
        "porosity %",
        "porosity (%)",
    ],
    "relative_density_percent": [
        "relative density",
        "relative density percent",
        "relative density %",
        "relative density (%)",
        "density %",
        "density (%)",
        "consolidation",
        "consolidation %",
        "consolidation (%)",
    ],
    "density_measurement_method": [
        "density measurement method",
    ],
    "defect_type": [
        "defect type",
        "defect",
        "pore type",
        "flaw type",
    ],
    "test_type": [
        "types of fatigue tests",
        "test type",
        "mechanical test",
        "fatigue test type",
    ],
    "test_temperature_C": [
        "test temperature",
        "test temerature",
        "test temperature c",
        "test temerature c",
        "test temperature ℃",
        "test temerature ℃",
        "fatigue temperature",
        "fatigue temperature c",
        "fatigue temperature °c",
        "fatigue temperature (°c)",
    ],
    "yield_strength_MPa": [
        "yield strength",
        "yield strength mpa",
        "yield strength (mpa)",
        "yield stress",
        "yield stress mpa",
        "yield stress (mpa)",
    ],
    "uts_MPa": [
        "ultimate tensile strength",
        "ultimate tensile strength mpa",
        "ultimate tensile strength (mpa)",
        "tensile strength",
        "uts",
    ],
    "elongation_percent": [
        "elongation",
        "elongation percent",
        "elongation %",
        "elongation (%)",
    ],
    "youngs_modulus_GPa": [
        "young's modulus",
        "youngs modulus",
        "youngs modulus gpa",
        "youngs modulus (gpa)",
    ],
    "hardness_HV": [
        "hardness",
        "hardness hv",
        "hardness (hv)",
    ],
    "stress_amplitude_MPa": [
        "stress amplitude",
        "stress amplitude mpa",
        "stress amplitude (mpa)",
        "stress amplitude σa mpa",
        "stress amplitude σa (mpa)",
        "stress amplitude sigma a mpa",
        "stress amp",
    ],
    "max_stress_MPa": [
        "max stress",
        "max stress mpa",
        "max stress (mpa)",
        "maximum stress",
        "maximum stress mpa",
        "maximum stress (mpa)",
        "max stress/strain",
    ],
    "strain_amplitude": [
        "strain amplitude",
        "strain amplitude %",
        "strain amplitude (%)",
        "strain amplitude εa",
        "strain amplitude εa (%)",
        "strain amplitude ea",
    ],
    "delta_K_MPa_sqrt_m": [
        "delta k",
        "delta k mpa sqrt m",
        "Δk",
        "dk",
        "stress intensity factor range",
        "stress intensity factor range mpa sqrt m",
    ],
    "da_dN_m_per_cycle": [
        "da/dn",
        "dadn",
        "da dn",
        "crack growth rate",
        "crack growth rate da/dn",
    ],
    "r_ratio": [
        "load ratio",
        "stress ratio",
        "r ratio",
        "r value",
        "r",
    ],
    "frequency_Hz": [
        "frequency",
        "frequency hz",
        "frequency (hz)",
    ],
    "fatigue_life_cycles": [
        "fatigue life",
        "fatigue life cycles",
        "fatigue life (cycles)",
        "cycles to failure",
        "number of cycles",
        "life cycles",
        "life n cycle",
        "life n cycles",
        "life n (cycle)",
        "life n (cycles)",
        "n cycle",
        "n cycles",
        "n_f",
        "nf",
    ],
    "fatigue_life_h": [
        "fatigue life h",
        "fatigue life (h)",
        "creep life",
        "creep life h",
        "creep life (h)",
    ],
    "runout": [
        "runout",
        "run out",
        "censored",
        "survived",
    ],
    "failure_mode": [
        "failure mode",
        "fracture mode",
    ],
    "fracture_origin": [
        "fracture origin",
        "crack initiation site",
    ],
}


def normalise_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("μ", "u")
    text = text.replace("µ", "u")
    text = text.replace("ε", "e")
    text = text.replace("σ", "sigma")
    text = text.replace("δ", "delta")
    text = text.replace("Δ", "delta")
    text = text.replace("℃", "c")
    text = text.replace("°c", "c")
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}

    for standard_column, aliases in COLUMN_ALIASES.items():
        lookup[normalise_column_name(standard_column)] = standard_column

        for alias in aliases:
            lookup[normalise_column_name(alias)] = standard_column

    return lookup


def infer_standard_column(original_column: object) -> str | None:
    normalised = normalise_column_name(original_column)
    lookup = build_alias_lookup()

    if normalised in lookup:
        return lookup[normalised]

    if "dataset" in normalised and "id" in normalised:
        return "dataset_id"

    if "power" in normalised and "laser" in normalised:
        return "laser_power_W"

    if "scan speed" in normalised or "scanning speed" in normalised:
        return "scan_speed_mm_s"

    if "hatch" in normalised and ("space" in normalised or "spacing" in normalised):
        return "hatch_spacing_um"

    if "layer thickness" in normalised:
        return "layer_thickness_um"

    if "volumetric energy density" in normalised or normalised == "ved":
        return "ved_J_mm3"

    if "yield" in normalised and ("strength" in normalised or "stress" in normalised):
        return "yield_strength_MPa"

    if "ultimate tensile strength" in normalised or normalised == "uts":
        return "uts_MPa"

    if "elongation" in normalised:
        return "elongation_percent"

    if "relative density" in normalised or "consolidation" in normalised:
        return "relative_density_percent"

    if "porosity" in normalised:
        return "porosity_percent"

    if "stress amplitude" in normalised:
        return "stress_amplitude_MPa"

    if "strain amplitude" in normalised:
        return "strain_amplitude"

    if "max stress" in normalised or "maximum stress" in normalised:
        return "max_stress_MPa"

    if "load ratio" in normalised or "stress ratio" in normalised:
        return "r_ratio"

    if normalised == "r":
        return "r_ratio"

    if "frequency" in normalised:
        return "frequency_Hz"

    if "life" in normalised and ("cycle" in normalised or "cycles" in normalised):
        return "fatigue_life_cycles"

    if normalised in {"n cycle", "n cycles", "life n cycle", "life n cycles"}:
        return "fatigue_life_cycles"

    if "fatigue life" in normalised and "h" in normalised:
        return "fatigue_life_h"

    if "delta k" in normalised or "stress intensity factor range" in normalised:
        return "delta_K_MPa_sqrt_m"

    if "da dn" in normalised or "dadn" in normalised or "crack growth rate" in normalised:
        return "da_dN_m_per_cycle"

    if "runout" in normalised or "run out" in normalised:
        return "runout"

    return None


def build_column_mapping(columns: Iterable[object]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for original_column in columns:
        inferred = infer_standard_column(original_column)

        if inferred is not None:
            mapping[str(original_column)] = inferred

    return mapping


def make_unique_columns(columns: list[object]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []

    for index, col in enumerate(columns):
        if pd.isna(col):
            base = f"unnamed_column_{index}"
        else:
            base = str(col).strip() or f"unnamed_column_{index}"

        if base not in seen:
            seen[base] = 0
            result.append(base)
        else:
            seen[base] += 1
            result.append(f"{base}.{seen[base]}")

    return result


def combine_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    combined: dict[str, pd.Series] = {}

    for column_index, column_name in enumerate(df.columns):
        column_name = str(column_name)
        series = df.iloc[:, column_index].copy()
        series.name = column_name

        if column_name not in combined:
            combined[column_name] = series
        else:
            combined[column_name] = combined[column_name].combine_first(series)

    return pd.DataFrame(combined, index=df.index)


def standardise_table_to_project_schema(df: pd.DataFrame) -> pd.DataFrame:
    mapping = build_column_mapping(df.columns)
    renamed = df.rename(columns=mapping).copy()
    renamed = combine_duplicate_columns(renamed)

    for col in MASTER_COLUMNS:
        if col not in renamed.columns:
            renamed[col] = pd.NA

    return renamed[MASTER_COLUMNS]


def build_column_mapping_report(columns: Iterable[object]) -> pd.DataFrame:
    mapping = build_column_mapping(columns)

    rows = []

    for col in columns:
        col_text = str(col)

        rows.append(
            {
                "original_column": col_text,
                "normalised_column": normalise_column_name(col_text),
                "mapped_column": mapping.get(col_text, ""),
                "is_mapped": col_text in mapping,
            }
        )

    return pd.DataFrame(rows)