from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from am_mvt.config import get_path


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
    "modelling_group_id",
]

MODEL1_FEATURE_COLUMNS = [
    "alloy",
    "alloy_family",
    "am_process",
    "machine_model",
    "am_environment",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "build_orientation",
    "test_direction",
    "scan_strategy",
    "heat_treatment",
    "material_state",
    "surface_condition",
    "surface_roughness_Ra_um",
    "surface_roughness_Rz_um",
    "post_processing",
    "porosity_percent",
    "relative_density_percent",
    "density_measurement_method",
    "defect_type",
    "residual_stress_indicator",
    "residual_stress_MPa",
]

MODEL1_TARGET_COLUMNS = [
    "uts_MPa",
]

MODEL3_TARGET_COLUMNS = [
    "elongation_percent",
    "yield_strength_MPa",
]

MODEL4_TARGET_COLUMNS = [
    "youngs_modulus_GPa",
]

MODEL2_FEATURE_COLUMNS = [
    "alloy",
    "alloy_family",
    "am_process",
    "machine_model",
    "am_environment",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "build_orientation",
    "test_direction",
    "scan_strategy",
    "heat_treatment",
    "material_state",
    "surface_condition",
    "surface_roughness_Ra_um",
    "surface_roughness_Rz_um",
    "post_processing",
    "porosity_percent",
    "relative_density_percent",
    "density_measurement_method",
    "defect_type",
    "residual_stress_indicator",
    "residual_stress_MPa",
    "specimen_description",
    "specimen_geometry",
    "critical_section_dimensions_mm",
    "critical_section_size_mm",
    "stress_concentration_factor",
    "fatigue_environment",
    "fatigue_machine",
    "fatigue_standard",
    "load_control",
    "test_type",
    "yield_strength_MPa",
    "uts_MPa",
    "elongation_percent",
    "stress_amplitude_MPa",
    "max_stress_MPa",
    "r_ratio",
    "frequency_Hz",
    "test_temperature_C",
    "total_strain_amplitude",
    "plastic_strain_amplitude",
    "elastic_strain_amplitude",
    "strain_ratio",
    "strain_rate",
]

MODEL2_TARGET_COLUMNS = [
    "fatigue_life_cycles",
    "log10_fatigue_life_cycles",
    "runout",
]

VIEW_COLUMNS = list(
    dict.fromkeys(
        PROVENANCE_COLUMNS
        + ["task_type", "sample_weight"]
        + MODEL1_FEATURE_COLUMNS
        + MODEL1_TARGET_COLUMNS
        + MODEL3_TARGET_COLUMNS
        + MODEL4_TARGET_COLUMNS
        + ["hardness_HV"]
        + MODEL2_FEATURE_COLUMNS
        + MODEL2_TARGET_COLUMNS
    )
)

NUMERIC_COLUMNS = [
    "source_year",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "porosity_percent",
    "relative_density_percent",
    "surface_roughness_Ra_um",
    "surface_roughness_Rz_um",
    "residual_stress_MPa",
    "critical_section_size_mm",
    "stress_concentration_factor",
    "yield_strength_MPa",
    "uts_MPa",
    "elongation_percent",
    "youngs_modulus_GPa",
    "hardness_HV",
    "stress_amplitude_MPa",
    "max_stress_MPa",
    "r_ratio",
    "frequency_Hz",
    "test_temperature_C",
    "fatigue_life_cycles",
    "log10_fatigue_life_cycles",
]


def read_master_dataset(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = get_path("data", "processed", "master_modelling_dataset.csv")
    else:
        path = Path(path)

    if not path.exists():
        fallback = get_path("data", "processed", "modelling_dataset.csv")

        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                "Could not find master dataset. Expected either "
                f"{path} or {fallback}."
            )

    return pd.read_csv(path, low_memory=False)


def ensure_view_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for col in VIEW_COLUMNS:
        if col not in result.columns:
            result[col] = pd.NA

    return result[VIEW_COLUMNS]


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if "log10_fatigue_life_cycles" not in result.columns:
        result["log10_fatigue_life_cycles"] = np.nan

    if "fatigue_life_cycles" in result.columns:
        fatigue_life = pd.to_numeric(result["fatigue_life_cycles"], errors="coerce")

        missing_log = result["log10_fatigue_life_cycles"].isna()
        valid_life = fatigue_life.notna() & np.isfinite(fatigue_life) & (
            fatigue_life > 0
        )

        result.loc[missing_log & valid_life, "log10_fatigue_life_cycles"] = np.log10(
            fatigue_life.loc[missing_log & valid_life]
        )

    return result


def count_non_missing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]

    if not available:
        return pd.Series(0, index=df.index)

    return df[available].notna().sum(axis=1)


def normalise_group_key(df: pd.DataFrame) -> pd.Series:
    if "dataset_id" in df.columns:
        dataset_id = df["dataset_id"].astype("string").str.strip()
    else:
        dataset_id = pd.Series(pd.NA, index=df.index, dtype="string")

    if "source_id" in df.columns:
        source_id = df["source_id"].astype("string").str.strip()
    else:
        source_id = pd.Series("unknown_source", index=df.index, dtype="string")

    fallback = pd.Series(
        [f"row_{i:06d}" for i in range(len(df))],
        index=df.index,
        dtype="string",
    )

    dataset_id = dataset_id.fillna("").replace("", pd.NA)
    dataset_id = dataset_id.fillna(fallback)

    return source_id.fillna("unknown_source") + "::" + dataset_id


def normalise_split_group_key(df: pd.DataFrame) -> pd.Series:
    """Keep records from the same paper or experimental dataset in one split."""
    source_id = df.get(
        "source_id",
        pd.Series("unknown_source", index=df.index, dtype="string"),
    ).astype("string")
    source_id = source_id.fillna("unknown_source").str.strip()

    dataset_id = df.get(
        "dataset_id",
        pd.Series(pd.NA, index=df.index, dtype="string"),
    ).astype("string")
    dataset_id = dataset_id.str.strip().replace("", pd.NA)

    return source_id + "::" + dataset_id.fillna(source_id)


def build_static_target_view(
    master_df: pd.DataFrame,
    model_key: str,
    target_columns: list[str],
) -> pd.DataFrame:
    """
    Build one static mechanical-property modelling view.

    This view keeps one best row per source_id + dataset_id to avoid duplicated
    parameter rows from fatigue curve joins.
    """
    df = coerce_numeric_columns(master_df)

    task_type = df.get("task_type", pd.Series("", index=df.index)).astype(
        "string"
    ).str.lower()

    source_sheet = df.get("source_sheet", pd.Series("", index=df.index)).astype(
        "string"
    ).str.lower()

    is_static = (
        task_type.isin(
            [
                "static_tensile",
                "parameter_static",
                "model1_static",
                "model1_uts",
                "model3_elongation_yield",
                "model4_elastic_modulus",
            ]
        )
        | source_sheet.str.contains("parameter", na=False)
    )

    static_df = df.loc[is_static].copy()

    target_non_missing = count_non_missing(static_df, target_columns)
    static_df = static_df.loc[target_non_missing > 0].copy()

    if static_df.empty:
        static_df["sample_weight"] = pd.Series(dtype="float64")
        return ensure_view_columns(static_df)

    static_df["_group_key"] = normalise_group_key(static_df)
    static_df["_information_score"] = count_non_missing(
        static_df,
        MODEL1_FEATURE_COLUMNS + target_columns,
    )

    static_df = static_df.sort_values(
        by=["_group_key", "_information_score"],
        ascending=[True, False],
    )

    static_df = static_df.drop_duplicates(subset=["_group_key"], keep="first")
    static_df = static_df.drop(columns=["_group_key", "_information_score"])

    static_df["task_type"] = model_key
    static_df["sample_weight"] = 1.0
    static_df["modelling_group_id"] = normalise_split_group_key(static_df)

    return ensure_view_columns(static_df)


def build_model1_uts_view(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Model 1:
    AM material/process/porosity/surface variables -> UTS.
    """
    return build_static_target_view(
        master_df=master_df,
        model_key="model1_uts",
        target_columns=MODEL1_TARGET_COLUMNS,
    )


def build_model3_elongation_yield_view(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Model 3:
    AM material/process/porosity/surface variables -> elongation and yield.

    Qualitative fracture evidence is intentionally excluded from modelling
    views because public labels are sparse, inconsistent, and not expert
    ground truth.
    """
    return build_static_target_view(
        master_df=master_df,
        model_key="model3_elongation_yield",
        target_columns=MODEL3_TARGET_COLUMNS,
    )


def build_model4_elastic_modulus_view(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Model 4:
    AM material/process/porosity/surface variables -> Young's/elastic modulus.
    """
    return build_static_target_view(
        master_df=master_df,
        model_key="model4_elastic_modulus",
        target_columns=MODEL4_TARGET_COLUMNS,
    )


def select_evenly_spaced_rows(
    group: pd.DataFrame,
    max_rows: int,
    sort_candidates: list[str],
) -> pd.DataFrame:
    if len(group) <= max_rows:
        return group

    sort_column = None

    for col in sort_candidates:
        if col not in group.columns:
            continue

        values = pd.to_numeric(group[col], errors="coerce")

        if values.notna().sum() > 0:
            sort_column = col
            break

    if sort_column is not None:
        sorted_group = group.copy()
        sorted_group[sort_column] = pd.to_numeric(
            sorted_group[sort_column],
            errors="coerce",
        )
        sorted_group = sorted_group.sort_values(by=sort_column)
    else:
        sorted_group = group.copy()

    indices = np.linspace(0, len(sorted_group) - 1, max_rows)
    indices = np.unique(np.round(indices).astype(int))

    return sorted_group.iloc[indices].copy()


def cap_rows_per_dataset_id(
    df: pd.DataFrame,
    max_rows_per_dataset_id: int = 10,
) -> pd.DataFrame:
    if df.empty:
        return df

    working_df = df.copy()
    working_df["_group_key"] = normalise_group_key(working_df)

    result_parts = []

    for _, group in working_df.groupby("_group_key", dropna=False):
        selected = select_evenly_spaced_rows(
            group=group,
            max_rows=max_rows_per_dataset_id,
            sort_candidates=[
                "stress_amplitude_MPa",
                "max_stress_MPa",
                "log10_fatigue_life_cycles",
                "fatigue_life_cycles",
            ],
        )
        result_parts.append(selected)

    result = pd.concat(result_parts, ignore_index=True, sort=False)
    result = result.drop(columns=["_group_key"])

    return result


def add_equal_dataset_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each dataset_id gets approximately equal total influence.

    If one dataset_id has 10 S-N points and another has 2 points, this prevents
    the 10-point dataset from dominating only because it has more curve points.
    """
    if df.empty:
        df["sample_weight"] = pd.Series(dtype="float64")
        return df

    result = df.copy()
    group_key = normalise_group_key(result)

    group_sizes = group_key.map(group_key.value_counts()).astype(float)
    weights = 1.0 / group_sizes

    mean_weight = weights.mean()

    if mean_weight > 0:
        weights = weights / mean_weight

    result["sample_weight"] = weights

    return result


def build_model2_sn_fatigue_view(
    master_df: pd.DataFrame,
    max_rows_per_dataset_id: int = 10,
) -> pd.DataFrame:
    """
    Model 2:
    AM material/process/static-property/loading variables
    -> log10 fatigue life.

    This view only uses S-N fatigue data. It limits the number of points per
    dataset_id and uses sample_weight to reduce overfitting from repeated
    parameter information.
    """
    df = coerce_numeric_columns(master_df)

    task_type = df.get("task_type", pd.Series("", index=df.index)).astype(
        "string"
    ).str.lower()

    source_sheet = df.get("source_sheet", pd.Series("", index=df.index)).astype(
        "string"
    ).str.lower()

    is_sn = (
        task_type.eq("sn_fatigue")
        | source_sheet.str.contains("s-n", na=False)
        | source_sheet.str.fullmatch("sn", na=False)
    )

    sn_df = df.loc[is_sn].copy()

    if sn_df.empty:
        sn_df["sample_weight"] = pd.Series(dtype="float64")
        return ensure_view_columns(sn_df)

    sn_df = coerce_numeric_columns(sn_df)

    has_life = sn_df["log10_fatigue_life_cycles"].notna()
    has_loading = (
        sn_df.get("stress_amplitude_MPa", pd.Series(np.nan, index=sn_df.index)).notna()
        | sn_df.get("max_stress_MPa", pd.Series(np.nan, index=sn_df.index)).notna()
    )

    sn_df = sn_df.loc[has_life & has_loading].copy()

    if sn_df.empty:
        sn_df["sample_weight"] = pd.Series(dtype="float64")
        return ensure_view_columns(sn_df)

    sn_df = cap_rows_per_dataset_id(
        sn_df,
        max_rows_per_dataset_id=max_rows_per_dataset_id,
    )

    sn_df = add_equal_dataset_weights(sn_df)
    sn_df["task_type"] = "model2_sn_fatigue"
    sn_df["modelling_group_id"] = normalise_split_group_key(sn_df)

    return ensure_view_columns(sn_df)


def build_model_views(
    master_path: str | Path | None = None,
    max_sn_rows_per_dataset_id: int = 10,
) -> dict[str, pd.DataFrame]:
    master_df = read_master_dataset(master_path)

    model1_df = build_model1_uts_view(master_df)
    model2_df = build_model2_sn_fatigue_view(
        master_df,
        max_rows_per_dataset_id=max_sn_rows_per_dataset_id,
    )
    model3_df = build_model3_elongation_yield_view(master_df)
    model4_df = build_model4_elastic_modulus_view(master_df)

    return {
        "model1_uts": model1_df,
        "model2_sn_fatigue": model2_df,
        "model3_elongation_yield": model3_df,
        "model4_elastic_modulus": model4_df,
    }


def make_view_summary(views: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for view_name, df in views.items():
        row = {
            "view_name": view_name,
            "rows": len(df),
            "unique_dataset_id": df["dataset_id"].nunique(dropna=True)
            if "dataset_id" in df.columns
            else 0,
            "source_count": df["source_id"].nunique(dropna=True)
            if "source_id" in df.columns
            else 0,
            "modelling_group_count": df["modelling_group_id"].nunique(dropna=True)
            if "modelling_group_id" in df.columns
            else 0,
        }

        for target in [
            "yield_strength_MPa",
            "uts_MPa",
            "elongation_percent",
            "youngs_modulus_GPa",
            "hardness_HV",
            "log10_fatigue_life_cycles",
            "fatigue_life_cycles",
            "runout",
        ]:
            if target in df.columns:
                row[f"{target}_non_missing"] = df[target].notna().sum()

        rows.append(row)

    return pd.DataFrame(rows)


def save_modelling_views(
    master_path: str | Path | None = None,
    max_sn_rows_per_dataset_id: int = 10,
) -> tuple[dict[str, Path], pd.DataFrame]:
    processed_dir = get_path("data", "processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    views = build_model_views(
        master_path=master_path,
        max_sn_rows_per_dataset_id=max_sn_rows_per_dataset_id,
    )

    output_paths = {
        "model1_uts": processed_dir / "view_model1_uts.csv",
        "model2_sn_fatigue": processed_dir / "view_model2_sn_fatigue.csv",
        "model3_elongation_yield": processed_dir
        / "view_model3_elongation_yield.csv",
        "model4_elastic_modulus": processed_dir
        / "view_model4_elastic_modulus.csv",
    }

    for view_name, output_path in output_paths.items():
        views[view_name].to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_df = make_view_summary(views)
    summary_path = processed_dir / "model_view_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    output_paths["summary"] = summary_path

    return output_paths, summary_df
