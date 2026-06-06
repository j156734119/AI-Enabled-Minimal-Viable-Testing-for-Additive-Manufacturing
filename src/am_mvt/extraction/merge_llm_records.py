from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from am_mvt.cleaning.project_schema import MASTER_COLUMNS
from am_mvt.config import get_path


LLM_AUDIT_EXTRA_COLUMNS = [
    "source_title",
    "doi",
    "page_or_section",
    "evidence_text",
    "confidence",
]


def get_core_output_columns() -> list[str]:
    """
    Return master columns plus LLM audit columns.
    """
    columns = list(MASTER_COLUMNS)

    for col in LLM_AUDIT_EXTRA_COLUMNS:
        if col not in columns:
            columns.append(col)

    return columns


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=get_core_output_columns())

    return pd.read_csv(path, low_memory=False)


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataset has all modelling columns and LLM audit columns.

    This function does not drop unknown columns because engineered features or
    future audit fields may also be useful.
    """
    result = df.copy()

    for col in get_core_output_columns():
        if col not in result.columns:
            result[col] = pd.NA

    ordered_columns = get_core_output_columns()
    remaining_columns = [col for col in result.columns if col not in ordered_columns]

    return result[ordered_columns + remaining_columns]


def remove_empty_llm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove LLM rows that contain no useful AM/mechanical testing information.
    """
    if df.empty:
        return df

    useful_columns = [
        "alloy",
        "am_process",
        "yield_strength_MPa",
        "uts_MPa",
        "elongation_percent",
        "youngs_modulus_GPa",
        "hardness_HV",
        "stress_amplitude_MPa",
        "max_stress_MPa",
        "fatigue_life_cycles",
        "porosity_percent",
        "relative_density_percent",
        "defect_type",
        "surface_condition",
        "heat_treatment",
    ]

    available = [col for col in useful_columns if col in df.columns]

    if not available:
        return df.iloc[0:0].copy()

    mask = df[available].notna().any(axis=1)

    return df.loc[mask].copy()


def deduplicate_master_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records using source_id and record_id when available.
    """
    result = df.copy()

    if "source_id" not in result.columns or "record_id" not in result.columns:
        return result.drop_duplicates().reset_index(drop=True)

    result["source_id"] = result["source_id"].astype("string")
    result["record_id"] = result["record_id"].astype("string")

    return result.drop_duplicates(
        subset=["source_id", "record_id"],
        keep="last",
    ).reset_index(drop=True)


def add_engineered_features_if_available(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate engineered features after LLM rows are added.

    For example:
    - VED
    - porosity from relative density
    - log10 fatigue life
    """
    try:
        from am_mvt.cleaning.calculate_features import add_engineered_features
    except Exception:
        return df

    try:
        return add_engineered_features(df)
    except Exception:
        return df


def append_llm_records_to_master(
    master_path: str | Path | None = None,
    llm_csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    make_backup: bool = True,
) -> tuple[Path, pd.DataFrame]:
    """
    Append LLM extracted literature records to the master modelling dataset.
    """
    if master_path is None:
        master_path = get_path("data", "processed", "master_modelling_dataset.csv")
    else:
        master_path = Path(master_path)

    if llm_csv_path is None:
        llm_csv_path = get_path("data", "interim", "llm_extracted_records.csv")
    else:
        llm_csv_path = Path(llm_csv_path)

    if output_path is None:
        output_path = master_path
    else:
        output_path = Path(output_path)

    if not master_path.exists():
        fallback = get_path("data", "processed", "modelling_dataset.csv")

        if fallback.exists():
            master_path = fallback
        else:
            raise FileNotFoundError(
                "No master dataset found. Run python scripts/05_build_dataset.py first."
            )

    master_df = ensure_required_columns(read_csv_if_exists(master_path))
    llm_df = ensure_required_columns(read_csv_if_exists(llm_csv_path))
    llm_df = remove_empty_llm_rows(llm_df)

    if "extraction_method" in master_df.columns:
        is_previous_llm_record = (
            master_df["extraction_method"]
            .astype("string")
            .str.lower()
            .eq("llm_extraction")
        )
        master_df = master_df.loc[~is_previous_llm_record].copy()

    if make_backup and output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_name(f"{output_path.stem}_backup_{timestamp}.csv")
        output_path.replace(backup_path)
        print(f"Backup created: {backup_path}")

    combined_df = pd.concat(
        [master_df, llm_df],
        ignore_index=True,
        sort=False,
    )

    combined_df = ensure_required_columns(combined_df)
    combined_df = add_engineered_features_if_available(combined_df)
    combined_df = ensure_required_columns(combined_df)
    combined_df = deduplicate_master_records(combined_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    if "extraction_method" in combined_df.columns:
        llm_mask = (
            combined_df["extraction_method"]
            .astype("string")
            .str.lower()
            .eq("llm_extraction")
        )
        llm_source_rows_after = int(llm_mask.sum())
        llm_source_count_after = combined_df.loc[
            llm_mask,
            "source_id",
        ].nunique(dropna=True)
    else:
        llm_source_rows_after = 0
        llm_source_count_after = 0

    summary = pd.DataFrame(
        [
            {
                "master_rows_before": len(master_df),
                "llm_rows_added": len(llm_df),
                "master_rows_after": len(combined_df),
                "llm_source_rows_after": llm_source_rows_after,
                "llm_source_count_after": llm_source_count_after,
                "llm_rows_with_evidence_text": (
                    llm_df["evidence_text"].notna().sum()
                    if "evidence_text" in llm_df.columns
                    else 0
                ),
                "llm_rows_with_confidence": (
                    llm_df["confidence"].notna().sum()
                    if "confidence" in llm_df.columns
                    else 0
                ),
            }
        ]
    )

    summary_path = get_path("data", "processed", "llm_merge_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return output_path, summary
