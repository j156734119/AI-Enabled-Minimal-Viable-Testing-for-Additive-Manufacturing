from __future__ import annotations

from pathlib import Path

import pandas as pd

from am_mvt.cleaning.project_schema import (
    MASTER_COLUMNS,
    build_column_mapping,
    make_unique_columns,
    standardise_table_to_project_schema,
)
from am_mvt.config import get_path


MATERIALS_SOURCE_ID = "materials_design_statistical_assessment_2025"
MATERIALS_SOURCE_NAME = "Critical statistical assessment of data in metal additive manufacturing"


def find_materials_design_excel(root_dir: str | Path | None = None) -> Path:
    if root_dir is None:
        root_dir = get_path("data", "raw", "open_datasets", "materials_design_2025")
    else:
        root_dir = Path(root_dir)

    candidates = sorted(root_dir.rglob("*.xlsx")) + sorted(root_dir.rglob("*.xls"))

    if not candidates:
        raise FileNotFoundError(
            f"No Materials Design Excel file found under: {root_dir}"
        )

    return candidates[0]


def score_header_row(row_values: list[object]) -> int:
    columns = make_unique_columns(row_values)
    mapping = build_column_mapping(columns)

    non_empty = sum(
        not pd.isna(value) and str(value).strip() != "" for value in row_values
    )

    return len(mapping) * 100 + min(non_empty, 50)


def detect_header_row(raw_df: pd.DataFrame, max_rows: int = 20) -> int:
    raw_df = raw_df.dropna(how="all").reset_index(drop=True)

    best_row = 0
    best_score = -1

    for row_index in range(min(max_rows, len(raw_df))):
        score = score_header_row(raw_df.iloc[row_index].tolist())

        if score > best_score:
            best_score = score
            best_row = row_index

    return best_row


def parse_sheet(raw_sheet_df: pd.DataFrame) -> pd.DataFrame:
    raw_sheet_df = raw_sheet_df.dropna(how="all").reset_index(drop=True)

    if raw_sheet_df.empty:
        return pd.DataFrame()

    header_row = detect_header_row(raw_sheet_df)
    columns = make_unique_columns(raw_sheet_df.iloc[header_row].tolist())

    data = raw_sheet_df.iloc[header_row + 1 :].copy()
    data.columns = columns
    data = data.dropna(how="all").reset_index(drop=True)

    return data


def load_materials_design_master_rows(
    file_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if file_path is None:
        file_path = find_materials_design_excel()
    else:
        file_path = Path(file_path)

    raw_sheets = pd.read_excel(
        file_path,
        sheet_name=None,
        header=None,
        dtype=object,
    )

    master_parts: list[pd.DataFrame] = []
    report_rows = []

    for sheet_name, raw_sheet_df in raw_sheets.items():
        parsed = parse_sheet(raw_sheet_df)

        if parsed.empty:
            continue

        parsed["source_id"] = MATERIALS_SOURCE_ID
        parsed["source_name"] = MATERIALS_SOURCE_NAME
        parsed["source_file"] = str(file_path)
        parsed["source_sheet"] = str(sheet_name)
        parsed["task_type"] = "static_tensile"
        parsed["extraction_method"] = "existing_open_dataset"
        parsed["needs_human_check"] = False

        standard = standardise_table_to_project_schema(parsed)

        # Materials Design does not use Fatigue Database dataset_id.
        if standard["dataset_id"].isna().all():
            standard["dataset_id"] = [
                f"materials_design_{i:06d}" for i in range(1, len(standard) + 1)
            ]

        standard["record_id"] = [
            f"materials_static_{sheet_name}_{i:06d}"
            for i in range(1, len(standard) + 1)
        ]

        master_parts.append(standard[MASTER_COLUMNS])

        report_rows.append(
            {
                "source_id": MATERIALS_SOURCE_ID,
                "file_path": str(file_path),
                "sheet_name": sheet_name,
                "rows": len(standard),
                "mapped_columns": len(build_column_mapping(parsed.columns)),
                "task_type": "static_tensile",
            }
        )

    if master_parts:
        master_df = pd.concat(master_parts, ignore_index=True, sort=False)
    else:
        master_df = pd.DataFrame(columns=MASTER_COLUMNS)

    return master_df[MASTER_COLUMNS], pd.DataFrame(report_rows)


def save_materials_design_debug_outputs() -> tuple[Path, Path]:
    master_df, report_df = load_materials_design_master_rows()

    processed_dir = get_path("data", "processed")
    interim_dir = get_path("data", "interim")
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    master_path = processed_dir / "materials_design_master_rows.csv"
    report_path = interim_dir / "materials_design_load_report.csv"

    master_df.to_csv(master_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    return master_path, report_path