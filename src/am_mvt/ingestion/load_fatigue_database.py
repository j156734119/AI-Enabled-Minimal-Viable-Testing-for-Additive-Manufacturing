from __future__ import annotations

from pathlib import Path

import pandas as pd

from am_mvt.cleaning.project_schema import (
    MASTER_COLUMNS,
    build_column_mapping,
    build_column_mapping_report,
    make_unique_columns,
    standardise_table_to_project_schema,
)
from am_mvt.config import get_path


FATIGUE_SOURCE_ID = "fatigue_am_alloys_figshare_2023"
FATIGUE_SOURCE_NAME = "Fatigue Database of Additively Manufactured Alloys"


def find_fatigue_excel(root_dir: str | Path | None = None) -> Path:
    if root_dir is None:
        root_dir = get_path("data", "raw", "open_datasets", "fatigue_database")
    else:
        root_dir = Path(root_dir)

    candidates = sorted(root_dir.rglob("*.xlsx")) + sorted(root_dir.rglob("*.xls"))

    if not candidates:
        raise FileNotFoundError(f"No Fatigue Database Excel found under: {root_dir}")

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


def read_all_sheets(file_path: Path) -> dict[str, pd.DataFrame]:
    raw_sheets = pd.read_excel(
        file_path,
        sheet_name=None,
        header=None,
        dtype=object,
    )

    parsed_sheets: dict[str, pd.DataFrame] = {}

    for sheet_name, raw_sheet_df in raw_sheets.items():
        parsed = parse_sheet(raw_sheet_df)

        if not parsed.empty:
            parsed_sheets[str(sheet_name)] = parsed

    return parsed_sheets


def infer_task_type(sheet_name: str) -> str:
    lower = sheet_name.lower().strip()

    if "parameter" in lower:
        return "parameter_static"

    if lower in {"sn", "s-n"} or "sn" in lower or "s-n" in lower:
        return "sn_fatigue"

    if lower in {"en", "e-n"} or "en" in lower or "strain" in lower:
        return "strain_life_fatigue"

    if "dadn" in lower or "da/dn" in lower or "crack" in lower:
        return "crack_growth"

    return f"fatigue_{lower}"


def add_source_fields(
    df: pd.DataFrame,
    *,
    file_path: Path,
    sheet_name: str,
    task_type: str,
) -> pd.DataFrame:
    result = df.copy()

    result["source_id"] = FATIGUE_SOURCE_ID
    result["source_name"] = FATIGUE_SOURCE_NAME
    result["source_file"] = str(file_path)
    result["source_sheet"] = sheet_name
    result["task_type"] = task_type
    result["extraction_method"] = "existing_open_dataset"
    result["needs_human_check"] = False

    return result


def standardise_sheet(
    raw_df: pd.DataFrame,
    *,
    file_path: Path,
    sheet_name: str,
) -> pd.DataFrame:
    task_type = infer_task_type(sheet_name)

    with_source = add_source_fields(
        raw_df,
        file_path=file_path,
        sheet_name=sheet_name,
        task_type=task_type,
    )

    standard = standardise_table_to_project_schema(with_source)

    if "dataset_id" in standard.columns:
        standard["dataset_id"] = standard["dataset_id"].astype("string").str.strip()

    return standard


def find_parameter_sheet_name(sheets: dict[str, pd.DataFrame]) -> str:
    for sheet_name in sheets:
        if "parameter" in sheet_name.lower():
            return sheet_name

    raise ValueError(
        "Could not find the parameter sheet. "
        "Expected one sheet name to contain 'parameter'."
    )


def prepare_parameter_context(parameter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one contextual row per dataset_id.

    The parameter sheet is the parent table. It should provide material,
    process parameters, static mechanical properties, and source information.
    """
    result = parameter_df.copy()

    if "dataset_id" not in result.columns:
        raise ValueError("The parameter sheet does not contain dataset_id.")

    result["dataset_id"] = result["dataset_id"].astype("string").str.strip()

    result = result.dropna(subset=["dataset_id"])
    result = result.loc[result["dataset_id"] != ""]

    result = result.drop_duplicates(subset=["dataset_id"], keep="first")

    return result


def coalesce_parameter_and_result_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """
    After joining result rows to parameter rows, combine columns.

    Rule:
    - result-specific values have priority
    - missing result values are filled from parameter values
    - source_sheet and task_type should describe the result sheet
    """
    output = pd.DataFrame(index=merged.index)

    for col in MASTER_COLUMNS:
        result_col = f"{col}_result"
        parameter_col = f"{col}_parameter"

        if result_col in merged.columns and parameter_col in merged.columns:
            output[col] = merged[result_col].combine_first(merged[parameter_col])
        elif result_col in merged.columns:
            output[col] = merged[result_col]
        elif parameter_col in merged.columns:
            output[col] = merged[parameter_col]
        elif col in merged.columns:
            output[col] = merged[col]
        else:
            output[col] = pd.NA

    return output[MASTER_COLUMNS]


def join_child_sheet_to_parameter(
    child_df: pd.DataFrame,
    parameter_context: pd.DataFrame,
    *,
    child_sheet_name: str,
) -> pd.DataFrame:
    """
    Join one child/result sheet to the parameter sheet.

    Example:
        parameter dataset_id = 1
        sn rows for dataset_id = 1: 1a, 1b, 1c

    Output:
        1 + 1a
        1 + 1b
        1 + 1c
    """
    child = child_df.copy()

    if "dataset_id" not in child.columns:
        return child

    child["dataset_id"] = child["dataset_id"].astype("string").str.strip()
    child = child.dropna(subset=["dataset_id"])
    child = child.loc[child["dataset_id"] != ""]

    if child.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    merged = child.merge(
        parameter_context,
        on="dataset_id",
        how="left",
        suffixes=("_result", "_parameter"),
    )

    joined = coalesce_parameter_and_result_columns(merged)

    joined["source_id"] = FATIGUE_SOURCE_ID
    joined["source_name"] = FATIGUE_SOURCE_NAME
    joined["source_sheet"] = child_sheet_name
    joined["task_type"] = infer_task_type(child_sheet_name)
    joined["extraction_method"] = "existing_open_dataset"
    joined["needs_human_check"] = False

    return joined[MASTER_COLUMNS]


def build_parameter_static_rows(parameter_context: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one static/tensile row per dataset_id from the parameter sheet.

    These rows are useful for training tensile/static property models.
    """
    static_rows = parameter_context.copy()

    static_rows["task_type"] = "static_tensile"
    static_rows["source_sheet"] = "parameter"

    return static_rows[MASTER_COLUMNS]


def make_unique_record_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    dataset_id = result["dataset_id"].astype("string").fillna("missing_dataset")
    task_type = result["task_type"].astype("string").fillna("unknown_task")
    source_sheet = result["source_sheet"].astype("string").fillna("unknown_sheet")

    result["record_id"] = [
        f"{task}_{sheet}_dataset_{ds}_row_{i:06d}"
        for i, (task, sheet, ds) in enumerate(
            zip(task_type, source_sheet, dataset_id, strict=False),
            start=1,
        )
    ]

    return result


def load_fatigue_database_master_rows(
    file_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if file_path is None:
        file_path = find_fatigue_excel()
    else:
        file_path = Path(file_path)

    raw_sheets = read_all_sheets(file_path)

    if not raw_sheets:
        raise ValueError(f"No readable sheets found in {file_path}")

    parameter_sheet_name = find_parameter_sheet_name(raw_sheets)

    standard_sheets: dict[str, pd.DataFrame] = {}
    report_rows: list[dict[str, object]] = []

    for sheet_name, raw_df in raw_sheets.items():
        standard = standardise_sheet(
            raw_df,
            file_path=file_path,
            sheet_name=sheet_name,
        )

        standard_sheets[sheet_name] = standard

        report_rows.append(
            {
                "source_id": FATIGUE_SOURCE_ID,
                "file_path": str(file_path),
                "sheet_name": sheet_name,
                "rows": len(standard),
                "mapped_columns": len(build_column_mapping(raw_df.columns)),
                "task_type": infer_task_type(sheet_name),
            }
        )

    parameter_context = prepare_parameter_context(
        standard_sheets[parameter_sheet_name]
    )

    master_parts: list[pd.DataFrame] = []

    # Keep static/tensile rows from parameter.
    master_parts.append(build_parameter_static_rows(parameter_context))

    # Join every non-parameter sheet to parameter by dataset_id.
    for sheet_name, child_df in standard_sheets.items():
        if sheet_name == parameter_sheet_name:
            continue

        joined = join_child_sheet_to_parameter(
            child_df,
            parameter_context,
            child_sheet_name=sheet_name,
        )

        if not joined.empty:
            master_parts.append(joined)

    master_df = pd.concat(master_parts, ignore_index=True, sort=False)
    master_df = make_unique_record_ids(master_df)

    report_df = pd.DataFrame(report_rows)

    return master_df[MASTER_COLUMNS], report_df


def save_fatigue_debug_outputs() -> tuple[Path, Path, Path]:
    master_df, report_df = load_fatigue_database_master_rows()

    processed_dir = get_path("data", "processed")
    interim_dir = get_path("data", "interim")
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    master_path = processed_dir / "fatigue_database_master_rows.csv"
    report_path = interim_dir / "fatigue_database_load_report.csv"
    mapping_path = interim_dir / "fatigue_database_column_mapping_report.csv"

    master_df.to_csv(master_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    file_path = find_fatigue_excel()
    raw_sheets = read_all_sheets(file_path)

    mapping_reports = []

    for sheet_name, raw_df in raw_sheets.items():
        mapping_report = build_column_mapping_report(raw_df.columns)
        mapping_report.insert(0, "sheet_name", sheet_name)
        mapping_reports.append(mapping_report)

    if mapping_reports:
        pd.concat(mapping_reports, ignore_index=True).to_csv(
            mapping_path,
            index=False,
            encoding="utf-8-sig",
        )

    return master_path, report_path, mapping_path