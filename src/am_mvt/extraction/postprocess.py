from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from am_mvt.cleaning.project_schema import MASTER_COLUMNS, standardise_table_to_project_schema
from am_mvt.config import get_path


def flatten_dict(
    data: dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            flattened.update(
                flatten_dict(
                    value,
                    parent_key=new_key,
                    separator=separator,
                )
            )
        else:
            flattened[new_key] = value

    return flattened


def normalise_extracted_json(data: Any, source_file: Path) -> list[dict[str, Any]]:
    """
    Convert one LLM JSON output file into a list of flat records.

    This function is intentionally permissive because LLM extraction outputs
    may be structured as:
    - a list of records
    - a dict containing "records"
    - a single dict record
    """
    records: list[dict[str, Any]] = []

    if isinstance(data, list):
        raw_records = data
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        raw_records = data["records"]
    elif isinstance(data, dict) and isinstance(data.get("extracted_records"), list):
        raw_records = data["extracted_records"]
    elif isinstance(data, dict):
        raw_records = [data]
    else:
        raw_records = []

    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, dict):
            continue

        flattened = flatten_dict(raw_record)

        flattened["source_id"] = "llm_literature_extraction"
        flattened["source_name"] = "LLM-assisted literature extraction"
        flattened["source_file"] = str(source_file)
        flattened["source_sheet"] = "llm_json"
        flattened["record_id"] = f"{source_file.stem}_{index:04d}"
        flattened["extraction_method"] = "llm_extraction"
        flattened["needs_human_check"] = True

        records.append(flattened)

    return records


def load_llm_json_outputs(
    llm_output_dir: str | Path | None = None,
) -> pd.DataFrame:
    if llm_output_dir is None:
        llm_output_dir = get_path("data", "interim", "llm_outputs")
    else:
        llm_output_dir = Path(llm_output_dir)

    llm_output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(llm_output_dir.glob("*.json"))

    if not json_files:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    all_records: list[dict[str, Any]] = []

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            records = normalise_extracted_json(data, source_file=json_path)
            all_records.extend(records)

        except Exception as exc:
            all_records.append(
                {
                    "source_id": "llm_literature_extraction",
                    "source_name": "LLM-assisted literature extraction",
                    "source_file": str(json_path),
                    "source_sheet": "llm_json",
                    "record_id": f"{json_path.stem}_error",
                    "extraction_method": "llm_extraction",
                    "needs_human_check": True,
                    "evidence_text": f"Failed to parse JSON file: {exc}",
                }
            )

    if not all_records:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    raw_df = pd.DataFrame(all_records)
    standardised_df = standardise_table_to_project_schema(raw_df)

    return standardised_df


def save_llm_extracted_records(
    llm_output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    if output_path is None:
        output_path = get_path("data", "interim", "llm_extracted_records.csv")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extracted_df = load_llm_json_outputs(llm_output_dir)

    for col in MASTER_COLUMNS:
        if col not in extracted_df.columns:
            extracted_df[col] = pd.NA

    extracted_df = extracted_df[MASTER_COLUMNS]

    extracted_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path