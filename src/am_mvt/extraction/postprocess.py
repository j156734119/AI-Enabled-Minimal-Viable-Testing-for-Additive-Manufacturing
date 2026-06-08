from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from am_mvt.cleaning.project_schema import MASTER_COLUMNS, standardise_table_to_project_schema
from am_mvt.config import get_path
from am_mvt.utils.values import is_missing, parse_boolean


LLM_AUDIT_EXTRA_COLUMNS = [
    "source_title",
    "doi",
    "page_or_section",
    "evidence_text",
    "confidence",
]


AUDIT_COLUMNS = [
    "source_id",
    "source_file",
    "source_sheet",
    "record_id",
    "doi",
    "source_title",
    "page_or_section",
    "evidence_text",
    "confidence",
    "needs_human_check",
]


def get_llm_output_columns(df: pd.DataFrame) -> list[str]:
    """
    Build output columns for LLM extracted records.

    MASTER_COLUMNS are the core modelling columns.
    LLM_AUDIT_EXTRA_COLUMNS are kept for traceability and human checking.
    """
    columns = list(MASTER_COLUMNS)

    for col in LLM_AUDIT_EXTRA_COLUMNS:
        if col in df.columns and col not in columns:
            columns.append(col)

    return columns


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_task_type(record: dict[str, Any]) -> str:
    """
    Infer the modelling task type from the extracted record.
    """
    if record.get("da_dN_m_per_cycle") is not None or record.get("delta_K_MPa_sqrt_m") is not None:
        return "crack_growth"

    if record.get("fatigue_life_cycles") is not None:
        return "sn_fatigue"

    if record.get("stress_amplitude_MPa") is not None or record.get("max_stress_MPa") is not None:
        return "sn_fatigue"

    if (
        record.get("uts_MPa") is not None
        or record.get("yield_strength_MPa") is not None
        or record.get("elongation_percent") is not None
        or record.get("youngs_modulus_GPa") is not None
        or record.get("hardness_HV") is not None
    ):
        return "static_tensile"

    return "unknown_llm_extraction"


def normalise_runout(value: Any) -> Any:
    """
    Convert common runout values into booleans where possible.
    """
    if is_missing(value):
        return pd.NA

    parsed = parse_boolean(value)
    return value if parsed is None else parsed


def normalise_pdf_filename(value: Any) -> str:
    filename = Path(str(value or "")).name

    while filename.lower().endswith(".pdf.pdf"):
        filename = filename[:-4]

    return filename


def source_id_from_pdf_filename(value: Any) -> str:
    filename = normalise_pdf_filename(value)
    stem = filename[:-4] if filename.lower().endswith(".pdf") else filename
    source_id = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return source_id or "unknown_literature_source"


def extract_records_from_llm_json(json_path: Path) -> list[dict[str, Any]]:
    """
    Load one LLM JSON output file and convert it into flat record dictionaries.
    """
    data = load_json_file(json_path)
    records = data.get("records", [])

    if not isinstance(records, list):
        return []

    metadata = data.get("_metadata", {})
    chunk_id = metadata.get("chunk_id", json_path.stem)
    source_file = normalise_pdf_filename(metadata.get("source_file", ""))
    source_id = source_id_from_pdf_filename(source_file)

    normalised_records: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue

        output_record = record.copy()

        output_record["source_id"] = source_id
        output_record["source_name"] = (
            output_record.get("source_title")
            or source_id.replace("_", " ")
        )
        output_record["source_file"] = normalise_pdf_filename(
            output_record.get("source_file") or source_file
        )
        output_record["source_sheet"] = chunk_id
        output_record["record_id"] = f"{chunk_id}_record_{index:04d}"
        output_record["source_url"] = pd.NA
        output_record["extraction_method"] = "llm_extraction"
        output_record["task_type"] = infer_task_type(output_record)

        if output_record.get("needs_human_check") is None:
            output_record["needs_human_check"] = True

        if output_record.get("confidence") is None:
            output_record["confidence"] = pd.NA

        output_record["runout"] = normalise_runout(output_record.get("runout"))

        normalised_records.append(output_record)

    return normalised_records


def load_llm_json_outputs(
    llm_output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read all LLM JSON outputs and return a standardised DataFrame.

    This function preserves audit fields such as evidence_text and confidence.
    """
    if llm_output_dir is None:
        llm_output_dir = get_path("data", "interim", "llm_outputs")
    else:
        llm_output_dir = Path(llm_output_dir)

    llm_output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(llm_output_dir.glob("*.json"))

    if not json_files:
        empty_columns = list(MASTER_COLUMNS)

        for col in LLM_AUDIT_EXTRA_COLUMNS:
            if col not in empty_columns:
                empty_columns.append(col)

        return pd.DataFrame(columns=empty_columns)

    all_records: list[dict[str, Any]] = []

    for json_path in json_files:
        try:
            all_records.extend(extract_records_from_llm_json(json_path))
        except Exception as exc:
            all_records.append(
                {
                    "source_id": "llm_literature_extraction",
                    "source_name": "LLM-assisted open-access literature extraction",
                    "source_file": str(json_path),
                    "source_sheet": json_path.stem,
                    "record_id": f"{json_path.stem}_error",
                    "extraction_method": "llm_extraction",
                    "task_type": "llm_parse_error",
                    "needs_human_check": True,
                    "evidence_text": f"Failed to parse JSON file: {exc}",
                    "confidence": pd.NA,
                }
            )

    if not all_records:
        empty_columns = list(MASTER_COLUMNS)

        for col in LLM_AUDIT_EXTRA_COLUMNS:
            if col not in empty_columns:
                empty_columns.append(col)

        return pd.DataFrame(columns=empty_columns)

    raw_df = pd.DataFrame(all_records).reset_index(drop=True)

    evidence_columns = [col for col in AUDIT_COLUMNS if col in raw_df.columns]
    audit_df = raw_df[evidence_columns].copy()

    audit_path = get_path("data", "interim", "llm_extraction_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    standardised_df = standardise_table_to_project_schema(raw_df).reset_index(drop=True)

    for col in LLM_AUDIT_EXTRA_COLUMNS:
        if col in raw_df.columns:
            standardised_df[col] = raw_df[col].reset_index(drop=True)
        else:
            standardised_df[col] = pd.NA

    return standardised_df


def save_llm_extracted_records(
    llm_output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """
    Save all post-processed LLM extracted records to CSV.
    """
    if output_path is None:
        output_path = get_path("data", "interim", "llm_extracted_records.csv")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extracted_df = load_llm_json_outputs(llm_output_dir)

    for col in MASTER_COLUMNS:
        if col not in extracted_df.columns:
            extracted_df[col] = pd.NA

    for col in LLM_AUDIT_EXTRA_COLUMNS:
        if col not in extracted_df.columns:
            extracted_df[col] = pd.NA

    output_columns = get_llm_output_columns(extracted_df)
    extracted_df = extracted_df[output_columns]

    extracted_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path
