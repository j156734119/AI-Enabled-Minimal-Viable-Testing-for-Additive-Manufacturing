from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from am_mvt.cleaning.schema_mapping import ensure_standard_columns
from am_mvt.config import get_path


def load_llm_json_outputs(output_dir: str | Path | None = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = get_path("data", "interim", "llm_outputs")
    else:
        output_dir = Path(output_dir)

    rows: list[dict[str, object]] = []

    if not output_dir.exists():
        return ensure_standard_columns(pd.DataFrame())

    for json_file in sorted(output_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        records = data.get("records", [])

        for index, record in enumerate(records):
            row = dict(record)
            row["record_id"] = row.get("record_id") or f"{json_file.stem}_{index:03d}"
            row["extraction_method"] = "llm_extraction"
            rows.append(row)

    df = pd.DataFrame(rows)

    return ensure_standard_columns(df)


def save_llm_extracted_records() -> Path:
    df = load_llm_json_outputs()

    output_path = get_path("data", "interim", "llm_extracted_records.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path