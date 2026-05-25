from __future__ import annotations

import json
from pathlib import Path

from am_mvt.config import get_path
from am_mvt.extraction.openai_extractor import DEFAULT_MODEL, extract_records_from_chunk


def infer_source_pdf_from_chunk_name(chunk_path: Path) -> str:
    name = chunk_path.stem

    if "_chunk_" in name:
        return name.split("_chunk_")[0] + ".pdf"

    return name + ".pdf"


def run_batch_extraction(
    limit: int | None = 20,
    overwrite: bool = False,
    model: str = DEFAULT_MODEL,
) -> list[Path]:
    chunk_dir = get_path("data", "interim", "text_chunks")
    output_dir = get_path("data", "interim", "llm_outputs")

    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(chunk_dir.glob("*.txt"))

    if limit is not None and limit > 0:
        chunk_files = chunk_files[:limit]

    output_paths: list[Path] = []

    if not chunk_files:
        print("No text chunks found for LLM extraction.")
        return output_paths

    for index, chunk_path in enumerate(chunk_files, start=1):
        output_path = output_dir / f"{chunk_path.stem}.json"

        if output_path.exists() and not overwrite:
            print(f"[{index}/{len(chunk_files)}] Skipping existing output: {output_path.name}")
            output_paths.append(output_path)
            continue

        print(f"[{index}/{len(chunk_files)}] Extracting: {chunk_path.name}")

        chunk_text = chunk_path.read_text(encoding="utf-8", errors="ignore")
        source_pdf = infer_source_pdf_from_chunk_name(chunk_path)

        result = extract_records_from_chunk(
            chunk_text=chunk_text,
            source_file=source_pdf,
            chunk_id=chunk_path.stem,
            model=model,
        )

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        output_paths.append(output_path)

        record_count = len(result.get("records", []))
        print(f"    Records extracted: {record_count}")

    return output_paths