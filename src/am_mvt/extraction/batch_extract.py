from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from am_mvt.config import get_path
from am_mvt.extraction.openai_extractor import (
    DEFAULT_MODEL,
    extract_records_from_chunk,
    extract_records_from_pdf,
)

DEFAULT_SKIP_FILE = Path("config/llm_extraction_skip.txt")


def infer_source_pdf_from_chunk_name(chunk_path: Path) -> str:
    name = chunk_path.stem

    if "_chunk_" in name:
        source_name = name.split("_chunk_")[0]
        return source_name if source_name.lower().endswith(".pdf") else source_name + ".pdf"

    return name if name.lower().endswith(".pdf") else name + ".pdf"


def load_extraction_skip_stems(
    skip_file: str | Path = DEFAULT_SKIP_FILE,
) -> set[str]:
    path = Path(skip_file)
    if not path.exists():
        return set()
    return {
        line.strip().removesuffix(".pdf")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def chunk_is_skipped(chunk_path: Path, skip_stems: set[str]) -> bool:
    source_stem = infer_source_pdf_from_chunk_name(chunk_path).removesuffix(
        ".pdf"
    )
    return source_stem in skip_stems


def output_has_error(output_path: Path) -> bool:
    if not output_path.exists():
        return False

    try:
        payload: dict[str, Any] = json.loads(
            output_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return True

    metadata = payload.get("_metadata", {})
    return bool(metadata.get("error"))


def chunk_has_usable_text(chunk_path: Path) -> bool:
    text = chunk_path.read_text(encoding="utf-8", errors="ignore")
    content = "\n".join(
        line
        for line in text.splitlines()
        if not line.strip().startswith("--- Page ")
    )
    return len(content.strip()) >= 80


def output_needs_pdf_vision(output_path: Path, chunk_path: Path) -> bool:
    chunk_text = chunk_path.read_text(encoding="utf-8", errors="ignore")
    if (
        not output_path.exists()
        or "--- Page " not in chunk_text
        or chunk_has_usable_text(chunk_path)
    ):
        return False
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    metadata = payload.get("_metadata", {})
    return (
        not payload.get("records")
        and metadata.get("input_mode") != "pdf_vision"
    )


def select_chunks_for_extraction(
    chunk_files: list[Path],
    output_dir: Path,
    *,
    limit: int | None,
    overwrite: bool,
) -> tuple[list[Path], int]:
    pending = []
    skipped = 0

    for chunk_path in chunk_files:
        output_path = output_dir / f"{chunk_path.stem}.json"

        if (
            output_path.exists()
            and not overwrite
            and not output_has_error(output_path)
            and not output_needs_pdf_vision(output_path, chunk_path)
        ):
            skipped += 1
            continue

        pending.append(chunk_path)

    if limit is not None and limit > 0:
        pending = pending[:limit]

    return pending, skipped


def write_extraction_result(
    output_path: Path,
    result: dict[str, Any],
) -> bool:
    if result.get("_metadata", {}).get("error") and output_path.exists():
        return False
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


def run_batch_extraction(
    limit: int | None = 20,
    overwrite: bool = False,
    model: str = DEFAULT_MODEL,
) -> list[Path]:
    chunk_dir = get_path("data", "interim", "text_chunks")
    output_dir = get_path("data", "interim", "llm_outputs")
    pdf_dir = get_path("data", "raw", "pdfs")

    chunk_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunk_files = sorted(chunk_dir.glob("*.txt"))
    skip_stems = load_extraction_skip_stems()
    skipped_by_policy = [
        path for path in all_chunk_files if chunk_is_skipped(path, skip_stems)
    ]
    all_chunk_files = [
        path for path in all_chunk_files if not chunk_is_skipped(path, skip_stems)
    ]

    output_paths: list[Path] = []

    if skipped_by_policy:
        print(
            "Chunks skipped by config/llm_extraction_skip.txt: "
            f"{len(skipped_by_policy)}"
        )

    if not all_chunk_files:
        print("No text chunks found for LLM extraction.")
        return output_paths

    chunk_files, skipped = select_chunks_for_extraction(
        all_chunk_files,
        output_dir,
        limit=limit,
        overwrite=overwrite,
    )
    print(f"Existing successful outputs skipped: {skipped}")
    print(f"Pending chunks selected: {len(chunk_files)}")

    if not chunk_files:
        print("No new or failed chunks require API extraction.")
        return output_paths

    for index, chunk_path in enumerate(chunk_files, start=1):
        output_path = output_dir / f"{chunk_path.stem}.json"

        print(f"[{index}/{len(chunk_files)}] Extracting: {chunk_path.name}")

        chunk_text = chunk_path.read_text(encoding="utf-8", errors="ignore")
        source_pdf = infer_source_pdf_from_chunk_name(chunk_path)

        pdf_path = pdf_dir / source_pdf
        if not chunk_has_usable_text(chunk_path) and pdf_path.exists():
            print("    No usable text layer; using PDF vision input.")
            result = extract_records_from_pdf(
                pdf_path=pdf_path,
                source_file=source_pdf,
                chunk_id=chunk_path.stem,
                model=model,
            )
        else:
            result = extract_records_from_chunk(
                chunk_text=chunk_text,
                source_file=source_pdf,
                chunk_id=chunk_path.stem,
                model=model,
            )

        if not write_extraction_result(output_path, result):
            print("    API failed; existing JSON output was preserved.")
            continue

        output_paths.append(output_path)

        record_count = len(result.get("records", []))
        print(f"    Records extracted: {record_count}")

    return output_paths
