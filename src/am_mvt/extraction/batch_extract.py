from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from am_mvt.config import get_path
from am_mvt.extraction.openai_extractor import extract_records_from_file


def run_batch_extraction(
    chunk_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    if chunk_dir is None:
        chunk_dir = get_path("data", "interim", "text_chunks")
    else:
        chunk_dir = Path(chunk_dir)

    if output_dir is None:
        output_dir = get_path("data", "interim", "llm_outputs")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunk_dir.exists():
        return []

    chunk_files = sorted(chunk_dir.glob("*.txt"))

    if limit is not None:
        chunk_files = chunk_files[:limit]

    output_files: list[Path] = []

    for chunk_file in tqdm(chunk_files, desc="Extracting AM records"):
        output_path = output_dir / f"{chunk_file.stem}.json"

        if output_path.exists() and not overwrite:
            output_files.append(output_path)
            continue

        source_hint = chunk_file.stem
        extracted_path = extract_records_from_file(
            input_path=chunk_file,
            output_path=output_path,
            source_hint=source_hint,
        )

        output_files.append(extracted_path)

    return output_files