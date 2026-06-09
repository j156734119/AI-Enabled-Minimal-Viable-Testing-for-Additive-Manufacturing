from __future__ import annotations

from pathlib import Path

from am_mvt.config import get_path


DEFAULT_CHUNK_SIZE = 3500
DEFAULT_CHUNK_OVERLAP = 300


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if not text:
        return []
    if max_chars <= overlap:
        raise ValueError("max_chars must be larger than overlap.")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_parsed_text_files(
    parsed_text_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Path]:
    if parsed_text_dir is None:
        parsed_text_dir = get_path("data", "interim", "parsed_text")
    else:
        parsed_text_dir = Path(parsed_text_dir)

    if output_dir is None:
        output_dir = get_path("data", "interim", "text_chunks")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []

    if not parsed_text_dir.exists():
        return output_files

    for text_file in sorted(parsed_text_dir.glob("*.txt")):
        text = text_file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)

        for index, chunk in enumerate(chunks):
            output_path = output_dir / f"{text_file.stem}_chunk_{index:04d}.txt"
            output_path.write_text(chunk, encoding="utf-8")
            output_files.append(output_path)

    return output_files
