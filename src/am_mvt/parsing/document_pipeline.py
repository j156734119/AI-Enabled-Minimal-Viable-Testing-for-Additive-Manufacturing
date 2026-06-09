from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from am_mvt.config import get_path
from am_mvt.ingestion.literature_manifest import save_literature_manifest
from am_mvt.parsing.chunk_text import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_text,
)
from am_mvt.parsing.parse_pdf_text import parse_pdf_to_text
from am_mvt.utils.artifacts import (
    archive_files,
    sha256_file,
    utc_archive_name,
)


ACTIVE_CHUNK_COLUMNS = [
    "chunk_id",
    "chunk_path",
    "chunk_sha256",
    "source_id",
    "source_file",
    "source_content_sha256",
    "chunk_size",
    "overlap",
]
DEFAULT_SKIP_FILE = Path("config/llm_extraction_skip.txt")


def active_chunk_manifest_path() -> Path:
    return get_path("data", "interim", "active_chunk_manifest.csv")


def load_active_chunk_manifest(
    path: str | Path | None = None,
) -> pd.DataFrame:
    manifest_path = Path(path) if path is not None else active_chunk_manifest_path()
    if not manifest_path.exists():
        return pd.DataFrame(columns=ACTIVE_CHUNK_COLUMNS)
    return pd.read_csv(manifest_path, low_memory=False)


def active_chunk_paths(
    manifest_path: str | Path | None = None,
) -> list[Path]:
    manifest = load_active_chunk_manifest(manifest_path)
    return [
        get_path(*Path(relative).parts)
        for relative in manifest.get("chunk_path", pd.Series(dtype="string"))
        .dropna()
        .astype(str)
    ]


def load_skipped_pdf_stems(
    skip_file: str | Path = DEFAULT_SKIP_FILE,
) -> set[str]:
    path = get_path(*Path(skip_file).parts)
    if not path.exists():
        return set()
    return {
        line.strip().removesuffix(".pdf")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _source_stem_from_derivative(path: Path) -> str:
    stem = path.stem
    return stem.split("_chunk_", maxsplit=1)[0]


def _archive_stale_derivatives(
    active_sources: set[str],
    expected_chunks: set[str],
    parsed_text_dir: Path,
    chunk_dir: Path,
    llm_output_dir: Path,
) -> Path | None:
    stale: list[Path] = []
    stale.extend(
        path
        for path in parsed_text_dir.glob("*.txt")
        if path.stem not in active_sources
    )
    stale.extend(
        path
        for path in chunk_dir.glob("*.txt")
        if path.stem not in expected_chunks
    )
    stale.extend(
        path
        for path in llm_output_dir.glob("*.json")
        if path.stem not in expected_chunks
    )
    if not stale:
        return None
    archive_root = get_path(
        "archive",
        utc_archive_name("document_derivatives"),
    )
    return archive_files(stale, archive_root, project_root=get_path())


def parse_active_pdf_documents(
    *,
    pdf_dir: str | Path | None = None,
    parsed_text_dir: str | Path | None = None,
    chunk_dir: str | Path | None = None,
    llm_output_dir: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[int, int, Path, Path | None]:
    pdf_dir = Path(pdf_dir) if pdf_dir is not None else get_path("data", "raw", "pdfs")
    parsed_text_dir = (
        Path(parsed_text_dir)
        if parsed_text_dir is not None
        else get_path("data", "interim", "parsed_text")
    )
    chunk_dir = (
        Path(chunk_dir)
        if chunk_dir is not None
        else get_path("data", "interim", "text_chunks")
    )
    llm_output_dir = (
        Path(llm_output_dir)
        if llm_output_dir is not None
        else get_path("data", "interim", "llm_outputs")
    )
    for directory in [pdf_dir, parsed_text_dir, chunk_dir, llm_output_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    _, literature_manifest = save_literature_manifest()
    active = literature_manifest.loc[
        literature_manifest["processing_status"].eq("canonical")
        & literature_manifest["ready_for_parsing"].fillna(False).astype(bool)
    ].copy()
    skipped_stems = load_skipped_pdf_stems()
    active = active.loc[
        ~active["local_pdf_filename"]
        .astype(str)
        .map(lambda value: Path(value).stem in skipped_stems)
    ].copy()

    rows: list[dict[str, object]] = []
    parsed_count = 0
    for row in active.itertuples(index=False):
        pdf_path = pdf_dir / str(row.local_pdf_filename)
        text_path = parsed_text_dir / f"{pdf_path.stem}.txt"
        parse_pdf_to_text(pdf_path, text_path)
        text = text_path.read_text(encoding="utf-8")
        for index, chunk in enumerate(
            chunk_text(text, max_chars=chunk_size, overlap=overlap)
        ):
            chunk_id = f"{pdf_path.stem}_chunk_{index:04d}"
            chunk_path = chunk_dir / f"{chunk_id}.txt"
            chunk_path.write_text(chunk, encoding="utf-8")
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_path": chunk_path.relative_to(get_path()).as_posix(),
                    "chunk_sha256": sha256_file(chunk_path),
                    "source_id": row.canonical_source_id,
                    "source_file": row.local_pdf_filename,
                    "source_content_sha256": row.content_sha256,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                }
            )
        parsed_count += 1

    manifest = pd.DataFrame(rows, columns=ACTIVE_CHUNK_COLUMNS)
    output_path = active_chunk_manifest_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")
    expected_chunks = set(manifest["chunk_id"].astype(str))
    active_sources = {
        Path(str(filename)).stem
        for filename in active["local_pdf_filename"].astype(str)
    }
    archive_manifest = _archive_stale_derivatives(
        active_sources,
        expected_chunks,
        parsed_text_dir,
        chunk_dir,
        llm_output_dir,
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "active_pdf_count": parsed_count,
                "active_chunk_count": len(manifest),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return parsed_count, len(manifest), output_path, archive_manifest
