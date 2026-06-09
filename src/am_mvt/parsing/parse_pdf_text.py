from __future__ import annotations

from pathlib import Path

import fitz

from am_mvt.config import get_path
from am_mvt.parsing.clean_text import clean_extracted_text


def parse_pdf_to_text(pdf_path: str | Path, output_path: str | Path) -> Path:
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages: list[str] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_text = page.get_text("text")
            pages.append(f"\n\n--- Page {page_index + 1} ---\n\n{page_text}")

    cleaned = clean_extracted_text("\n".join(pages))
    output_path.write_text(cleaned, encoding="utf-8")

    return output_path


def parse_all_pdfs(
    pdf_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    if pdf_dir is None:
        pdf_dir = get_path("data", "raw", "pdfs")
    else:
        pdf_dir = Path(pdf_dir)

    if output_dir is None:
        output_dir = get_path("data", "interim", "parsed_text")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_files: list[Path] = []

    if not pdf_dir.exists():
        return parsed_files

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        output_path = output_dir / f"{pdf_path.stem}.txt"
        parsed_files.append(parse_pdf_to_text(pdf_path, output_path))

    return parsed_files
