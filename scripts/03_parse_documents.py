from __future__ import annotations

from pathlib import Path

from am_mvt.config import get_path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract plain text from a PDF using PyMuPDF.

    This script is optional for the current open-dataset baseline workflow.
    If no PDFs are provided under data/raw/pdfs, the script exits safely.
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with: python -m pip install pymupdf"
        ) from exc

    document = fitz.open(pdf_path)
    pages: list[str] = []

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        text = page.get_text("text")
        pages.append(f"\n\n--- Page {page_index + 1} ---\n\n{text}")

    document.close()

    return "\n".join(pages)


def clean_text(text: str) -> str:
    lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped:
            lines.append(stripped)

    return "\n".join(lines)


def chunk_text(
    text: str,
    chunk_size: int = 3500,
    overlap: int = 300,
) -> list[str]:
    """
    Split text into overlapping chunks for optional LLM extraction.
    """
    if not text:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap.")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def parse_pdf_documents() -> tuple[int, int]:
    pdf_dir = get_path("data", "raw", "pdfs")
    parsed_text_dir = get_path("data", "interim", "parsed_text")
    chunk_dir = get_path("data", "interim", "text_chunks")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    parsed_text_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found. Step 03 skipped safely.")
        print("This is OK for the current open-dataset baseline workflow.")
        return 0, 0

    parsed_count = 0
    chunk_count = 0

    for pdf_path in pdf_files:
        print(f"Parsing PDF: {pdf_path.name}")

        raw_text = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw_text)

        text_output_path = parsed_text_dir / f"{pdf_path.stem}.txt"
        text_output_path.write_text(cleaned, encoding="utf-8")

        chunks = chunk_text(cleaned)

        for chunk_index, chunk in enumerate(chunks):
            chunk_output_path = chunk_dir / f"{pdf_path.stem}_chunk_{chunk_index:04d}.txt"
            chunk_output_path.write_text(chunk, encoding="utf-8")

        parsed_count += 1
        chunk_count += len(chunks)

    return parsed_count, chunk_count


def main() -> None:
    parsed_count, chunk_count = parse_pdf_documents()

    print("Step 03 complete: document parsing finished.")
    print(f"PDF files parsed: {parsed_count}")
    print(f"Text chunks created: {chunk_count}")
    print(f"Parsed text folder: {get_path('data', 'interim', 'parsed_text')}")
    print(f"Text chunk folder: {get_path('data', 'interim', 'text_chunks')}")


if __name__ == "__main__":
    main()