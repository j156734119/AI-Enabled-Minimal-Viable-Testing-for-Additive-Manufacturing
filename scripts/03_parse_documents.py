from __future__ import annotations

from am_mvt.config import get_path
from am_mvt.parsing.document_pipeline import parse_active_pdf_documents


def parse_pdf_documents() -> tuple[int, int]:
    parsed_count, chunk_count, manifest_path, archive_manifest = (
        parse_active_pdf_documents()
    )
    print(f"Active chunk manifest: {manifest_path}")
    if archive_manifest is not None:
        print(f"Stale derivatives archived: {archive_manifest}")
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
