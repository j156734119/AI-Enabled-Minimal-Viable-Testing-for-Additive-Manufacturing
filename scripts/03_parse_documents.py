"""
Step 03: Parse documents.

This script will later parse open PDFs, supplementary files, and datasets into
machine-readable text or tables.
"""

from am_mvt.config import get_path


def main() -> None:
    parsed_text_dir = get_path("data", "interim", "parsed_text")
    text_chunks_dir = get_path("data", "interim", "text_chunks")

    parsed_text_dir.mkdir(parents=True, exist_ok=True)
    text_chunks_dir.mkdir(parents=True, exist_ok=True)

    print("Step 03 complete: parsing folders prepared.")
    print(f"Parsed text folder: {parsed_text_dir}")
    print(f"Text chunks folder: {text_chunks_dir}")


if __name__ == "__main__":
    main()