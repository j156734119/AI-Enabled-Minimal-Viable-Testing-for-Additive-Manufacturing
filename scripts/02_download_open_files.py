"""
Step 02: Download lawful open files.

This script will later download only files that are publicly available and lawful
to use, such as open-access PDFs, supplementary materials, and public datasets.
"""

from am_mvt.config import get_path


def main() -> None:
    pdf_dir = get_path("data", "raw", "pdfs")
    supplementary_dir = get_path("data", "raw", "supplementary")
    dataset_dir = get_path("data", "raw", "open_datasets")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Step 02 complete: download folders prepared.")
    print(f"PDF folder: {pdf_dir}")
    print(f"Supplementary folder: {supplementary_dir}")
    print(f"Open dataset folder: {dataset_dir}")


if __name__ == "__main__":
    main()