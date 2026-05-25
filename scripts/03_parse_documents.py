from __future__ import annotations

from am_mvt.ingestion.load_open_datasets import save_open_dataset_preview
from am_mvt.parsing.chunk_text import chunk_parsed_text_files
from am_mvt.parsing.parse_pdf_text import parse_all_pdfs


def main() -> None:
    parsed_files = parse_all_pdfs()
    chunk_files = chunk_parsed_text_files()

    (
        raw_output,
        standard_output,
        report_output,
        mapping_report_output,
    ) = save_open_dataset_preview()

    print("Step 03 complete: documents and open datasets parsed.")
    print(f"Parsed PDF text files: {len(parsed_files)}")
    print(f"Text chunks created: {len(chunk_files)}")
    print(f"Raw open dataset preview: {raw_output}")
    print(f"Standardised open dataset preview: {standard_output}")
    print(f"Open dataset load report: {report_output}")
    print(f"Column mapping report: {mapping_report_output}")


if __name__ == "__main__":
    main()