from __future__ import annotations

from am_mvt.extraction.batch_extract import run_batch_extraction
from am_mvt.extraction.postprocess import save_llm_extracted_records


def main() -> None:
    extracted_files = run_batch_extraction(
        limit=5,
        overwrite=False,
    )

    output_csv = save_llm_extracted_records()

    print("Step 04 complete: LLM extraction finished.")
    print(f"JSON files available: {len(extracted_files)}")
    print(f"Combined LLM extracted CSV: {output_csv}")


if __name__ == "__main__":
    main()