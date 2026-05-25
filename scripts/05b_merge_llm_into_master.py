from __future__ import annotations

from am_mvt.extraction.merge_llm_records import append_llm_records_to_master
from am_mvt.extraction.postprocess import save_llm_extracted_records


def main() -> None:
    print("Post-processing LLM JSON outputs...")
    llm_csv = save_llm_extracted_records()
    print(f"LLM extracted records CSV: {llm_csv}")

    print("\nMerging LLM extracted records into master dataset...")
    output_path, summary = append_llm_records_to_master()

    print("\nStep 05b complete: LLM records merged into master dataset.")
    print(f"Updated master dataset: {output_path}")
    print("\nMerge summary:")
    print(summary)


if __name__ == "__main__":
    main()