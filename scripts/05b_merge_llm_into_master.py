from __future__ import annotations

from am_mvt.extraction.merge_llm_records import append_llm_records_to_master


def main() -> None:
    print("Merging audited LLM extracted records into master dataset...")
    output_path, summary = append_llm_records_to_master(make_backup=False)

    print("\nStep 05b complete: LLM records merged into master dataset.")
    print(f"Updated master dataset: {output_path}")
    print("\nMerge summary:")
    print(summary)


if __name__ == "__main__":
    main()
