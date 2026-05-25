from __future__ import annotations

import os
from pathlib import Path

from am_mvt.config import get_path
from am_mvt.extraction.postprocess import save_llm_extracted_records


def has_text_chunks(chunk_dir: Path) -> bool:
    return chunk_dir.exists() and any(chunk_dir.glob("*.txt"))


def has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def create_empty_llm_output() -> Path:
    output_path = get_path("data", "interim", "llm_extracted_records.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_llm_extracted_records()

    return output_path


def main() -> None:
    chunk_dir = get_path("data", "interim", "text_chunks")
    llm_output_dir = get_path("data", "interim", "llm_outputs")
    llm_output_dir.mkdir(parents=True, exist_ok=True)

    if not has_text_chunks(chunk_dir):
        output_csv = create_empty_llm_output()

        print("No text chunks found. Step 04 LLM extraction skipped safely.")
        print("This is OK for the current open-dataset baseline workflow.")
        print(f"Empty LLM extracted CSV created: {output_csv}")
        print("Step 04 complete: optional LLM extraction stage finished.")
        return

    if not has_openai_api_key():
        output_csv = create_empty_llm_output()

        print("OPENAI_API_KEY was not found. Step 04 LLM extraction skipped safely.")
        print("Set OPENAI_API_KEY in your environment if you want to run LLM extraction.")
        print(f"Empty LLM extracted CSV created: {output_csv}")
        print("Step 04 complete: optional LLM extraction stage finished.")
        return

    try:
        from am_mvt.extraction.batch_extract import run_batch_extraction
    except Exception as exc:
        output_csv = create_empty_llm_output()

        print("Could not import the LLM extraction module.")
        print(f"Reason: {exc}")
        print("Step 04 skipped safely.")
        print(f"Empty LLM extracted CSV created: {output_csv}")
        print("Step 04 complete: optional LLM extraction stage finished.")
        return

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