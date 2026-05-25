from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from am_mvt.config import get_path
from am_mvt.extraction.batch_extract import run_batch_extraction
from am_mvt.extraction.openai_extractor import DEFAULT_MODEL
from am_mvt.extraction.postprocess import save_llm_extracted_records


def load_project_env() -> None:
    """
    Load environment variables from the project-level .env file.

    This allows the user to store OPENAI_API_KEY in:
        .env

    Example:
        OPENAI_API_KEY=your_api_key
        OPENAI_MODEL=gpt-4o-mini
    """
    project_root = Path.cwd()
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def has_text_chunks(chunk_dir: Path) -> bool:
    return chunk_dir.exists() and any(chunk_dir.glob("*.txt"))


def has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def create_empty_llm_output() -> Path:
    output_path = get_path("data", "interim", "llm_extracted_records.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_llm_extracted_records(output_path=output_path)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured AM mechanical testing records from parsed PDF chunks using OpenAI."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of text chunks to process. Use 0 to process all chunks.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON extraction outputs.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model name.",
    )

    return parser.parse_args()


def main() -> None:
    load_project_env()

    args = parse_args()

    chunk_dir = get_path("data", "interim", "text_chunks")
    llm_output_dir = get_path("data", "interim", "llm_outputs")
    llm_output_dir.mkdir(parents=True, exist_ok=True)

    if not has_text_chunks(chunk_dir):
        output_csv = create_empty_llm_output()

        print("No text chunks found. Step 04 LLM extraction skipped safely.")
        print("Run python scripts/03_parse_documents.py first.")
        print(f"Empty LLM extracted CSV created: {output_csv}")
        print("Step 04 complete.")
        return

    if not has_openai_api_key():
        output_csv = create_empty_llm_output()

        print("OPENAI_API_KEY was not found.")
        print("Please add it to the project root .env file:")
        print("OPENAI_API_KEY=your_api_key")
        print("OPENAI_MODEL=gpt-4o-mini")
        print(f"Empty LLM extracted CSV created: {output_csv}")
        print("Step 04 complete.")
        return

    limit = None if args.limit == 0 else args.limit

    extracted_files = run_batch_extraction(
        limit=limit,
        overwrite=args.overwrite,
        model=args.model,
    )

    output_csv = save_llm_extracted_records()

    print("Step 04 complete: LLM extraction finished.")
    print(f"JSON files available: {len(extracted_files)}")
    print(f"Combined LLM extracted CSV: {output_csv}")
    print("Audit file: data/interim/llm_extraction_audit.csv")


if __name__ == "__main__":
    main()