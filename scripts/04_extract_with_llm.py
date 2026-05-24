"""
Step 04: Extract structured records with LLM assistance.

This script will later call the OpenAI API to extract structured AM records
from lawful public text chunks.

The LLM must only be used as an assistive extraction tool. Extracted records
must be validated before analysis.
"""

from am_mvt.config import get_path, load_project_environment


def main() -> None:
    load_project_environment()

    llm_output_dir = get_path("data", "interim", "llm_outputs")
    extraction_log_dir = get_path("data", "interim", "extraction_logs")

    llm_output_dir.mkdir(parents=True, exist_ok=True)
    extraction_log_dir.mkdir(parents=True, exist_ok=True)

    print("Step 04 complete: LLM extraction folders prepared.")
    print(f"LLM output folder: {llm_output_dir}")
    print(f"Extraction log folder: {extraction_log_dir}")


if __name__ == "__main__":
    main()