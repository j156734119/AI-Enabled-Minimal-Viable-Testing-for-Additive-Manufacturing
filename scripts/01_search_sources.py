from __future__ import annotations

import argparse

from am_mvt.config import load_config
from am_mvt.ingestion.dataset_registry import write_core_sources_csv
from am_mvt.ingestion.llm_source_screening import (
    DEFAULT_SCREENING_MODEL,
    run_llm_web_source_screening,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register core sources and optionally run OpenAI web-assisted "
            "source screening within the Meeting one journal scope."
        )
    )
    parser.add_argument(
        "--llm-web-search",
        action="store_true",
        help="Use OpenAI web search to screen candidate AM papers.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=50,
        help="Maximum number of screened candidate papers to keep.",
    )
    parser.add_argument(
        "--per-journal-limit",
        type=int,
        default=8,
        help="Maximum candidate papers requested from each approved journal.",
    )
    parser.add_argument(
        "--year-from",
        type=int,
        default=2015,
        help="Start year for web source screening.",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=2026,
        help="End year for web source screening.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_SCREENING_MODEL,
        help="OpenAI model used for web source screening.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    output_path = write_core_sources_csv()

    print("Step 01 complete: core dataset sources registered.")
    print(f"Project: {config['project']['title']}")
    print(f"Candidate sources saved to: {output_path}")

    if not args.llm_web_search:
        print("\nLLM web source screening was not requested.")
        print("To run it, use: python scripts/01_search_sources.py --llm-web-search")
        return

    print("\nRunning OpenAI web-assisted source screening...")
    screened_df, output_paths = run_llm_web_source_screening(
        target_count=args.target_count,
        per_journal_limit=args.per_journal_limit,
        year_from=args.year_from,
        year_to=args.year_to,
        model=args.model,
    )

    print("\nLLM web source screening complete.")
    print(f"Screened candidates: {len(screened_df)}")
    print(f"Workflow CSV: {output_paths['interim']}")
    print(f"Output table: {output_paths['table']}")
    print(f"Journal scope table: {output_paths['journal_scope']}")


if __name__ == "__main__":
    main()
