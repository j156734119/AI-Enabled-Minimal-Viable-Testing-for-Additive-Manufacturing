from __future__ import annotations

import argparse

from am_mvt.ingestion.llm_source_screening import (
    DEFAULT_SCREENING_MODEL,
    run_openai_agent_source_screening,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenAI agent web source screening within the approved "
            "additive-manufacturing journal scope."
        )
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
        help="Maximum candidate papers requested from each journal per search round.",
    )
    parser.add_argument(
        "--min-per-journal",
        type=int,
        default=4,
        help="Minimum candidates reserved for each approved journal when available.",
    )
    parser.add_argument(
        "--search-rounds",
        type=int,
        default=3,
        help="Number of focused search rounds to run across the approved journals.",
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

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print("Step 01: running OpenAI agent web source screening...")
    screened_df, output_paths = run_openai_agent_source_screening(
        target_count=args.target_count,
        per_journal_limit=args.per_journal_limit,
        min_per_journal=args.min_per_journal,
        year_from=args.year_from,
        year_to=args.year_to,
        model=args.model,
        search_rounds=args.search_rounds,
    )

    print("\nStep 01 complete: OpenAI agent web source screening finished.")
    print(f"Screened candidates: {len(screened_df)}")
    print(f"Workflow CSV: {output_paths['interim']}")
    print(f"Output table: {output_paths['table']}")
    print(f"Journal scope table: {output_paths['journal_scope']}")


if __name__ == "__main__":
    main()
