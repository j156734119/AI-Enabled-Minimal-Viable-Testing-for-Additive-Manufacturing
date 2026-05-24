"""
Step 01: Search candidate public sources.

This script will later call metadata APIs such as Crossref, OpenAlex,
or other scholarly search services to identify candidate papers and datasets.

At this initial stage, it creates the expected metadata output file structure.
"""

from pathlib import Path

from am_mvt.config import get_path, load_config


def main() -> None:
    config = load_config()

    output_dir = get_path("data", "raw", "metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "candidate_sources.csv"

    if not output_file.exists():
        output_file.write_text(
            "doi,title,journal,year,source_type,url,screening_status\n",
            encoding="utf-8",
        )

    print("Step 01 complete: source search structure prepared.")
    print(f"Project: {config['project']['title']}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()