from __future__ import annotations

from am_mvt.config import load_config
from am_mvt.ingestion.dataset_registry import write_core_sources_csv


def main() -> None:
    config = load_config()
    output_path = write_core_sources_csv()

    print("Step 01 complete: core dataset sources registered.")
    print(f"Project: {config['project']['title']}")
    print(f"Candidate sources saved to: {output_path}")


if __name__ == "__main__":
    main()