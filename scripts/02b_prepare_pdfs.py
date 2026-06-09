from __future__ import annotations

import argparse

from am_mvt.config import get_path
from am_mvt.ingestion.pdf_title_normaliser import prepare_pdf_normalisation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read downloaded PDFs, identify paper titles/DOIs, and prepare "
            "consistent compact filenames for the parsing workflow."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move and rename PDFs. Without this flag, only write a preview plan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inbox_dir = get_path("data", "raw", "pdfs", "inbox")
    output_dir = get_path("data", "raw", "pdfs")
    report_path = get_path("outputs", "tables", "pdf_title_normalisation_plan.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_df = prepare_pdf_normalisation(
        inbox_dir=inbox_dir,
        output_dir=output_dir,
        apply_changes=args.apply,
    )
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("Step 02b complete: PDF title normalisation checked.")
    print(f"Download inbox: {inbox_dir}")
    print(f"Normalised PDF folder: {output_dir}")
    print(f"Report: {report_path}")
    print(f"PDFs checked: {len(report_df)}")

    if report_df.empty:
        print("No PDFs found in the inbox.")
        return

    if args.apply:
        print(f"PDFs moved: {(report_df['action'] == 'moved').sum()}")
    else:
        print("Preview only. Review the report, then rerun with --apply.")

    print(
        "Needs human check: "
        f"{report_df['needs_human_check'].fillna(True).astype(bool).sum()}"
    )


if __name__ == "__main__":
    main()
