from __future__ import annotations

from am_mvt.extraction.audit_records import save_extraction_audit


def main() -> None:
    output_path, audited_df = save_extraction_audit()
    status_counts = audited_df["audit_status"].value_counts(dropna=False)

    print("Step 04b complete: deterministic extraction audit finished.")
    print(f"Audit review file: {output_path}")

    for status in ["approved", "human_review_required", "rejected"]:
        print(f"{status}: {int(status_counts.get(status, 0))}")

    print("Only approved records are eligible for dataset integration.")


if __name__ == "__main__":
    main()
