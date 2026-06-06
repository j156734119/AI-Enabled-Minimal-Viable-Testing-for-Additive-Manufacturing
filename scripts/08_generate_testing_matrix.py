"""
Step 08: Generate reduced testing matrix.

This script will later combine feature importance, sensitivity analysis,
coverage analysis, and risk-prioritised logic to propose a reduced but
representative testing strategy.
"""

from am_mvt.config import get_path


def main() -> None:
    report_dir = get_path("outputs", "reports")
    table_dir = get_path("outputs", "tables")

    report_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    output_file = table_dir / "reduced_testing_matrix.csv"

    if not output_file.exists():
        output_file.write_text(
            (
                "priority,alloy_family,am_process,build_orientation,"
                "surface_condition,test_type,target_property,"
                "recommended_test_condition,reason,supporting_features,"
                "model_evidence,coverage_risk,confidence_level,"
                "needs_validation\n"
            ),
            encoding="utf-8",
        )

    print("Step 08 specification placeholder: output schema prepared.")
    print(f"Testing matrix file: {output_file}")


if __name__ == "__main__":
    main()
