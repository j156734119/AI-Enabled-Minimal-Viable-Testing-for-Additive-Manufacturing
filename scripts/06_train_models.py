from __future__ import annotations

import pandas as pd

from am_mvt.config import get_path
from am_mvt.modelling.build_views import save_modelling_views
from am_mvt.modelling.train_regression import train_project_models


def main() -> None:
    print("Building task-specific modelling views...")

    view_paths, view_summary = save_modelling_views(
        max_sn_rows_per_dataset_id=10,
    )

    print("\nModelling views saved:")
    for name, path in view_paths.items():
        print(f"{name}: {path}")

    print("\nView summary:")
    print(view_summary)

    print("\nTraining project modelling tasks...")
    print("Model 1: UTS prediction")
    print("Model 2: S-N fatigue life prediction")
    print("Model 3: elongation/yield prediction")
    print("Model 4: elastic modulus prediction")
    print(
        "\nAlgorithms per target: Dummy baseline, Ridge, Random Forest, and XGBoost."
    )
    print(
        "This is regression, so performance is reported with R2, MAE, and RMSE "
        "rather than classification accuracy."
    )

    metrics_df, importance_df, errors_df = train_project_models(
        rebuild_views=False,
        max_sn_rows_per_dataset_id=10,
    )

    print("\nTraining complete.")

    if not metrics_df.empty:
        display_columns = [
            "model_key",
            "target",
            "model",
            "mae",
            "rmse",
            "r2",
            "r2_percent_variance_explained",
            "mae_improvement_vs_dummy_percent",
            "is_best_non_dummy_model",
        ]
        print("\nRegression metrics:")
        print(metrics_df[[column for column in display_columns if column in metrics_df]])

        best_summary_path = get_path(
            "outputs",
            "tables",
            "project_best_model_summary.csv",
        )
        best_summary = pd.read_csv(best_summary_path)
        print("\nBest non-dummy model for each target:")
        print(best_summary)
    else:
        print("\nNo successful model metrics were generated.")

    if not errors_df.empty:
        print("\nSome targets were skipped or failed:")
        print(errors_df)

    print("\nOutputs:")
    print("Metrics: outputs/tables/project_regression_model_metrics.csv")
    print("Feature importance: outputs/tables/project_feature_importance.csv")
    print("Training errors: outputs/tables/project_training_errors.csv")
    print("Best models: outputs/tables/project_best_model_summary.csv")
    print("Data audit: outputs/tables/project_training_data_audit.csv")
    print("Physical checks: outputs/tables/project_physical_sanity_checks.csv")
    print("Models: outputs/models/")


if __name__ == "__main__":
    main()
