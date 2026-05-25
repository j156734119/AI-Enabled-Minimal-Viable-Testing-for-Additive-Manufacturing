from __future__ import annotations

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

    print("\nTraining Model 1 and Model 2...")

    metrics_df, importance_df, errors_df = train_project_models(
        rebuild_views=False,
        max_sn_rows_per_dataset_id=10,
    )

    print("\nTraining complete.")

    if not metrics_df.empty:
        print("\nMetrics:")
        print(metrics_df)
    else:
        print("\nNo successful model metrics were generated.")

    if not errors_df.empty:
        print("\nSome targets were skipped or failed:")
        print(errors_df)

    print("\nOutputs:")
    print("Metrics: outputs/tables/project_regression_model_metrics.csv")
    print("Feature importance: outputs/tables/project_feature_importance.csv")
    print("Training errors: outputs/tables/project_training_errors.csv")
    print("Models: outputs/models/")


if __name__ == "__main__":
    main()