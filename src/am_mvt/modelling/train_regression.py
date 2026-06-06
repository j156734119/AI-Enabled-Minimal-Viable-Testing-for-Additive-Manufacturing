from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

from am_mvt.config import get_path
from am_mvt.modelling.build_views import save_modelling_views
from am_mvt.modelling.evaluate import evaluate_regression
from am_mvt.modelling.make_dataset import MODEL_CONFIGS, prepare_regression_data


MODEL_RATIONALE = {
    "dummy_mean_baseline": {
        "family": "baseline",
        "rationale": "Mean baseline for checking whether learned models add signal.",
    },
    "ridge": {
        "family": "regularised_linear",
        "rationale": "Stable linear baseline for small, noisy tabular AM data.",
    },
    "elastic_net": {
        "family": "regularised_linear",
        "rationale": "Sparse/regularised linear model for correlated process variables.",
    },
    "svr_rbf_light": {
        "family": "kernel",
        "rationale": "Nonlinear kernel baseline for small to medium tabular datasets.",
    },
    "random_forest_light": {
        "family": "bagging_tree_ensemble",
        "rationale": "Robust tree ensemble for nonlinear mixed-feature interactions.",
    },
    "extra_trees_light": {
        "family": "bagging_tree_ensemble",
        "rationale": "High-variance randomized tree ensemble for robustness comparison.",
    },
    "gradient_boosting_light": {
        "family": "boosting_tree_ensemble",
        "rationale": "Classic boosting baseline for nonlinear process-property trends.",
    },
    "hist_gradient_boosting": {
        "family": "boosting_tree_ensemble",
        "rationale": "Efficient sklearn boosting model for tabular regression.",
    },
    "xgboost_light": {
        "family": "boosting_tree_ensemble",
        "rationale": "Strong gradient-boosted tree baseline for structured tabular data.",
    },
}


def get_model_metadata(model_name: str) -> dict[str, str]:
    return MODEL_RATIONALE.get(
        model_name,
        {
            "family": "unknown",
            "rationale": "No rationale recorded for this model.",
        },
    )


def get_models() -> dict[str, object]:
    """Return controlled candidate regressors for AM tabular data.

    The set intentionally mixes simple baselines, regularised linear models,
    kernel regression, bagging ensembles, and boosting ensembles so the project
    can compare whether nonlinear models add value beyond conservative
    baselines before any expensive hyperparameter search.
    """
    return {
        "dummy_mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(
            alpha=0.01,
            l1_ratio=0.25,
            max_iter=10000,
            random_state=42,
        ),
        "svr_rbf_light": SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
            gamma="scale",
        ),
        "random_forest_light": RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_light": ExtraTreesRegressor(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting_light": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
        "xgboost_light": XGBRegressor(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }


def get_feature_names_from_preprocessor(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def extract_feature_importance(
    pipeline: Pipeline,
    model_key: str,
    target: str,
    model_name: str,
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if not feature_names:
        return pd.DataFrame()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_)
    else:
        return pd.DataFrame()

    if len(feature_names) != len(importance_values):
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {
            "model_key": model_key,
            "target": target,
            "model": model_name,
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    return importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)


def fit_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series | None,
) -> Pipeline:
    if sample_weight is not None:
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        pipeline.fit(X_train, y_train)

    return pipeline


def train_one_target(
    model_key: str,
    target: str,
    dataset_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_regression_data(
        model_key=model_key,
        target=target,
        dataset_path=dataset_path,
    )

    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    w_train = prepared["w_train"]
    preprocessor = prepared["preprocessor"]

    models = get_models()

    models_dir = get_path("outputs", "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    importance_frames = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline = fit_pipeline(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            sample_weight=w_train,
        )

        predictions = pipeline.predict(X_test)
        metrics = evaluate_regression(y_test, predictions)
        model_metadata = get_model_metadata(model_name)

        model_path = models_dir / f"{model_key}_{target}_{model_name}.joblib"
        joblib.dump(pipeline, model_path)

        metric_rows.append(
            {
                "model_key": model_key,
                "target": target,
                "model": model_name,
                "model_family": model_metadata["family"],
                "model_rationale": model_metadata["rationale"],
                "n_rows_used": prepared["n_rows"],
                "n_groups": prepared["n_groups"],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "split_method": prepared["split_method"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "model_path": str(model_path),
            }
        )

        importance_df = extract_feature_importance(
            pipeline=pipeline,
            model_key=model_key,
            target=target,
            model_name=model_name,
        )

        if not importance_df.empty:
            importance_frames.append(importance_df)

    metrics_df = pd.DataFrame(metric_rows)

    if importance_frames:
        importance_df = pd.concat(importance_frames, ignore_index=True, sort=False)
    else:
        importance_df = pd.DataFrame(
            columns=["model_key", "target", "model", "feature", "importance"]
        )

    return metrics_df, importance_df


def train_project_models(
    rebuild_views: bool = True,
    max_sn_rows_per_dataset_id: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rebuild_views:
        save_modelling_views(max_sn_rows_per_dataset_id=max_sn_rows_per_dataset_id)

    training_plan = {
        model_key: config["targets"] for model_key, config in MODEL_CONFIGS.items()
    }

    metric_frames = []
    importance_frames = []
    error_rows = []

    for model_key, targets in training_plan.items():
        for target in targets:
            try:
                metrics_df, importance_df = train_one_target(
                    model_key=model_key,
                    target=target,
                )

                metric_frames.append(metrics_df)

                if not importance_df.empty:
                    importance_frames.append(importance_df)

            except Exception as exc:
                error_rows.append(
                    {
                        "model_key": model_key,
                        "target": target,
                        "error": str(exc),
                    }
                )

    if metric_frames:
        all_metrics_df = pd.concat(metric_frames, ignore_index=True, sort=False)
    else:
        all_metrics_df = pd.DataFrame()

    if importance_frames:
        all_importance_df = pd.concat(importance_frames, ignore_index=True, sort=False)
    else:
        all_importance_df = pd.DataFrame(
            columns=["model_key", "target", "model", "feature", "importance"]
        )

    errors_df = pd.DataFrame(
        error_rows,
        columns=["model_key", "target", "error"],
    )

    tables_dir = get_path("outputs", "tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_metrics_df.to_csv(
        tables_dir / "project_regression_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_importance_df.to_csv(
        tables_dir / "project_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    errors_df.to_csv(
        tables_dir / "project_training_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return all_metrics_df, all_importance_df, errors_df
