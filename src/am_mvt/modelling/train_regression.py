from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
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
    "random_forest": {
        "family": "bagging_tree_ensemble",
        "rationale": "Robust tree ensemble for nonlinear mixed-feature interactions.",
    },
    "xgboost": {
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
    """Return a compact, dissertation-focused regression comparison set."""
    return {
        "dummy_mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": Ridge(
            alpha=1.0,
            solver="sag",
            max_iter=3000,
            tol=1e-3,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
    }


def add_comparative_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add regression comparisons without presenting R2 as classification accuracy."""
    if metrics_df.empty:
        return metrics_df

    result = metrics_df.copy()
    result["r2_percent_variance_explained"] = result["r2"] * 100.0
    result["mae_improvement_vs_dummy_percent"] = np.nan
    result["rmse_improvement_vs_dummy_percent"] = np.nan
    result["is_best_non_dummy_model"] = False

    for _, indices in result.groupby(["model_key", "target"]).groups.items():
        group = result.loc[indices]
        baseline = group.loc[group["model"].eq("dummy_mean_baseline")]

        if not baseline.empty:
            baseline_mae = float(baseline.iloc[0]["mae"])
            baseline_rmse = float(baseline.iloc[0]["rmse"])

            if baseline_mae > 0:
                result.loc[indices, "mae_improvement_vs_dummy_percent"] = (
                    (baseline_mae - group["mae"]) / baseline_mae * 100.0
                )

            if baseline_rmse > 0:
                result.loc[indices, "rmse_improvement_vs_dummy_percent"] = (
                    (baseline_rmse - group["rmse"]) / baseline_rmse * 100.0
                )

        candidates = group.loc[~group["model"].eq("dummy_mean_baseline")]

        if not candidates.empty:
            result.loc[candidates["r2"].idxmax(), "is_best_non_dummy_model"] = True

    return result


def make_best_model_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or "is_best_non_dummy_model" not in metrics_df.columns:
        return pd.DataFrame()

    columns = [
        "model_key",
        "target",
        "model",
        "n_rows_used",
        "n_groups",
        "source_count",
        "invalid_target_rows_removed",
        "mae",
        "rmse",
        "r2",
        "r2_percent_variance_explained",
        "mae_improvement_vs_dummy_percent",
        "rmse_improvement_vs_dummy_percent",
        "split_method",
        "model_path",
    ]
    available = [column for column in columns if column in metrics_df.columns]

    return (
        metrics_df.loc[metrics_df["is_best_non_dummy_model"], available]
        .sort_values(["model_key", "target"])
        .reset_index(drop=True)
    )


def remove_stale_model_files(
    model_key: str,
    target: str,
    active_model_names: set[str],
) -> None:
    models_dir = get_path("outputs", "models")

    if not models_dir.exists():
        return

    prefix = f"{model_key}_{target}_"

    for path in models_dir.glob(f"{prefix}*.joblib"):
        model_name = path.stem.removeprefix(prefix)

        if model_name not in active_model_names:
            path.unlink()


def remove_legacy_model_files() -> None:
    models_dir = get_path("outputs", "models")

    if not models_dir.exists():
        return

    for path in models_dir.glob("model1_static_*.joblib"):
        path.unlink()


def build_training_data_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_key, config in MODEL_CONFIGS.items():
        dataset_path = Path(config["dataset_path"])

        if not dataset_path.exists():
            continue

        df = pd.read_csv(dataset_path, low_memory=False)
        feature_columns = config["numeric_features"] + config["categorical_features"]
        available_features = [column for column in feature_columns if column in df.columns]
        feature_coverage = (
            df[available_features].notna().mean() if available_features else pd.Series(dtype=float)
        )

        source_counts = (
            df["source_id"].value_counts(dropna=True)
            if "source_id" in df.columns
            else pd.Series(dtype=int)
        )
        source_count = int(len(source_counts))
        dominant_source_fraction = (
            float(source_counts.iloc[0] / source_counts.sum())
            if not source_counts.empty
            else np.nan
        )

        group_column = config.get("group_column")
        group_count = (
            int(df[group_column].nunique(dropna=True))
            if group_column in df.columns
            else 0
        )

        for target in config["targets"]:
            target_values = pd.to_numeric(df[target], errors="coerce").dropna()
            target_bounds = config.get("target_bounds", {}).get(target)
            invalid_target_rows = 0

            if target_bounds is not None:
                lower_bound, upper_bound = target_bounds
                invalid_target_rows = int(
                    (~target_values.between(lower_bound, upper_bound, inclusive="both")).sum()
                )

            if source_count < 5:
                status = "limited_source_diversity"
            elif group_count < 30:
                status = "limited_independent_groups"
            elif dominant_source_fraction > 0.8:
                status = "dominated_by_one_source"
            elif not feature_coverage.empty and float(feature_coverage.median()) < 0.5:
                status = "sparse_feature_coverage"
            else:
                status = "adequate_for_baseline_modelling"

            rows.append(
                {
                    "model_key": model_key,
                    "target": target,
                    "view_rows": len(df),
                    "target_non_missing": len(target_values),
                    "source_count": source_count,
                    "independent_group_count": group_count,
                    "dominant_source_fraction": dominant_source_fraction,
                    "median_feature_coverage": (
                        float(feature_coverage.median())
                        if not feature_coverage.empty
                        else np.nan
                    ),
                    "invalid_target_rows": invalid_target_rows,
                    "target_min": target_values.min() if not target_values.empty else np.nan,
                    "target_median": (
                        target_values.median() if not target_values.empty else np.nan
                    ),
                    "target_max": target_values.max() if not target_values.empty else np.nan,
                    "sufficiency_status": status,
                }
            )

    return pd.DataFrame(rows)


def build_physical_sanity_checks() -> pd.DataFrame:
    checks = [
        (
            "model1_uts",
            "porosity_percent",
            "uts_MPa",
            "negative",
            "Higher porosity is generally associated with lower load-bearing strength.",
        ),
        (
            "model1_uts",
            "relative_density_percent",
            "uts_MPa",
            "positive",
            "Higher relative density is generally associated with higher strength.",
        ),
        (
            "model2_sn_fatigue",
            "stress_amplitude_MPa",
            "log10_fatigue_life_cycles",
            "negative",
            "Higher cyclic stress amplitude is expected to reduce fatigue life.",
        ),
        (
            "model2_sn_fatigue",
            "max_stress_MPa",
            "log10_fatigue_life_cycles",
            "negative",
            "Higher maximum cyclic stress is expected to reduce fatigue life.",
        ),
        (
            "model3_elongation_yield",
            "uts_MPa",
            "yield_strength_MPa",
            "positive",
            "Yield strength and UTS are expected to be positively associated.",
        ),
        (
            "model4_elastic_modulus",
            "porosity_percent",
            "youngs_modulus_GPa",
            "negative",
            "Porosity generally reduces effective elastic stiffness.",
        ),
        (
            "model4_elastic_modulus",
            "relative_density_percent",
            "youngs_modulus_GPa",
            "positive",
            "Higher relative density generally increases effective stiffness.",
        ),
    ]
    rows: list[dict[str, object]] = []

    for model_key, input_column, output_column, expected_direction, rationale in checks:
        dataset_path = Path(MODEL_CONFIGS[model_key]["dataset_path"])

        if not dataset_path.exists():
            continue

        df = pd.read_csv(
            dataset_path,
            usecols=lambda column: column in {input_column, output_column},
        )
        pair = df[[input_column, output_column]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if len(pair) >= 3:
            x_rank = pair[input_column].rank(method="average").tolist()
            y_rank = pair[output_column].rank(method="average").tolist()
            x_mean = sum(x_rank) / len(x_rank)
            y_mean = sum(y_rank) / len(y_rank)
            numerator = sum(
                (x_value - x_mean) * (y_value - y_mean)
                for x_value, y_value in zip(x_rank, y_rank)
            )
            x_sum_squares = sum((value - x_mean) ** 2 for value in x_rank)
            y_sum_squares = sum((value - y_mean) ** 2 for value in y_rank)
            denominator = math.sqrt(x_sum_squares * y_sum_squares)
            rho = numerator / denominator if denominator else np.nan
        else:
            rho = np.nan

        if len(pair) < 30 or pd.isna(rho):
            status = "insufficient_pairs"
        elif expected_direction == "negative" and rho <= -0.1:
            status = "direction_consistent"
        elif expected_direction == "positive" and rho >= 0.1:
            status = "direction_consistent"
        elif abs(rho) < 0.1:
            status = "weak_or_inconclusive"
        else:
            status = "direction_conflict_requires_audit"

        rows.append(
            {
                "model_key": model_key,
                "input_variable": input_column,
                "output_variable": output_column,
                "pair_count": len(pair),
                "spearman_rho": rho,
                "expected_direction": expected_direction,
                "sanity_status": status,
                "physical_rationale": rationale,
                "interpretation_limit": (
                    "Association check only; mixed alloys, processes, and sources "
                    "prevent a causal interpretation."
                ),
            }
        )

    return pd.DataFrame(rows)


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
    model = pipeline.named_steps["model"]

    if sample_weight is not None and not isinstance(model, Ridge):
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        pipeline.fit(X_train, y_train)

    return pipeline


def train_one_target(
    model_key: str,
    target: str,
    dataset_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    remove_stale_model_files(
        model_key=model_key,
        target=target,
        active_model_names=set(models),
    )

    models_dir = get_path("outputs", "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    importance_frames = []
    error_rows = []

    for model_name, model in models.items():
        print(f"  Training {model_key} / {target} / {model_name}...")

        try:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
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
                    "source_count": prepared["source_count"],
                    "invalid_target_rows_removed": prepared[
                        "invalid_target_rows_removed"
                    ],
                    "sample_weight_used": (
                        w_train is not None and not isinstance(model, Ridge)
                    ),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "split_method": prepared["split_method"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "model_path": model_path.relative_to(get_path()).as_posix(),
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

        except Exception as exc:
            error_rows.append(
                {
                    "model_key": model_key,
                    "target": target,
                    "model": model_name,
                    "error": str(exc),
                }
            )
            print(f"    Skipped after error: {exc}")

    metrics_df = pd.DataFrame(metric_rows)

    if importance_frames:
        importance_df = pd.concat(importance_frames, ignore_index=True, sort=False)
    else:
        importance_df = pd.DataFrame(
            columns=["model_key", "target", "model", "feature", "importance"]
        )

    errors_df = pd.DataFrame(
        error_rows,
        columns=["model_key", "target", "model", "error"],
    )

    return metrics_df, importance_df, errors_df


def train_project_models(
    rebuild_views: bool = True,
    max_sn_rows_per_dataset_id: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    remove_legacy_model_files()

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
                metrics_df, importance_df, model_errors_df = train_one_target(
                    model_key=model_key,
                    target=target,
                )

                metric_frames.append(metrics_df)

                if not importance_df.empty:
                    importance_frames.append(importance_df)

                if not model_errors_df.empty:
                    error_rows.extend(model_errors_df.to_dict(orient="records"))

            except Exception as exc:
                error_rows.append(
                    {
                        "model_key": model_key,
                        "target": target,
                        "model": "",
                        "error": str(exc),
                    }
                )

    if metric_frames:
        all_metrics_df = pd.concat(metric_frames, ignore_index=True, sort=False)
        all_metrics_df = add_comparative_metrics(all_metrics_df)
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
        columns=["model_key", "target", "model", "error"],
    )

    tables_dir = get_path("outputs", "tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    build_training_data_audit().to_csv(
        tables_dir / "project_training_data_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_physical_sanity_checks().to_csv(
        tables_dir / "project_physical_sanity_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )

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

    make_best_model_summary(all_metrics_df).to_csv(
        tables_dir / "project_best_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return all_metrics_df, all_importance_df, errors_df
