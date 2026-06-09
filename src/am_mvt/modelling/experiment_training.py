from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from am_mvt.config import get_path
from am_mvt.modelling.basquin import (
    HierarchicalBasquin,
    basquin_residual_features,
    normalise_runout,
)
from am_mvt.modelling.experiment_config import (
    get_experiment_config,
    get_training_profile,
    iter_task_mode_targets,
    selected_modes,
)
from am_mvt.modelling.experiment_data import (
    assert_disjoint_groups,
    build_preprocessor,
    catboost_frame,
    clean_features,
    load_experiment_frame,
    make_group_folds,
    select_final_holdout_groups,
    select_usable_features,
    split_development_and_test,
)
from am_mvt.modelling.experiment_metrics import (
    conformal_radius,
    harrell_c_index,
    regression_metrics,
)


@dataclass
class AlloyFamilyMedianModel:
    global_median: float
    family_medians: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, target: str) -> "AlloyFamilyMedianModel":
        target_values = pd.to_numeric(frame[target], errors="coerce")
        family = (
            frame.get("alloy_family", pd.Series("missing", index=frame.index))
            .astype("string")
            .fillna("missing")
        )
        medians = (
            pd.DataFrame({"family": family, "target": target_values})
            .dropna(subset=["target"])
            .groupby("family")["target"]
            .median()
            .to_dict()
        )
        return cls(
            global_median=float(target_values.median()),
            family_medians={str(key): float(value) for key, value in medians.items()},
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        family = (
            frame.get("alloy_family", pd.Series("missing", index=frame.index))
            .astype("string")
            .fillna("missing")
        )
        return np.asarray(
            [
                self.family_medians.get(str(value), self.global_median)
                for value in family
            ],
            dtype=float,
        )


def filter_valid_fatigue_loading(frame: pd.DataFrame) -> pd.DataFrame:
    stress = pd.to_numeric(frame["stress_amplitude_MPa"], errors="coerce")
    return frame.loc[stress.between(1.0, 3000.0, inclusive="both")].reset_index(
        drop=True
    )


def get_model_candidates(profile: str) -> dict[str, dict[str, Any]]:
    profile_config = get_training_profile(profile)
    candidates = {
        "dummy_mean": {"kind": "sklearn", "model": DummyRegressor(strategy="mean")},
        "dummy_median": {
            "kind": "sklearn",
            "model": DummyRegressor(strategy="median"),
        },
        "alloy_family_median": {"kind": "alloy_median"},
        "linear_l2_sgd": {
            "kind": "sklearn",
            "model": SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-3,
                random_state=42,
                average=True,
            ),
        },
        "random_forest": {
            "kind": "sklearn",
            "model": RandomForestRegressor(
                n_estimators=int(profile_config["random_forest_estimators"]),
                max_depth=14,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        },
        "xgboost": {
            "kind": "sklearn",
            "model": XGBRegressor(
                n_estimators=int(profile_config["xgboost_estimators"]),
                max_depth=4,
                learning_rate=0.04,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
        },
    }
    candidates.update(
        {
            name: {
                "kind": "catboost",
                "params": params,
                "early_stopping_rounds": int(
                    profile_config["catboost_early_stopping_rounds"]
                ),
            }
            for name, params in dict(
                profile_config["catboost_candidates"]
            ).items()
        }
    )
    if bool(profile_config.get("include_mlp", False)):
        candidates["mlp_128_64_32"] = {
            "kind": "sklearn",
            "dense": True,
            "accepts_sample_weight": False,
            "model": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42,
            ),
        }
    return candidates


def get_sample_weight(frame: pd.DataFrame) -> np.ndarray | None:
    if "sample_weight" not in frame.columns:
        return None
    values = pd.to_numeric(frame["sample_weight"], errors="coerce").fillna(1.0)
    return values.to_numpy(dtype=float)


def fit_catboost(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict[str, Any],
    early_stopping_rounds: int,
    target_override: pd.Series | None = None,
):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "CatBoost is required. Install project requirements again."
        ) from exc

    train_x, medians = catboost_frame(
        train_df,
        numeric_features,
        categorical_features,
    )
    validation_x, _ = catboost_frame(
        validation_df,
        numeric_features,
        categorical_features,
        numeric_medians=medians,
    )
    train_y = (
        target_override.to_numpy(dtype=float)
        if target_override is not None
        else pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=float)
    )
    validation_y = pd.to_numeric(
        validation_df[target],
        errors="coerce",
    ).to_numpy(dtype=float)
    model_params = {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        **params,
    }
    model = CatBoostRegressor(**model_params)
    model.fit(
        train_x,
        train_y,
        cat_features=categorical_features,
        sample_weight=get_sample_weight(train_df),
        eval_set=(validation_x, validation_y),
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True,
        verbose=False,
    )
    return model, medians


def fit_catboost_full(
    train_df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict[str, Any],
    iterations: int,
    target_override: pd.Series | None = None,
):
    from catboost import CatBoostRegressor

    train_x, medians = catboost_frame(
        train_df,
        numeric_features,
        categorical_features,
    )
    train_y = (
        target_override.to_numpy(dtype=float)
        if target_override is not None
        else pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=float)
    )
    model_params = {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        **params,
        "iterations": max(1, int(iterations)),
    }
    model = CatBoostRegressor(**model_params)
    model.fit(
        train_x,
        train_y,
        cat_features=categorical_features,
        sample_weight=get_sample_weight(train_df),
        verbose=False,
    )
    return model, medians


def predict_catboost(
    model,
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    medians: dict[str, float],
) -> np.ndarray:
    features, _ = catboost_frame(
        frame,
        numeric_features,
        categorical_features,
        numeric_medians=medians,
    )
    return np.asarray(model.predict(features), dtype=float)


def refit_candidate_on_development(
    candidate: dict[str, Any],
    development_df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    if candidate["kind"] == "catboost":
        inner_train, inner_validation = make_inner_validation_split(development_df)
        tuned_model, _ = fit_catboost(
            inner_train,
            inner_validation,
            target,
            numeric_features,
            categorical_features,
            candidate["params"],
            int(candidate["early_stopping_rounds"]),
        )
        model, medians = fit_catboost_full(
            development_df,
            target,
            numeric_features,
            categorical_features,
            candidate["params"],
            iterations=tuned_model.tree_count_,
        )
        return {
            "kind": "catboost",
            "model": model,
            "numeric_medians": medians,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        }

    bundle, _ = fit_candidate(
        candidate,
        development_df,
        development_df,
        target,
        numeric_features,
        categorical_features,
    )
    return bundle


def fit_candidate(
    candidate: dict[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    kind = candidate["kind"]

    if kind == "alloy_median":
        model = AlloyFamilyMedianModel.fit(train_df, target)
        return {"kind": kind, "model": model}, model.predict(validation_df)

    if kind == "catboost":
        model, medians = fit_catboost(
            train_df,
            validation_df,
            target,
            numeric_features,
            categorical_features,
            candidate["params"],
            int(candidate["early_stopping_rounds"]),
        )
        return (
            {
                "kind": kind,
                "model": model,
                "numeric_medians": medians,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
            },
            predict_catboost(
                model,
                validation_df,
                numeric_features,
                categorical_features,
                medians,
            ),
        )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        sparse=not bool(candidate.get("dense", False)),
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", clone(candidate["model"])),
        ]
    )
    features = numeric_features + categorical_features
    train_clean = clean_features(train_df, numeric_features, categorical_features)
    validation_clean = clean_features(
        validation_df,
        numeric_features,
        categorical_features,
    )
    fit_params = {}
    if (
        not isinstance(candidate["model"], DummyRegressor)
        and candidate.get("accepts_sample_weight", True)
    ):
        sample_weight = get_sample_weight(train_df)
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight
    pipeline.fit(
        train_clean[features],
        pd.to_numeric(train_df[target], errors="coerce"),
        **fit_params,
    )
    return (
        {
            "kind": "sklearn",
            "model": pipeline,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        np.asarray(
            pipeline.predict(validation_clean[features]),
            dtype=float,
        ),
    )


def make_inner_validation_split(
    development_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_development_and_test(
        development_df,
        test_size=0.12,
        random_state=43,
    )


def feature_domain(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, object]:
    numeric_ranges = {}
    categorical_values = {}

    for column in numeric_features:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        numeric_ranges[column] = {
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
        }

    for column in categorical_features:
        categorical_values[column] = sorted(
            frame[column].astype("string").dropna().unique().tolist()
        )

    return {
        "numeric_ranges": numeric_ranges,
        "categorical_values": categorical_values,
    }


def split_audit_fields(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    return {
        "n_development": len(development_df),
        "n_final_test": len(test_df),
        "development_groups": int(
            development_df["evaluation_group_id"].nunique()
        ),
        "final_test_groups": int(test_df["evaluation_group_id"].nunique()),
    }


def select_candidate_by_oof(candidate_summary: pd.DataFrame) -> str:
    return str(
        candidate_summary.sort_values(
            ["oof_r2", "oof_rmse", "oof_mae"],
            ascending=[False, True, True],
            na_position="last",
        ).iloc[0]["candidate"]
    )


def train_conventional_experiment(
    model_key: str,
    target: str,
    mode: str,
    run_dir: Path,
    n_splits: int,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = get_experiment_config(model_key, target, mode)
    bounds = config["target_bounds"].get(target)
    frame = load_experiment_frame(config["dataset_path"], target, bounds)
    final_holdout_groups = None
    if model_key == "model2_sn_fatigue":
        frame = filter_valid_fatigue_loading(frame)
        final_holdout_groups = select_final_holdout_groups(frame)
        runout = normalise_runout(frame["runout"])
        frame = frame.loc[runout.eq(False)].reset_index(drop=True)
    numeric_features, categorical_features = select_usable_features(
        frame,
        config["numeric_features"],
        config["categorical_features"],
    )
    development_df, test_df = split_development_and_test(
        frame,
        test_groups=final_holdout_groups,
    )
    assert_disjoint_groups(development_df, test_df)
    print(
        "    Data rows: "
        f"total={len(frame)}, development={len(development_df)}, "
        f"final_test={len(test_df)}"
    )
    print(
        "    Evidence groups: "
        f"development={development_df['evaluation_group_id'].nunique()}, "
        f"final_test={test_df['evaluation_group_id'].nunique()}"
    )
    print(
        "    Features: "
        f"numeric={len(numeric_features)}, "
        f"categorical={len(categorical_features)}"
    )
    folds = make_group_folds(development_df, n_splits=n_splits)
    candidates = get_model_candidates(profile)
    cv_rows: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {
        name: np.full(len(development_df), np.nan)
        for name in candidates
    }

    for candidate_name, candidate in candidates.items():
        print(f"    CV {model_key}/{target}/{mode}/{candidate_name}")
        for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
            train_df = development_df.iloc[train_index].copy()
            validation_df = development_df.iloc[validation_index].copy()
            assert_disjoint_groups(train_df, validation_df)
            _, predictions = fit_candidate(
                candidate,
                train_df,
                validation_df,
                target,
                numeric_features,
                categorical_features,
            )
            oof_predictions[candidate_name][validation_index] = predictions
            metrics = regression_metrics(validation_df[target], predictions)
            cv_rows.append(
                {
                    "model_key": model_key,
                    "target": target,
                    "mode": mode,
                    "route": "ordinary_regression",
                    "candidate": candidate_name,
                    "fold": fold_number,
                    "n_train": len(train_df),
                    "n_validation": len(validation_df),
                    **metrics,
                }
            )

    cv_df = pd.DataFrame(cv_rows)
    candidate_summary = (
        cv_df.groupby("candidate")[["mae", "rmse", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    candidate_summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in candidate_summary.columns
    ]
    oof_rows = []
    for candidate_name, predictions in oof_predictions.items():
        oof_metrics = regression_metrics(development_df[target], predictions)
        oof_rows.append(
            {
                "candidate": candidate_name,
                "oof_mae": oof_metrics["mae"],
                "oof_rmse": oof_metrics["rmse"],
                "oof_r2": oof_metrics["r2"],
            }
        )
    candidate_summary = candidate_summary.merge(
        pd.DataFrame(oof_rows),
        on="candidate",
        how="left",
    )
    selected_name = select_candidate_by_oof(candidate_summary)
    selected_candidate = candidates[selected_name]
    conformal_q = conformal_radius(
        development_df[target],
        oof_predictions[selected_name],
        coverage=0.90,
    )
    final_bundle = refit_candidate_on_development(
        selected_candidate,
        development_df,
        target,
        numeric_features,
        categorical_features,
    )

    if final_bundle["kind"] == "catboost":
        test_predictions = predict_catboost(
            final_bundle["model"],
            test_df,
            numeric_features,
            categorical_features,
            final_bundle["numeric_medians"],
        )
    elif final_bundle["kind"] == "alloy_median":
        test_predictions = final_bundle["model"].predict(test_df)
    else:
        clean_test = clean_features(
            test_df,
            numeric_features,
            categorical_features,
        )
        test_predictions = final_bundle["model"].predict(
            clean_test[numeric_features + categorical_features]
        )

    test_metrics = regression_metrics(test_df[target], test_predictions)
    artifact_name = f"{model_key}__{target}__{mode}__ordinary.joblib"
    artifact_path = run_dir / "models" / artifact_name
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        **final_bundle,
        "model_key": model_key,
        "target": target,
        "mode": mode,
        "route": "ordinary_regression",
        "candidate": selected_name,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "conformal_q90": conformal_q,
        "feature_domain": feature_domain(
            development_df,
            numeric_features,
            categorical_features,
        ),
    }
    joblib.dump(bundle, artifact_path)

    summary_rows = []
    for _, row in candidate_summary.iterrows():
        summary_rows.append(
            {
                "model_key": model_key,
                "target": target,
                "mode": mode,
                "route": "ordinary_regression",
                "candidate": row["candidate"],
                "cv_mae_mean": row["mae_mean"],
                "cv_mae_std": row["mae_std"],
                "cv_rmse_mean": row["rmse_mean"],
                "cv_rmse_std": row["rmse_std"],
                "cv_r2_mean": row["r2_mean"],
                "cv_r2_std": row["r2_std"],
                "oof_mae": row["oof_mae"],
                "oof_rmse": row["oof_rmse"],
                "oof_r2": row["oof_r2"],
                "selected": row["candidate"] == selected_name,
                "test_mae": test_metrics["mae"]
                if row["candidate"] == selected_name
                else np.nan,
                "test_rmse": test_metrics["rmse"]
                if row["candidate"] == selected_name
                else np.nan,
                "test_r2": test_metrics["r2"]
                if row["candidate"] == selected_name
                else np.nan,
                "conformal_q90": conformal_q
                if row["candidate"] == selected_name
                else np.nan,
                "diagnostic_only": bool(config["diagnostic_only"]),
                **split_audit_fields(development_df, test_df),
                "artifact": artifact_path.relative_to(run_dir).as_posix()
                if row["candidate"] == selected_name
                else "",
            }
        )

    registry_entry = {
        "model_key": model_key,
        "target": target,
        "mode": mode,
        "route": "ordinary_regression",
        "candidate": selected_name,
        "artifact": artifact_path.relative_to(run_dir).as_posix(),
        "diagnostic_only": bool(config["diagnostic_only"]),
    }
    return cv_rows + summary_rows, registry_entry


def fit_residual_catboost(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    train_residual: pd.Series,
    validation_residual: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    profile: str,
):
    profile_config = get_training_profile(profile)
    residual_candidate_name = str(
        profile_config["basquin_residual_catboost"]
    )
    residual_params = dict(profile_config["catboost_candidates"])[
        residual_candidate_name
    ]
    validation_copy = validation_df.copy()
    validation_copy["_residual_target"] = validation_residual.to_numpy()
    train_copy = train_df.copy()
    train_copy["_residual_target"] = train_residual.to_numpy()
    return fit_catboost(
        train_copy,
        validation_copy,
        "_residual_target",
        numeric_features,
        categorical_features,
        residual_params,
        int(profile_config["catboost_early_stopping_rounds"]),
        target_override=train_copy["_residual_target"],
    )


def train_basquin_experiment(
    mode: str,
    run_dir: Path,
    n_splits: int,
    profile: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    model_key = "model2_sn_fatigue"
    target = "log10_fatigue_life_cycles"
    config = get_experiment_config(model_key, target, mode)
    frame = load_experiment_frame(
        config["dataset_path"],
        target,
        config["target_bounds"][target],
    )
    frame = filter_valid_fatigue_loading(frame)
    final_holdout_groups = select_final_holdout_groups(frame)
    runout = normalise_runout(frame["runout"])
    frame = frame.loc[runout.eq(False)].reset_index(drop=True)
    numeric_features, categorical_features = select_usable_features(
        frame,
        config["numeric_features"],
        config["categorical_features"],
    )
    residual_numeric, residual_categorical = basquin_residual_features(
        numeric_features,
        categorical_features,
    )
    development_df, test_df = split_development_and_test(
        frame,
        test_groups=final_holdout_groups,
    )
    assert_disjoint_groups(development_df, test_df)
    folds = make_group_folds(development_df, n_splits=n_splits)
    routes = ["basquin_only", "basquin_catboost_residual"]
    oof = {route: np.full(len(development_df), np.nan) for route in routes}
    rows: list[dict[str, Any]] = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        train_df = development_df.iloc[train_index].copy()
        validation_df = development_df.iloc[validation_index].copy()
        assert_disjoint_groups(train_df, validation_df)
        basquin = HierarchicalBasquin().fit(train_df)
        train_basquin = basquin.predict(train_df)
        validation_basquin = basquin.predict(validation_df)
        oof["basquin_only"][validation_index] = validation_basquin
        train_residual = pd.Series(
            pd.to_numeric(train_df[target], errors="coerce").to_numpy()
            - train_basquin,
            index=train_df.index,
        )
        validation_residual = pd.Series(
            pd.to_numeric(validation_df[target], errors="coerce").to_numpy()
            - validation_basquin,
            index=validation_df.index,
        )
        residual_model, medians = fit_residual_catboost(
            train_df,
            validation_df,
            train_residual,
            validation_residual,
            residual_numeric,
            residual_categorical,
            profile,
        )
        correction = predict_catboost(
            residual_model,
            validation_df,
            residual_numeric,
            residual_categorical,
            medians,
        )
        hybrid_prediction = validation_basquin + correction
        oof["basquin_catboost_residual"][validation_index] = hybrid_prediction

        for route, predictions in [
            ("basquin_only", validation_basquin),
            ("basquin_catboost_residual", hybrid_prediction),
        ]:
            rows.append(
                {
                    "model_key": model_key,
                    "target": target,
                    "mode": mode,
                    "route": route,
                    "candidate": route,
                    "fold": fold_number,
                    "n_train": len(train_df),
                    "n_validation": len(validation_df),
                    **regression_metrics(validation_df[target], predictions),
                }
            )

    registry_entries = []
    physical_checks = []
    final_train_df, early_stop_df = make_inner_validation_split(development_df)
    basquin = HierarchicalBasquin().fit(development_df)
    basquin_path = run_dir / "models" / f"fatigue__{mode}__basquin.json"
    basquin.save(basquin_path)
    basquin.parameters_frame().to_csv(
        run_dir / "tables" / f"basquin_parameters__{mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_basquin = basquin.predict(test_df)

    for route in routes:
        predictions = test_basquin
        artifact = basquin_path
        residual_model = None
        medians = None

        if route == "basquin_catboost_residual":
            inner_basquin = HierarchicalBasquin().fit(final_train_df)
            final_basquin = inner_basquin.predict(final_train_df)
            validation_basquin = inner_basquin.predict(early_stop_df)
            final_residual = pd.Series(
                final_train_df[target].to_numpy(dtype=float) - final_basquin,
                index=final_train_df.index,
            )
            validation_residual = pd.Series(
                early_stop_df[target].to_numpy(dtype=float) - validation_basquin,
                index=early_stop_df.index,
            )
            tuned_model, _ = fit_residual_catboost(
                final_train_df,
                early_stop_df,
                final_residual,
                validation_residual,
                residual_numeric,
                residual_categorical,
                profile,
            )
            development_residual = pd.Series(
                development_df[target].to_numpy(dtype=float)
                - basquin.predict(development_df),
                index=development_df.index,
            )
            profile_config = get_training_profile(profile)
            residual_candidate_name = str(
                profile_config["basquin_residual_catboost"]
            )
            residual_params = dict(profile_config["catboost_candidates"])[
                residual_candidate_name
            ]
            residual_model, medians = fit_catboost_full(
                development_df,
                target,
                residual_numeric,
                residual_categorical,
                residual_params,
                iterations=tuned_model.tree_count_,
                target_override=development_residual,
            )
            predictions = test_basquin + predict_catboost(
                residual_model,
                test_df,
                residual_numeric,
                residual_categorical,
                medians,
            )
            artifact = (
                run_dir
                / "models"
                / f"fatigue__{mode}__basquin_catboost_residual.joblib"
            )
            joblib.dump(
                {
                    "kind": "basquin_catboost_residual",
                    "model_key": model_key,
                    "target": target,
                    "mode": mode,
                    "route": route,
                    "basquin_path": basquin_path.relative_to(run_dir).as_posix(),
                    "model": residual_model,
                    "numeric_features": residual_numeric,
                    "categorical_features": residual_categorical,
                    "numeric_medians": medians,
                    "feature_domain": feature_domain(
                        development_df,
                        residual_numeric,
                        residual_categorical,
                    ),
                    "conformal_q90": conformal_radius(
                        development_df[target],
                        oof[route],
                    ),
                },
                artifact,
            )

        route_cv = pd.DataFrame(rows).loc[lambda value: value["route"].eq(route)]
        test_metrics = regression_metrics(test_df[target], predictions)
        conformal_q = conformal_radius(
            development_df[target],
            oof[route],
        )
        rows.append(
            {
                "model_key": model_key,
                "target": target,
                "mode": mode,
                "route": route,
                "candidate": route,
                "fold": "summary",
                "cv_mae_mean": route_cv["mae"].mean(),
                "cv_mae_std": route_cv["mae"].std(),
                "cv_rmse_mean": route_cv["rmse"].mean(),
                "cv_rmse_std": route_cv["rmse"].std(),
                "cv_r2_mean": route_cv["r2"].mean(),
                "cv_r2_std": route_cv["r2"].std(),
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "conformal_q90": conformal_q,
                "artifact": artifact.relative_to(run_dir).as_posix(),
                "evaluation_subset": "uncensored_failures_only",
                **split_audit_fields(development_df, test_df),
            }
        )
        stress = pd.to_numeric(
            development_df["stress_amplitude_MPa"],
            errors="coerce",
        )
        stress = stress.loc[stress.gt(0)].dropna()
        scan = np.geomspace(float(stress.min()), float(stress.max()), num=50)
        scenario = pd.concat(
            [development_df.iloc[[0]].copy()] * len(scan),
            ignore_index=True,
        )
        scenario["stress_amplitude_MPa"] = scan
        scan_predictions = basquin.predict(scenario)
        if residual_model is not None and medians is not None:
            scan_predictions = scan_predictions + predict_catboost(
                residual_model,
                scenario,
                residual_numeric,
                residual_categorical,
                medians,
            )
        parameters = basquin.parameters_frame()
        physical_checks.append(
            {
                "mode": mode,
                "route": route,
                "check": "basquin_negative_slope_and_stress_monotonicity",
                "all_curve_slopes_negative": bool(parameters["slope"].lt(0).all()),
                "stress_scan_monotonic_nonincreasing": bool(
                    np.all(np.diff(scan_predictions) <= 1e-10)
                ),
                "stress_scan_min_MPa": float(scan.min()),
                "stress_scan_max_MPa": float(scan.max()),
                "scan_points": len(scan),
            }
        )
        registry_entries.append(
            {
                "model_key": model_key,
                "target": target,
                "mode": mode,
                "route": route,
                "candidate": route,
                "artifact": artifact.relative_to(run_dir).as_posix(),
                "evaluation_subset": "uncensored_failures_only",
                "conformal_q90": conformal_q,
            }
        )

    return rows, registry_entries, physical_checks


def prepare_aft_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
):
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        sparse=True,
    )
    train_clean = clean_features(train_df, numeric_features, categorical_features)
    validation_clean = clean_features(
        validation_df,
        numeric_features,
        categorical_features,
    )
    columns = numeric_features + categorical_features
    train_x = preprocessor.fit_transform(train_clean[columns])
    validation_x = preprocessor.transform(validation_clean[columns])
    return preprocessor, train_x, validation_x


def make_aft_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = pd.to_numeric(frame["fatigue_life_cycles"], errors="coerce").to_numpy(
        dtype=float
    )
    runout = normalise_runout(frame["runout"]).fillna(False).to_numpy(dtype=bool)
    upper = lower.copy()
    upper[runout] = np.inf
    event = ~runout
    return lower, upper, event


def train_aft_booster(
    train_x,
    validation_x,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    profile: str,
):
    profile_config = get_training_profile(profile)
    train_lower, train_upper, _ = make_aft_bounds(train_df)
    validation_lower, validation_upper, _ = make_aft_bounds(validation_df)
    dtrain = xgb.DMatrix(train_x)
    dvalidation = xgb.DMatrix(validation_x)
    dtrain.set_float_info("label_lower_bound", train_lower)
    dtrain.set_float_info("label_upper_bound", train_upper)
    dvalidation.set_float_info("label_lower_bound", validation_lower)
    dvalidation.set_float_info("label_upper_bound", validation_upper)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist",
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 5.0,
        "seed": 42,
        "nthread": -1,
    }
    evaluations: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(profile_config["aft_boost_rounds"]),
        evals=[(dvalidation, "validation")],
        evals_result=evaluations,
        early_stopping_rounds=int(
            profile_config["aft_early_stopping_rounds"]
        ),
        verbose_eval=False,
    )
    return booster, dvalidation, evaluations


def train_aft_fixed_rounds(train_x, train_df: pd.DataFrame, rounds: int):
    train_lower, train_upper, _ = make_aft_bounds(train_df)
    dtrain = xgb.DMatrix(train_x)
    dtrain.set_float_info("label_lower_bound", train_lower)
    dtrain.set_float_info("label_upper_bound", train_upper)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist",
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 5.0,
        "seed": 42,
        "nthread": -1,
    }
    return xgb.train(
        params,
        dtrain,
        num_boost_round=max(1, int(rounds)),
        verbose_eval=False,
    )


def aft_metrics(
    frame: pd.DataFrame,
    predicted_cycles: np.ndarray,
    aft_nloglik: float,
) -> dict[str, float]:
    lower, _, event = make_aft_bounds(frame)
    log_true = np.log10(lower[event])
    log_pred = np.log10(np.maximum(predicted_cycles[event], 1.0))
    metrics = regression_metrics(log_true, log_pred)
    return {
        **metrics,
        "aft_nloglik": float(aft_nloglik),
        "harrell_c_index": harrell_c_index(lower, event, predicted_cycles),
        "uncensored_count": int(event.sum()),
        "runout_count": int((~event).sum()),
    }


def train_aft_experiment(
    mode: str,
    run_dir: Path,
    n_splits: int,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_key = "model2_sn_fatigue"
    target = "log10_fatigue_life_cycles"
    config = get_experiment_config(model_key, target, mode)
    frame = load_experiment_frame(
        config["dataset_path"],
        target,
        config["target_bounds"][target],
    )
    frame = frame.loc[
        pd.to_numeric(frame["fatigue_life_cycles"], errors="coerce").gt(0)
    ].reset_index(drop=True)
    frame = filter_valid_fatigue_loading(frame)
    final_holdout_groups = select_final_holdout_groups(frame)
    numeric_features, categorical_features = select_usable_features(
        frame,
        config["numeric_features"],
        config["categorical_features"],
    )
    development_df, test_df = split_development_and_test(
        frame,
        test_groups=final_holdout_groups,
    )
    assert_disjoint_groups(development_df, test_df)
    folds = make_group_folds(development_df, n_splits=n_splits)
    rows: list[dict[str, Any]] = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        train_df = development_df.iloc[train_index].copy()
        validation_df = development_df.iloc[validation_index].copy()
        assert_disjoint_groups(train_df, validation_df)
        _, train_x, validation_x = prepare_aft_features(
            train_df,
            validation_df,
            numeric_features,
            categorical_features,
        )
        booster, dvalidation, evaluations = train_aft_booster(
            train_x,
            validation_x,
            train_df,
            validation_df,
            profile,
        )
        predictions = booster.predict(dvalidation)
        nloglik = evaluations["validation"]["aft-nloglik"][
            booster.best_iteration
        ]
        rows.append(
            {
                "model_key": model_key,
                "target": target,
                "mode": mode,
                "route": "xgboost_aft",
                "candidate": "xgboost_aft",
                "fold": fold_number,
                "n_train": len(train_df),
                "n_validation": len(validation_df),
                **aft_metrics(validation_df, predictions, nloglik),
            }
        )

    final_train_df, early_stop_df = make_inner_validation_split(development_df)
    _, train_x, validation_x = prepare_aft_features(
        final_train_df,
        early_stop_df,
        numeric_features,
        categorical_features,
    )
    tuned_booster, _, _ = train_aft_booster(
        train_x,
        validation_x,
        final_train_df,
        early_stop_df,
        profile,
    )
    development_clean = clean_features(
        development_df,
        numeric_features,
        categorical_features,
    )
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        sparse=True,
    )
    development_x = preprocessor.fit_transform(
        development_clean[numeric_features + categorical_features]
    )
    booster = train_aft_fixed_rounds(
        development_x,
        development_df,
        tuned_booster.best_iteration + 1,
    )
    test_clean = clean_features(test_df, numeric_features, categorical_features)
    test_x = preprocessor.transform(test_clean[numeric_features + categorical_features])
    dtest = xgb.DMatrix(test_x)
    test_predictions = booster.predict(dtest)
    test_lower, test_upper, _ = make_aft_bounds(test_df)
    dtest.set_float_info("label_lower_bound", test_lower)
    dtest.set_float_info("label_upper_bound", test_upper)
    test_nloglik = float(
        booster.eval(dtest, name="test").split("aft-nloglik:")[-1]
    )
    test_metrics = aft_metrics(test_df, test_predictions, test_nloglik)
    model_path = run_dir / "models" / f"fatigue__{mode}__xgboost_aft.json"
    preprocessor_path = (
        run_dir / "models" / f"fatigue__{mode}__xgboost_aft_preprocessor.joblib"
    )
    booster.save_model(model_path)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_domain": feature_domain(
                development_df,
                numeric_features,
                categorical_features,
            ),
        },
        preprocessor_path,
    )
    cv_df = pd.DataFrame(rows)
    rows.append(
        {
            "model_key": model_key,
            "target": target,
            "mode": mode,
            "route": "xgboost_aft",
            "candidate": "xgboost_aft",
            "fold": "summary",
            "cv_mae_mean": cv_df["mae"].mean(),
            "cv_mae_std": cv_df["mae"].std(),
            "cv_rmse_mean": cv_df["rmse"].mean(),
            "cv_rmse_std": cv_df["rmse"].std(),
            "cv_r2_mean": cv_df["r2"].mean(),
            "cv_r2_std": cv_df["r2"].std(),
            "cv_aft_nloglik_mean": cv_df["aft_nloglik"].mean(),
            "cv_aft_nloglik_std": cv_df["aft_nloglik"].std(),
            "cv_harrell_c_index_mean": cv_df["harrell_c_index"].mean(),
            "cv_harrell_c_index_std": cv_df["harrell_c_index"].std(),
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "test_aft_nloglik": test_metrics["aft_nloglik"],
            "test_harrell_c_index": test_metrics["harrell_c_index"],
            **split_audit_fields(development_df, test_df),
            "artifact": model_path.relative_to(run_dir).as_posix(),
            "preprocessor_artifact": preprocessor_path.relative_to(
                run_dir
            ).as_posix(),
            "interval_note": (
                "AFT point prediction is censor-aware; no conformal interval is "
                "reported for this route."
            ),
        }
    )
    registry_entry = {
        "model_key": model_key,
        "target": target,
        "mode": mode,
        "route": "xgboost_aft",
        "candidate": "xgboost_aft",
        "artifact": model_path.relative_to(run_dir).as_posix(),
        "preprocessor_artifact": preprocessor_path.relative_to(run_dir).as_posix(),
    }
    return rows, registry_entry


def write_run_configuration(
    run_dir: Path,
    run_name: str,
    profile: str,
    n_splits: int,
    mode: str,
) -> None:
    profile_config = get_training_profile(profile)
    task_configs = []
    for model_key, target, selected_mode in iter_task_mode_targets(mode):
        config = get_experiment_config(model_key, target, selected_mode)
        task_configs.append(
            {
                "model_key": model_key,
                "target": target,
                "mode": selected_mode,
                "dataset_path": str(config["dataset_path"]),
                "numeric_features": config["numeric_features"],
                "categorical_features": config["categorical_features"],
                "diagnostic_only": config["diagnostic_only"],
            }
        )
    payload = {
        "run_name": run_name,
        "profile": profile,
        "mode_selection": mode,
        "profile_parameters": profile_config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": 42,
        "test_fraction": 0.15,
        "cv": f"{n_splits}-fold GroupKFold",
        "grouping": "DOI first, then dataset_id, then source_id",
        "selection_metric": "grouped-CV OOF R2; RMSE and MAE tie-breakers",
        "fatigue_stress_amplitude_bounds_MPa": [1.0, 3000.0],
        "ordinary_fatigue_subset": "uncensored failures",
        "prediction_modes": selected_modes(mode),
        "fatigue_routes": [
            "ordinary_regression",
            "xgboost_aft",
            "basquin_only",
            "basquin_catboost_residual",
        ],
        "task_configs": task_configs,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_experiment_suite(
    run_name: str,
    profile: str = "balanced",
    n_splits: int | None = None,
    mode: str = "process_only",
) -> Path:
    profile_config = get_training_profile(profile)
    modes = selected_modes(mode)
    if n_splits is None:
        n_splits = int(profile_config["default_cv_folds"])
    if n_splits < 2:
        raise ValueError("cv folds must be at least 2.")

    run_dir = get_path("outputs", "experiments", run_name)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Experiment directory already exists and is not empty: {run_dir}"
        )
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    write_run_configuration(run_dir, run_name, profile, n_splits, mode)
    all_rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    physical_checks: list[dict[str, Any]] = []

    for model_key, target, selected_mode in iter_task_mode_targets(mode):
        print(
            f"\nTraining ordinary route: "
            f"{model_key} / {target} / {selected_mode}"
        )
        rows, entry = train_conventional_experiment(
            model_key,
            target,
            selected_mode,
            run_dir,
            n_splits,
            profile,
        )
        all_rows.extend(rows)
        registry.append(entry)

    for selected_mode in modes:
        print(f"\nTraining Basquin routes: {selected_mode}")
        rows, entries, checks = train_basquin_experiment(
            selected_mode,
            run_dir,
            n_splits,
            profile,
        )
        all_rows.extend(rows)
        registry.extend(entries)
        physical_checks.extend(checks)

        print(f"\nTraining XGBoost-AFT route: {selected_mode}")
        rows, entry = train_aft_experiment(
            selected_mode,
            run_dir,
            n_splits,
            profile,
        )
        all_rows.extend(rows)
        registry.append(entry)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(
        run_dir / "tables" / "experiment_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df = metrics_df.loc[
        metrics_df.get(
            "cv_mae_mean",
            pd.Series(np.nan, index=metrics_df.index),
        ).notna()
        | metrics_df.get(
            "fold",
            pd.Series(index=metrics_df.index, dtype="object"),
        ).astype(str).eq("summary")
    ].copy()
    summary_df.to_csv(
        run_dir / "tables" / "experiment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(physical_checks).to_csv(
        run_dir / "tables" / "physical_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (run_dir / "model_registry.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    return run_dir
