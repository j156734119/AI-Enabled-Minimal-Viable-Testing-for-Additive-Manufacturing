from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
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
from am_mvt.modelling.fatigue_protocol import (
    FATIGUE_THRESHOLDS,
    aft_life_quantile,
    aft_survival_probability,
    calibrate_threshold_probability,
    e606_assessment,
    fatigue_protocol_audit,
    fit_isotonic_calibration,
    protocolise_fatigue_data,
    regime_summary,
    threshold_labels,
)
from am_mvt.utils.artifacts import sha256_file


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


@lru_cache(maxsize=1)
def mlp_runtime_available() -> bool:
    probe = (
        "import numpy as np; "
        "a=np.ones((2,2),dtype=float); "
        "assert (a @ a).shape == (2,2)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def get_model_candidates(
    profile: str,
    *,
    verify_mlp_runtime: bool = False,
) -> dict[str, dict[str, Any]]:
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
            for name, params in dict(profile_config["catboost_candidates"]).items()
        }
    )
    include_mlp = bool(profile_config.get("include_mlp", False))
    if include_mlp and verify_mlp_runtime and not mlp_runtime_available():
        include_mlp = False
        print(
            "    WARNING: MLP skipped because the NumPy/BLAS runtime failed "
            "an isolated matrix-multiplication probe."
        )
    if include_mlp:
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
    if not isinstance(candidate["model"], DummyRegressor) and candidate.get(
        "accepts_sample_weight", True
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
        "development_groups": int(development_df["evaluation_group_id"].nunique()),
        "final_test_groups": int(test_df["evaluation_group_id"].nunique()),
    }


def dataset_provenance(
    dataset_path: Path,
    frame: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    project_root = get_path().resolve()
    resolved = dataset_path.resolve()
    return {
        "dataset_path": resolved.relative_to(project_root).as_posix(),
        "dataset_sha256": sha256_file(resolved),
        "eligible_rows": len(frame),
        "eligible_groups": int(frame["evaluation_group_id"].nunique()),
        "final_test_groups": sorted(
            test_df["evaluation_group_id"].dropna().astype(str).unique().tolist()
        ),
    }


def select_candidate_by_oof(candidate_summary: pd.DataFrame) -> str:
    return str(
        candidate_summary.sort_values(
            ["oof_r2", "oof_rmse", "oof_mae"],
            ascending=[False, True, True],
            na_position="last",
        ).iloc[0]["candidate"]
    )


def route_oof_frame(
    development_df: pd.DataFrame,
    *,
    target: str,
    model_key: str,
    mode: str,
    route: str,
    candidate: str,
    fold: np.ndarray,
    predictions: np.ndarray,
    conformal_q: float | None,
    censored: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    evidence_columns = [
        "source_id",
        "source_name",
        "source_file",
        "dataset_id",
        "record_id",
        "doi",
        "source_title",
        "alloy",
        "alloy_family",
        "am_process",
        "machine_model",
        "am_environment",
        "laser_power_W",
        "scan_speed_mm_s",
        "hatch_spacing_um",
        "layer_thickness_um",
        "ved_J_mm3",
        "build_orientation",
        "test_direction",
        "scan_strategy",
        "heat_treatment",
        "material_state",
        "surface_condition",
        "surface_roughness_Ra_um",
        "surface_roughness_Rz_um",
        "post_processing",
        "porosity_percent",
        "relative_density_percent",
        "defect_type",
        "residual_stress_indicator",
        "residual_stress_MPa",
        "specimen_description",
        "specimen_geometry",
        "critical_section_dimensions_mm",
        "critical_section_size_mm",
        "stress_concentration_factor",
        "fatigue_environment",
        "fatigue_machine",
        "fatigue_standard",
        "load_control",
        "test_type",
        "stress_amplitude_MPa",
        "max_stress_MPa",
        "r_ratio",
        "frequency_Hz",
        "test_temperature_C",
        "runout",
        "fatigue_protocol",
        "control_mode",
        "frequency_regime",
        "event_observed",
        "censor_lower_cycles",
        "runout_limit_cycles",
        "stress_definition",
        "r_ratio_bin",
        "evaluation_group_id",
        "modelling_group_id",
    ]
    available = [column for column in evidence_columns if column in development_df]
    result = development_df[available].copy()
    result.insert(0, "model_key", model_key)
    result.insert(1, "target", target)
    result.insert(2, "mode", mode)
    result.insert(3, "route", route)
    result.insert(4, "candidate", candidate)
    result.insert(5, "fold", fold)
    result["y_true"] = pd.to_numeric(development_df[target], errors="coerce").to_numpy()
    result["y_pred"] = np.asarray(predictions, dtype=float)
    is_censored = (
        pd.Series(censored, index=development_df.index).fillna(False).astype(bool)
        if censored is not None
        else pd.Series(False, index=development_df.index)
    )
    result["is_censored"] = is_censored.to_numpy()
    result["abs_error"] = np.where(
        result["is_censored"],
        np.nan,
        np.abs(result["y_true"] - result["y_pred"]),
    )
    if conformal_q is None or not np.isfinite(conformal_q):
        result["interval_lower_90"] = np.nan
        result["interval_upper_90"] = np.nan
        result["interval_hit_90"] = pd.NA
        result["conformal_q90"] = np.nan
    else:
        result["interval_lower_90"] = result["y_pred"] - conformal_q
        result["interval_upper_90"] = result["y_pred"] + conformal_q
        result["interval_hit_90"] = result["y_true"].between(
            result["interval_lower_90"],
            result["interval_upper_90"],
            inclusive="both",
        )
        result["conformal_q90"] = conformal_q
    return result


def train_conventional_experiment(
    model_key: str,
    target: str,
    mode: str,
    run_dir: Path,
    n_splits: int,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    config = get_experiment_config(model_key, target, mode)
    bounds = config["target_bounds"].get(target)
    frame = load_experiment_frame(config["dataset_path"], target, bounds)
    final_holdout_groups = None
    if model_key == "model2_sn_fatigue":
        frame = filter_valid_fatigue_loading(frame)
        frame = protocolise_fatigue_data(frame)
        frame = frame.loc[
            frame["fatigue_protocol"].eq("e466_conventional")
            & ~frame["stress_consistency_status"].eq("review_required")
        ].reset_index(drop=True)
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
    oof_fold = np.full(len(development_df), -1, dtype=int)
    for fold_number, (_, validation_index) in enumerate(folds, start=1):
        oof_fold[validation_index] = fold_number
    candidates = get_model_candidates(profile, verify_mlp_runtime=True)
    cv_rows: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {
        name: np.full(len(development_df), np.nan) for name in candidates
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
        "target_bounds": bounds,
        **dataset_provenance(
            Path(config["dataset_path"]),
            frame,
            test_df,
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
        "target_bounds": bounds,
        **dataset_provenance(
            Path(config["dataset_path"]),
            frame,
            test_df,
        ),
    }
    selected_oof = np.asarray(oof_predictions[selected_name], dtype=float)
    evidence_columns = [
        "source_id",
        "source_name",
        "source_file",
        "dataset_id",
        "record_id",
        "doi",
        "source_title",
        "alloy",
        "alloy_family",
        "am_process",
        "machine_model",
        "laser_power_W",
        "scan_speed_mm_s",
        "hatch_spacing_um",
        "layer_thickness_um",
        "ved_J_mm3",
        "build_orientation",
        "test_direction",
        "scan_strategy",
        "heat_treatment",
        "surface_condition",
        "post_processing",
        "porosity_percent",
        "relative_density_percent",
        "defect_type",
        "stress_amplitude_MPa",
        "max_stress_MPa",
        "r_ratio",
        "frequency_Hz",
        "test_temperature_C",
        "runout",
        "evaluation_group_id",
    ]
    available_columns = [
        column for column in evidence_columns if column in development_df.columns
    ]
    oof_frame = development_df[available_columns].copy()
    oof_frame.insert(0, "model_key", model_key)
    oof_frame.insert(1, "target", target)
    oof_frame.insert(2, "mode", mode)
    oof_frame.insert(3, "route", "ordinary_regression")
    oof_frame.insert(4, "candidate", selected_name)
    oof_frame.insert(5, "fold", oof_fold)
    oof_frame["y_true"] = pd.to_numeric(
        development_df[target], errors="coerce"
    ).to_numpy()
    oof_frame["y_pred"] = selected_oof
    oof_frame["abs_error"] = np.abs(oof_frame["y_true"] - selected_oof)
    oof_frame["interval_lower_90"] = selected_oof - conformal_q
    oof_frame["interval_upper_90"] = selected_oof + conformal_q
    oof_frame["interval_hit_90"] = oof_frame["y_true"].between(
        oof_frame["interval_lower_90"],
        oof_frame["interval_upper_90"],
        inclusive="both",
    )
    oof_frame["conformal_q90"] = conformal_q
    return cv_rows + summary_rows, registry_entry, oof_frame


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
    residual_candidate_name = str(profile_config["basquin_residual_catboost"])
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
    pd.DataFrame,
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
    frame = protocolise_fatigue_data(frame)
    frame = frame.loc[
        frame["fatigue_protocol"].eq("e466_conventional")
        & ~frame["stress_consistency_status"].eq("review_required")
    ].reset_index(drop=True)
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
    oof_fold = np.full(len(development_df), -1, dtype=int)
    rows: list[dict[str, Any]] = []

    for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
        oof_fold[validation_index] = fold_number
        train_df = development_df.iloc[train_index].copy()
        validation_df = development_df.iloc[validation_index].copy()
        assert_disjoint_groups(train_df, validation_df)
        basquin = HierarchicalBasquin().fit(train_df)
        train_basquin = basquin.predict(train_df)
        validation_basquin = basquin.predict(validation_df)
        oof["basquin_only"][validation_index] = validation_basquin
        train_residual = pd.Series(
            pd.to_numeric(train_df[target], errors="coerce").to_numpy() - train_basquin,
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
            residual_candidate_name = str(profile_config["basquin_residual_catboost"])
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

    oof_frames = []
    for route in routes:
        route_frame = route_oof_frame(
            development_df,
            target=target,
            model_key=model_key,
            mode=mode,
            route=route,
            candidate=route,
            fold=oof_fold,
            predictions=oof[route],
            conformal_q=conformal_radius(development_df[target], oof[route]),
        )
        oof_frames.append(route_frame)
    return (
        rows,
        registry_entries,
        physical_checks,
        pd.concat(
            oof_frames,
            ignore_index=True,
        ),
    )


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


def aft_monotone_constraints(
    transformed_feature_count: int,
    numeric_features: list[str],
) -> str:
    constraints = [
        -1 if feature == "log10_stress_amplitude" else 0 for feature in numeric_features
    ]
    constraints.extend([0] * (transformed_feature_count - len(constraints)))
    return "(" + ",".join(str(value) for value in constraints) + ")"


def make_aft_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = pd.to_numeric(frame["fatigue_life_cycles"], errors="coerce").to_numpy(
        dtype=float
    )
    runout_values = normalise_runout(frame["runout"])
    if runout_values.isna().any():
        raise ValueError("AFT bounds require an explicit failure or runout status.")
    runout = runout_values.to_numpy(dtype=bool)
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
    *,
    distribution: str = "normal",
    scale: float = 1.0,
    monotone_constraints: str | None = None,
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
        "aft_loss_distribution": distribution,
        "aft_loss_distribution_scale": float(scale),
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
    if monotone_constraints:
        params["monotone_constraints"] = monotone_constraints
    evaluations: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(profile_config["aft_boost_rounds"]),
        evals=[(dvalidation, "validation")],
        evals_result=evaluations,
        early_stopping_rounds=int(profile_config["aft_early_stopping_rounds"]),
        verbose_eval=False,
    )
    return booster, dvalidation, evaluations


def train_aft_fixed_rounds(
    train_x,
    train_df: pd.DataFrame,
    rounds: int,
    *,
    distribution: str = "normal",
    scale: float = 1.0,
    monotone_constraints: str | None = None,
):
    train_lower, train_upper, _ = make_aft_bounds(train_df)
    dtrain = xgb.DMatrix(train_x)
    dtrain.set_float_info("label_lower_bound", train_lower)
    dtrain.set_float_info("label_upper_bound", train_upper)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": distribution,
        "aft_loss_distribution_scale": float(scale),
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
    if monotone_constraints:
        params["monotone_constraints"] = monotone_constraints
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
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
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
    frame = protocolise_fatigue_data(frame)
    audit = fatigue_protocol_audit(frame)
    audit.to_csv(
        run_dir / "tables" / "fatigue_protocol_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regime_summary(frame).to_csv(
        run_dir / "tables" / "fatigue_regime_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{"assessment": e606_assessment(frame)}]).to_csv(
        run_dir / "tables" / "fatigue_e606_assessment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    frame = frame.loc[
        frame["fatigue_protocol"].eq("e466_conventional")
        & frame["event_observed"].notna()
        & ~frame["stress_consistency_status"].eq("review_required")
    ].reset_index(drop=True)
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
    oof_fold = np.full(len(development_df), -1, dtype=int)
    candidates = [
        (distribution, scale)
        for distribution in ("normal", "logistic", "extreme")
        for scale in (0.5, 1.0, 1.5)
    ]
    candidate_predictions: dict[str, np.ndarray] = {
        f"{distribution}_scale_{scale:g}": np.full(len(development_df), np.nan)
        for distribution, scale in candidates
    }
    return _continue_aft_training(
        candidates=candidates,
        candidate_predictions=candidate_predictions,
        rows=rows,
        folds=folds,
        oof_fold=oof_fold,
        development_df=development_df,
        test_df=test_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_key=model_key,
        target=target,
        mode=mode,
        run_dir=run_dir,
        profile=profile,
    )


def _fatigue_domain_key(row: pd.Series, level: str) -> str:
    if level == "exact":
        values = [
            row.get("alloy"),
            row.get("am_process"),
            row.get("fatigue_protocol"),
            row.get("r_ratio_bin"),
        ]
    elif level == "family":
        values = [
            row.get("alloy_family"),
            row.get("fatigue_protocol"),
            row.get("r_ratio_bin"),
        ]
    elif level == "protocol":
        values = [row.get("fatigue_protocol")]
    else:
        raise ValueError(f"Unknown fatigue domain level: {level}")
    return "|".join("missing" if pd.isna(value) else str(value) for value in values)


def fatigue_domain_support(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    definitions = {
        "exact": ["alloy", "am_process", "fatigue_protocol", "r_ratio_bin"],
        "family": ["alloy_family", "fatigue_protocol", "r_ratio_bin"],
    }
    for level, columns in definitions.items():
        for values, group in frame.groupby(columns, dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            descriptor = pd.Series(dict(zip(columns, values)))
            records = len(group)
            groups = int(group["evaluation_group_id"].nunique())
            stress_levels = int(
                pd.to_numeric(group["stress_amplitude_MPa"], errors="coerce")
                .round(6)
                .nunique()
            )
            result.append(
                {
                    "level": level,
                    "key": _fatigue_domain_key(descriptor, level),
                    "records": records,
                    "dataset_groups": groups,
                    "stress_levels": stress_levels,
                    "eligible": bool(
                        records >= 150 and groups >= 20 and stress_levels >= 4
                    ),
                }
            )
    return result


def train_aft_domain_models(
    development_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    distribution: str,
    scale: float,
    rounds: int,
    run_dir: Path,
    mode: str,
) -> list[dict[str, Any]]:
    support = fatigue_domain_support(development_df)
    manifests: list[dict[str, Any]] = []
    for item in support:
        if not item["eligible"]:
            continue
        level = str(item["level"])
        key = str(item["key"])
        mask = development_df.apply(
            lambda row: _fatigue_domain_key(row, level) == key,
            axis=1,
        )
        domain = development_df.loc[mask].reset_index(drop=True)
        domain_numeric, domain_categorical = select_usable_features(
            domain,
            numeric_features,
            categorical_features,
        )
        clean = clean_features(domain, domain_numeric, domain_categorical)
        preprocessor = build_preprocessor(
            domain_numeric,
            domain_categorical,
            sparse=True,
        )
        transformed = preprocessor.fit_transform(
            clean[domain_numeric + domain_categorical]
        )
        constraints = aft_monotone_constraints(
            transformed.shape[1],
            domain_numeric,
        )
        booster = train_aft_fixed_rounds(
            transformed,
            domain,
            rounds,
            distribution=distribution,
            scale=scale,
            monotone_constraints=constraints,
        )
        digest = hashlib.sha256(f"{level}::{key}".encode("utf-8")).hexdigest()[:12]
        model_path = (
            run_dir / "models" / f"fatigue__{mode}__xgboost_aft__{level}_{digest}.json"
        )
        metadata_path = (
            run_dir
            / "models"
            / f"fatigue__{mode}__xgboost_aft__{level}_{digest}.joblib"
        )
        booster.save_model(model_path)
        joblib.dump(
            {
                "preprocessor": preprocessor,
                "numeric_features": domain_numeric,
                "categorical_features": domain_categorical,
                "feature_domain": feature_domain(
                    domain,
                    domain_numeric,
                    domain_categorical,
                ),
            },
            metadata_path,
        )
        manifests.append(
            {
                **item,
                "artifact": model_path.relative_to(run_dir).as_posix(),
                "preprocessor_artifact": metadata_path.relative_to(run_dir).as_posix(),
            }
        )
    return manifests


def build_threshold_calibrations(
    frame: pd.DataFrame,
    oof_predictions: np.ndarray,
    distribution: str,
    scale: float,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    calibrations: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    definitions = {
        "exact": ["alloy", "am_process", "fatigue_protocol", "r_ratio_bin"],
        "family": ["alloy_family", "fatigue_protocol", "r_ratio_bin"],
        "protocol": ["fatigue_protocol"],
    }
    for threshold in FATIGUE_THRESHOLDS:
        raw = pd.Series(
            aft_survival_probability(
                oof_predictions,
                threshold,
                distribution,
                scale,
            ),
            index=frame.index,
        )
        labels = threshold_labels(frame, threshold)
        for level, columns in definitions.items():
            for _, group in frame.groupby(columns, dropna=False):
                key = _fatigue_domain_key(group.iloc[0], level)
                calibration = fit_isotonic_calibration(
                    raw.loc[group.index],
                    labels.loc[group.index],
                    level=level,
                    key=key,
                )
                labelled = labels.loc[group.index].dropna()
                positives = int(labelled.eq(1).sum())
                negatives = int(labelled.eq(0).sum())
                row = {
                    "threshold_cycles": threshold,
                    "level": level,
                    "key": key,
                    "labelled_records": len(labelled),
                    "positive_count": positives,
                    "negative_count": negatives,
                    "calibration_available": calibration is not None,
                }
                if calibration is not None:
                    lookup = f"{int(threshold)}::{level}::{key}"
                    calibrations[lookup] = calibration.to_dict()
                    calibrated = calibration.predict(raw.loc[labelled.index].to_numpy())
                    row["raw_brier"] = float(
                        np.mean(np.square(raw.loc[labelled.index] - labelled))
                    )
                    row["calibrated_brier"] = float(
                        np.mean(np.square(calibrated - labelled.to_numpy()))
                    )
                rows.append(row)
    return calibrations, pd.DataFrame(rows)


def _continue_aft_training(
    *,
    candidates: list[tuple[str, float]],
    candidate_predictions: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    folds: list[tuple[np.ndarray, np.ndarray]],
    oof_fold: np.ndarray,
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    model_key: str,
    target: str,
    mode: str,
    run_dir: Path,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    candidate_rounds: dict[str, list[int]] = {key: [] for key in candidate_predictions}

    for distribution, scale in candidates:
        candidate = f"{distribution}_scale_{scale:g}"
        print(f"    CV xgboost_aft/{mode}/{candidate}")
        for fold_number, (train_index, validation_index) in enumerate(
            folds,
            start=1,
        ):
            oof_fold[validation_index] = fold_number
            train_df = development_df.iloc[train_index].copy()
            validation_df = development_df.iloc[validation_index].copy()
            assert_disjoint_groups(train_df, validation_df)
            _, train_x, validation_x = prepare_aft_features(
                train_df,
                validation_df,
                numeric_features,
                categorical_features,
            )
            constraints = aft_monotone_constraints(
                train_x.shape[1],
                numeric_features,
            )
            booster, dvalidation, evaluations = train_aft_booster(
                train_x,
                validation_x,
                train_df,
                validation_df,
                profile,
                distribution=distribution,
                scale=scale,
                monotone_constraints=constraints,
            )
            predictions = booster.predict(dvalidation)
            candidate_predictions[candidate][validation_index] = predictions
            candidate_rounds[candidate].append(booster.best_iteration + 1)
            nloglik = evaluations["validation"]["aft-nloglik"][booster.best_iteration]
            rows.append(
                {
                    "model_key": model_key,
                    "target": target,
                    "mode": mode,
                    "route": "xgboost_aft",
                    "candidate": candidate,
                    "aft_distribution": distribution,
                    "aft_scale": scale,
                    "fold": fold_number,
                    "n_train": len(train_df),
                    "n_validation": len(validation_df),
                    **aft_metrics(validation_df, predictions, nloglik),
                }
            )

    fold_metrics = pd.DataFrame(rows)
    comparison = (
        fold_metrics.groupby(
            ["candidate", "aft_distribution", "aft_scale"],
            as_index=False,
        )
        .agg(
            grouped_oof_aft_nloglik=("aft_nloglik", "mean"),
            grouped_oof_c_index=("harrell_c_index", "mean"),
            grouped_oof_log_mae=("mae", "mean"),
            grouped_oof_log_r2=("r2", "mean"),
        )
        .sort_values(
            ["grouped_oof_aft_nloglik", "grouped_oof_c_index"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )
    comparison["selected"] = False
    comparison.loc[0, "selected"] = True
    comparison.to_csv(
        run_dir / "tables" / "fatigue_route_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_candidate = str(comparison.iloc[0]["candidate"])
    selected_distribution = str(comparison.iloc[0]["aft_distribution"])
    selected_scale = float(comparison.iloc[0]["aft_scale"])
    oof_predictions = candidate_predictions[selected_candidate]

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
        distribution=selected_distribution,
        scale=selected_scale,
        monotone_constraints=aft_monotone_constraints(
            train_x.shape[1],
            numeric_features,
        ),
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
    selected_rounds = int(np.median(candidate_rounds[selected_candidate]))
    booster = train_aft_fixed_rounds(
        development_x,
        development_df,
        selected_rounds,
        distribution=selected_distribution,
        scale=selected_scale,
        monotone_constraints=aft_monotone_constraints(
            development_x.shape[1],
            numeric_features,
        ),
    )
    test_clean = clean_features(test_df, numeric_features, categorical_features)
    test_x = preprocessor.transform(test_clean[numeric_features + categorical_features])
    dtest = xgb.DMatrix(test_x)
    test_predictions = booster.predict(dtest)
    test_lower, test_upper, _ = make_aft_bounds(test_df)
    dtest.set_float_info("label_lower_bound", test_lower)
    dtest.set_float_info("label_upper_bound", test_upper)
    test_nloglik = float(booster.eval(dtest, name="test").split("aft-nloglik:")[-1])
    test_metrics = aft_metrics(test_df, test_predictions, test_nloglik)
    calibrations, calibration_table = build_threshold_calibrations(
        development_df,
        oof_predictions,
        selected_distribution,
        selected_scale,
    )
    calibration_table.to_csv(
        run_dir / "tables" / "fatigue_threshold_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    domain_support = fatigue_domain_support(development_df)
    domain_models = train_aft_domain_models(
        development_df,
        numeric_features,
        categorical_features,
        distribution=selected_distribution,
        scale=selected_scale,
        rounds=selected_rounds,
        run_dir=run_dir,
        mode=mode,
    )
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
            "aft_distribution": selected_distribution,
            "aft_scale": selected_scale,
            "threshold_calibrations": calibrations,
            "domain_support": domain_support,
            "domain_models": domain_models,
            "training_rounds": selected_rounds,
            "fatigue_protocol": "e466_conventional",
        },
        preprocessor_path,
    )
    cv_df = pd.DataFrame(rows).loc[
        lambda data: data["candidate"].eq(selected_candidate)
    ]
    rows.append(
        {
            "model_key": model_key,
            "target": target,
            "mode": mode,
            "route": "xgboost_aft",
            "candidate": selected_candidate,
            "aft_distribution": selected_distribution,
            "aft_scale": selected_scale,
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
            "preprocessor_artifact": preprocessor_path.relative_to(run_dir).as_posix(),
            "interval_note": (
                "AFT quantiles use the selected survival distribution; threshold "
                "probabilities use OOF isotonic calibration when supported."
            ),
        }
    )
    registry_entry = {
        "model_key": model_key,
        "target": target,
        "mode": mode,
        "route": "xgboost_aft",
        "candidate": selected_candidate,
        "aft_distribution": selected_distribution,
        "aft_scale": selected_scale,
        "fatigue_protocol": "e466_conventional",
        "artifact": model_path.relative_to(run_dir).as_posix(),
        "preprocessor_artifact": preprocessor_path.relative_to(run_dir).as_posix(),
        "domain_models": domain_models,
    }
    oof_frame = route_oof_frame(
        development_df,
        target=target,
        model_key=model_key,
        mode=mode,
        route="xgboost_aft",
        candidate=selected_candidate,
        fold=oof_fold,
        predictions=np.log10(np.maximum(oof_predictions, 1.0)),
        conformal_q=None,
        censored=normalise_runout(development_df["runout"]),
    )
    for probability in (0.10, 0.20, 0.50, 0.80, 0.90):
        oof_frame[
            f"life_quantile_{int(probability * 100):02d}_cycles"
        ] = aft_life_quantile(
            oof_predictions,
            probability,
            selected_distribution,
            selected_scale,
        )
    for threshold in FATIGUE_THRESHOLDS:
        suffix = f"{int(threshold / 1_000_000)}m"
        raw = aft_survival_probability(
            oof_predictions,
            threshold,
            selected_distribution,
            selected_scale,
        )
        calibrated = []
        calibration_levels = []
        for index, (_, row) in enumerate(development_df.iterrows()):
            value, level = calibrate_threshold_probability(
                float(raw[index]),
                row,
                threshold,
                calibrations,
            )
            calibrated.append(value)
            calibration_levels.append(level)
        oof_frame[f"raw_probability_reach_{suffix}"] = raw
        oof_frame[f"probability_reach_{suffix}"] = calibrated
        oof_frame[f"calibration_level_{suffix}"] = calibration_levels
    return rows, registry_entry, oof_frame


def augment_aft_domain_models(
    run_dir: str | Path,
    mode: str = "process_only",
) -> int:
    run_dir = Path(run_dir)
    registry = json.loads((run_dir / "model_registry.json").read_text(encoding="utf-8"))
    entry = next(
        item
        for item in registry
        if item.get("route") == "xgboost_aft" and item.get("mode") == mode
    )
    metadata_path = run_dir / str(entry["preprocessor_artifact"])
    metadata = joblib.load(metadata_path)
    config = get_experiment_config(
        "model2_sn_fatigue",
        "log10_fatigue_life_cycles",
        mode,
    )
    frame = load_experiment_frame(
        config["dataset_path"],
        "log10_fatigue_life_cycles",
        config["target_bounds"]["log10_fatigue_life_cycles"],
    )
    frame = frame.loc[
        pd.to_numeric(frame["fatigue_life_cycles"], errors="coerce").gt(0)
    ].reset_index(drop=True)
    frame = protocolise_fatigue_data(filter_valid_fatigue_loading(frame))
    frame = frame.loc[
        frame["fatigue_protocol"].eq("e466_conventional")
        & frame["event_observed"].notna()
        & ~frame["stress_consistency_status"].eq("review_required")
    ].reset_index(drop=True)
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    task = next(
        item
        for item in run_config["task_configs"]
        if item["target"] == "log10_fatigue_life_cycles" and item["mode"] == mode
    )
    development_df, _ = split_development_and_test(
        frame,
        test_groups=set(task["final_test_groups"]),
    )
    booster = xgb.Booster()
    booster.load_model(run_dir / str(entry["artifact"]))
    rounds = int(metadata.get("training_rounds", booster.num_boosted_rounds()))
    domain_models = train_aft_domain_models(
        development_df,
        list(metadata["numeric_features"]),
        list(metadata["categorical_features"]),
        distribution=str(metadata.get("aft_distribution", "normal")),
        scale=float(metadata.get("aft_scale", 1.0)),
        rounds=rounds,
        run_dir=run_dir,
        mode=mode,
    )
    metadata["domain_models"] = domain_models
    metadata["training_rounds"] = rounds
    joblib.dump(metadata, metadata_path)
    entry["domain_models"] = domain_models
    (run_dir / "model_registry.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    return len(domain_models)


def write_run_configuration(
    run_dir: Path,
    run_name: str,
    profile: str,
    n_splits: int,
    mode: str,
    registry: list[dict[str, Any]] | None = None,
    mlp_available: bool | None = None,
    targets: list[str] | None = None,
) -> None:
    profile_config = get_training_profile(profile)
    task_configs = []
    for model_key, target, selected_mode in iter_task_mode_targets(
        mode,
        targets=targets,
    ):
        config = get_experiment_config(model_key, target, selected_mode)
        task_config = {
            "model_key": model_key,
            "target": target,
            "mode": selected_mode,
            "dataset_path": Path(config["dataset_path"])
            .resolve()
            .relative_to(get_path().resolve())
            .as_posix(),
            "numeric_features": config["numeric_features"],
            "categorical_features": config["categorical_features"],
            "diagnostic_only": config["diagnostic_only"],
        }
        if registry:
            matching = next(
                (
                    entry
                    for entry in registry
                    if entry.get("route") == "ordinary_regression"
                    and entry.get("model_key") == model_key
                    and entry.get("target") == target
                    and entry.get("mode") == selected_mode
                ),
                None,
            )
            if matching:
                task_config.update(
                    {
                        key: matching[key]
                        for key in [
                            "dataset_sha256",
                            "eligible_rows",
                            "eligible_groups",
                            "final_test_groups",
                        ]
                    }
                )
        task_configs.append(task_config)
    payload = {
        "run_name": run_name,
        "profile": profile,
        "mode_selection": mode,
        "profile_parameters": profile_config,
        "mlp_runtime_available": mlp_available,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": 42,
        "test_fraction": 0.15,
        "cv": f"{n_splits}-fold GroupKFold",
        "grouping": (
            "dataset_id first for fatigue S-N curves; DOI first for static "
            "targets; source_id and record_id are fallbacks"
        ),
        "selection_metric": "grouped-CV OOF R2; RMSE and MAE tie-breakers",
        "fatigue_stress_amplitude_bounds_MPa": [1.0, 3000.0],
        "ordinary_fatigue_subset": (
            "ASTM E466-style conventional-frequency uncensored failures; "
            "diagnostic baseline only"
        ),
        "formal_fatigue_route": "protocol-aware monotonic XGBoost-AFT",
        "fatigue_protocol_frequency_Hz": {
            "e466_conventional_max": 200,
            "ultrasonic_vhcf_min": 1000,
        },
        "prediction_modes": selected_modes(mode),
        "selected_targets": targets or "all",
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
    targets: list[str] | None = None,
) -> Path:
    profile_config = get_training_profile(profile)
    mlp_available = (
        mlp_runtime_available()
        if bool(profile_config.get("include_mlp", False))
        else None
    )
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
    write_run_configuration(
        run_dir,
        run_name,
        profile,
        n_splits,
        mode,
        mlp_available=mlp_available,
        targets=targets,
    )
    all_rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    physical_checks: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []

    for model_key, target, selected_mode in iter_task_mode_targets(
        mode,
        targets=targets,
    ):
        print(
            f"\nTraining ordinary route: " f"{model_key} / {target} / {selected_mode}"
        )
        rows, entry, oof_frame = train_conventional_experiment(
            model_key,
            target,
            selected_mode,
            run_dir,
            n_splits,
            profile,
        )
        all_rows.extend(rows)
        registry.append(entry)
        oof_parts.append(oof_frame)

    include_fatigue = targets is None or "log10_fatigue_life_cycles" in targets
    for selected_mode in modes if include_fatigue else []:
        print(f"\nTraining Basquin routes: {selected_mode}")
        rows, entries, checks, basquin_oof = train_basquin_experiment(
            selected_mode,
            run_dir,
            n_splits,
            profile,
        )
        all_rows.extend(rows)
        registry.extend(entries)
        physical_checks.extend(checks)
        oof_parts.append(basquin_oof)

        print(f"\nTraining XGBoost-AFT route: {selected_mode}")
        rows, entry, aft_oof = train_aft_experiment(
            selected_mode,
            run_dir,
            n_splits,
            profile,
        )
        all_rows.extend(rows)
        registry.append(entry)
        oof_parts.append(aft_oof)

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
        )
        .astype(str)
        .eq("summary")
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
    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    oof_df.to_csv(
        run_dir / "tables" / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (run_dir / "model_registry.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    write_run_configuration(
        run_dir,
        run_name,
        profile,
        n_splits,
        mode,
        registry=registry,
        mlp_available=mlp_available,
        targets=targets,
    )
    return run_dir
