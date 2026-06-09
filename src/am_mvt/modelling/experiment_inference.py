from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from am_mvt.modelling.basquin import HierarchicalBasquin
from am_mvt.modelling.experiment_data import catboost_frame, clean_features
from am_mvt.modelling.experiment_training import predict_catboost


def domain_warnings(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    domain: dict[str, object],
) -> list[list[str]]:
    warnings: list[list[str]] = [[] for _ in range(len(frame))]
    numeric_ranges = dict(domain.get("numeric_ranges", {}))
    categorical_values = dict(domain.get("categorical_values", {}))

    for column in numeric_features + categorical_features:
        if column not in frame.columns:
            for row_warnings in warnings:
                row_warnings.append(f"missing_column:{column}")

    for column in numeric_features:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        bounds = dict(numeric_ranges.get(column, {}))
        minimum = bounds.get("min")
        maximum = bounds.get("max")

        for index, value in enumerate(values):
            if pd.isna(value):
                warnings[index].append(f"missing_value:{column}")
            elif minimum is not None and value < minimum:
                warnings[index].append(f"below_training_range:{column}")
            elif maximum is not None and value > maximum:
                warnings[index].append(f"above_training_range:{column}")

    for column in categorical_features:
        if column not in frame.columns:
            continue
        known = set(str(value) for value in categorical_values.get(column, []))

        for index, value in enumerate(frame[column].astype("string")):
            if pd.isna(value) or not str(value).strip():
                warnings[index].append(f"missing_value:{column}")
            elif known and str(value) not in known:
                warnings[index].append(f"unknown_category:{column}")

    return warnings


def ensure_feature_columns(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in numeric_features:
        if column not in result.columns:
            result[column] = np.nan

    for column in categorical_features:
        if column not in result.columns:
            result[column] = "missing"

    return result


def predict_ordinary(
    bundle: dict[str, object],
    frame: pd.DataFrame,
) -> np.ndarray:
    numeric = list(bundle["numeric_features"])
    categorical = list(bundle["categorical_features"])
    prepared = ensure_feature_columns(frame, numeric, categorical)
    kind = bundle["kind"]

    if kind == "catboost":
        return predict_catboost(
            bundle["model"],
            prepared,
            numeric,
            categorical,
            bundle["numeric_medians"],
        )

    if kind == "alloy_median":
        return bundle["model"].predict(prepared)

    clean = clean_features(prepared, numeric, categorical)
    return np.asarray(
        bundle["model"].predict(clean[numeric + categorical]),
        dtype=float,
    )


def prediction_rows(
    run_dir: Path,
    scenarios: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    registry = json.loads(
        (run_dir / "model_registry.json").read_text(encoding="utf-8")
    )
    rows = []

    for entry in registry:
        if mode != "all" and entry["mode"] != mode:
            continue

        route = entry["route"]
        predictions: np.ndarray
        lower = np.full(len(scenarios), np.nan)
        upper = np.full(len(scenarios), np.nan)
        warnings = [[] for _ in range(len(scenarios))]

        if route == "ordinary_regression":
            bundle = joblib.load(run_dir / entry["artifact"])
            predictions = predict_ordinary(bundle, scenarios)
            warnings = domain_warnings(
                scenarios,
                bundle["numeric_features"],
                bundle["categorical_features"],
                bundle["feature_domain"],
            )
            radius = float(bundle.get("conformal_q90", np.nan))
            lower = predictions - radius
            upper = predictions + radius

        elif route == "basquin_only":
            basquin = HierarchicalBasquin.load(run_dir / entry["artifact"])
            predictions = basquin.predict(scenarios)
            radius = float(entry.get("conformal_q90", np.nan))
            lower = predictions - radius
            upper = predictions + radius
            for index, value in enumerate(predictions):
                if not np.isfinite(value):
                    warnings[index].append("missing_or_invalid:stress_amplitude_MPa")

        elif route == "basquin_catboost_residual":
            bundle = joblib.load(run_dir / entry["artifact"])
            basquin = HierarchicalBasquin.load(run_dir / bundle["basquin_path"])
            baseline = basquin.predict(scenarios)
            prepared = ensure_feature_columns(
                scenarios,
                bundle["numeric_features"],
                bundle["categorical_features"],
            )
            correction = predict_catboost(
                bundle["model"],
                prepared,
                bundle["numeric_features"],
                bundle["categorical_features"],
                bundle["numeric_medians"],
            )
            predictions = baseline + correction
            warnings = domain_warnings(
                scenarios,
                bundle["numeric_features"],
                bundle["categorical_features"],
                bundle["feature_domain"],
            )
            radius = float(bundle.get("conformal_q90", np.nan))
            lower = predictions - radius
            upper = predictions + radius

        elif route == "xgboost_aft":
            metadata = joblib.load(run_dir / entry["preprocessor_artifact"])
            numeric = metadata["numeric_features"]
            categorical = metadata["categorical_features"]
            prepared = ensure_feature_columns(scenarios, numeric, categorical)
            clean = clean_features(prepared, numeric, categorical)
            transformed = metadata["preprocessor"].transform(
                clean[numeric + categorical]
            )
            booster = xgb.Booster()
            booster.load_model(run_dir / entry["artifact"])
            predictions = booster.predict(xgb.DMatrix(transformed))
            warnings = domain_warnings(
                scenarios,
                numeric,
                categorical,
                metadata["feature_domain"],
            )

        else:
            continue

        for index, prediction in enumerate(predictions):
            target = entry["target"]
            row = {
                "input_row": index,
                "model_key": entry["model_key"],
                "target": target,
                "mode": entry["mode"],
                "route": route,
                "candidate": entry["candidate"],
                "prediction": float(prediction)
                if np.isfinite(prediction)
                else np.nan,
                "prediction_lower_90": float(lower[index])
                if np.isfinite(lower[index])
                else np.nan,
                "prediction_upper_90": float(upper[index])
                if np.isfinite(upper[index])
                else np.nan,
                "warnings": ";".join(sorted(set(warnings[index]))),
                "interval_basis": (
                    "censor-aware point estimate; no conformal interval"
                    if route == "xgboost_aft"
                    else "90% OOF conformal interval on the target scale"
                ),
            }

            if target == "log10_fatigue_life_cycles":
                if route == "xgboost_aft":
                    row["predicted_fatigue_life_cycles"] = float(prediction)
                    row["prediction_log10_cycles"] = (
                        math.log10(max(float(prediction), 1.0))
                        if np.isfinite(prediction)
                        else np.nan
                    )
                else:
                    row["prediction_log10_cycles"] = row["prediction"]
                    row["predicted_fatigue_life_cycles"] = (
                        10 ** float(prediction)
                        if np.isfinite(prediction)
                        else np.nan
                    )
                    row["prediction_lower_90_cycles"] = (
                        10 ** float(lower[index])
                        if np.isfinite(lower[index])
                        else np.nan
                    )
                    row["prediction_upper_90_cycles"] = (
                        10 ** float(upper[index])
                        if np.isfinite(upper[index])
                        else np.nan
                    )

            rows.append(row)

    return pd.DataFrame(rows)


def predict_scenarios(
    run_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "all",
) -> Path:
    run_dir = Path(run_dir)
    input_path = Path(input_path)
    output_path = Path(output_path)

    if mode not in {"all", "process_only", "reduced_testing"}:
        raise ValueError("mode must be all, process_only, or reduced_testing")

    scenarios = pd.read_csv(input_path, low_memory=False)
    results = prediction_rows(run_dir, scenarios, mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
