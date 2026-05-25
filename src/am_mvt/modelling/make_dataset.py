from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from am_mvt.config import get_path


MODEL_CONFIGS = {
    "model1_static": {
        "dataset_path": get_path("data", "processed", "view_model1_static.csv"),
        "targets": [
            "uts_MPa",
            "yield_strength_MPa",
            "elongation_percent",
        ],
        "numeric_features": [
            "laser_power_W",
            "scan_speed_mm_s",
            "hatch_spacing_um",
            "layer_thickness_um",
            "ved_J_mm3",
            "porosity_percent",
            "relative_density_percent",
        ],
        "categorical_features": [
            "alloy",
            "alloy_family",
            "am_process",
            "machine_model",
            "build_orientation",
            "test_direction",
            "scan_strategy",
            "heat_treatment",
            "surface_condition",
            "post_processing",
            "density_measurement_method",
            "defect_type",
        ],
        "group_column": "dataset_id",
        "weight_column": "sample_weight",
        "minimum_rows": 20,
    },
    "model2_sn_fatigue": {
        "dataset_path": get_path("data", "processed", "view_model2_sn_fatigue.csv"),
        "targets": [
            "log10_fatigue_life_cycles",
        ],
        "numeric_features": [
            "laser_power_W",
            "scan_speed_mm_s",
            "hatch_spacing_um",
            "layer_thickness_um",
            "ved_J_mm3",
            "porosity_percent",
            "relative_density_percent",
            "yield_strength_MPa",
            "uts_MPa",
            "elongation_percent",
            "stress_amplitude_MPa",
            "max_stress_MPa",
            "r_ratio",
            "frequency_Hz",
            "test_temperature_C",
        ],
        "categorical_features": [
            "alloy",
            "alloy_family",
            "am_process",
            "machine_model",
            "build_orientation",
            "test_direction",
            "scan_strategy",
            "heat_treatment",
            "surface_condition",
            "post_processing",
            "density_measurement_method",
            "defect_type",
        ],
        "group_column": "dataset_id",
        "weight_column": "sample_weight",
        "minimum_rows": 30,
    },
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    return pd.read_csv(path, low_memory=False)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_usable_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str]]:
    usable_numeric = []
    usable_categorical = []

    for col in numeric_features:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().sum() > 0 and values.nunique(dropna=True) > 1:
            usable_numeric.append(col)

    for col in categorical_features:
        if col not in df.columns:
            continue

        values = df[col].astype("string").str.strip()
        values = values.replace("", pd.NA)

        if values.notna().sum() > 0 and values.nunique(dropna=True) > 1:
            usable_categorical.append(col)

    return usable_numeric, usable_categorical


def clean_training_frame(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    result = df.copy()

    result[target] = pd.to_numeric(result[target], errors="coerce")
    result = result.dropna(subset=[target]).copy()

    for col in numeric_features:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    for col in categorical_features:
        result[col] = (
            result[col]
            .astype("string")
            .str.strip()
            .replace("", "missing")
            .fillna("missing")
        )

    return result


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        transformers.append(("numeric", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", make_one_hot_encoder()),
            ]
        )

        transformers.append(("categorical", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No usable feature columns found.")

    return ColumnTransformer(transformers=transformers)


def split_by_group_if_possible(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    sample_weight: pd.Series | None,
    test_size: float,
    random_state: int,
):
    if groups is not None:
        groups = groups.astype("string").fillna("missing_group")

    can_group_split = groups is not None and groups.nunique(dropna=True) >= 2

    if can_group_split:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        if sample_weight is not None:
            w_train = sample_weight.iloc[train_idx].copy()
            w_test = sample_weight.iloc[test_idx].copy()
        else:
            w_train = None
            w_test = None

        split_method = "GroupShuffleSplit_by_dataset_id"

    else:
        if sample_weight is not None:
            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X,
                y,
                sample_weight,
                test_size=test_size,
                random_state=random_state,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
            w_train = None
            w_test = None

        split_method = "random_train_test_split"

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "w_train": w_train,
        "w_test": w_test,
        "split_method": split_method,
    }


def prepare_regression_data(
    model_key: str,
    target: str,
    dataset_path: str | Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}")

    config = MODEL_CONFIGS[model_key]

    if target not in config["targets"]:
        raise ValueError(
            f"Target {target} is not configured for {model_key}. "
            f"Allowed targets: {config['targets']}"
        )

    if dataset_path is None:
        dataset_path = config["dataset_path"]

    df = load_dataset(dataset_path)

    if target not in df.columns:
        raise ValueError(f"Target column not found in dataset: {target}")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()

    minimum_rows = int(config.get("minimum_rows", 20))

    if len(df) < minimum_rows:
        raise ValueError(
            f"Not enough usable rows for {model_key} / {target}. "
            f"Rows available: {len(df)}, minimum required: {minimum_rows}."
        )

    numeric_features, categorical_features = get_usable_features(
        df,
        numeric_features=config["numeric_features"],
        categorical_features=config["categorical_features"],
    )

    working_df = clean_training_frame(
        df,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    feature_columns = numeric_features + categorical_features

    if not feature_columns:
        raise ValueError(f"No usable features for {model_key} / {target}.")

    X = working_df[feature_columns].copy()
    y = working_df[target].copy()

    group_column = config.get("group_column")
    groups = working_df[group_column].copy() if group_column in working_df.columns else None

    weight_column = config.get("weight_column")

    if weight_column in working_df.columns:
        sample_weight = pd.to_numeric(working_df[weight_column], errors="coerce")
        sample_weight = sample_weight.fillna(1.0)
    else:
        sample_weight = None

    split = split_by_group_if_possible(
        X=X,
        y=y,
        groups=groups,
        sample_weight=sample_weight,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    return {
        **split,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_key": model_key,
        "target": target,
        "n_rows": len(working_df),
        "n_groups": groups.nunique(dropna=True) if groups is not None else 0,
    }