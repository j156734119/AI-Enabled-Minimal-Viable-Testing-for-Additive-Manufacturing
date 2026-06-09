from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from am_mvt.utils.text import normalise_doi, normalise_identifier


def build_evaluation_groups(df: pd.DataFrame) -> pd.Series:
    doi = df.get("doi", pd.Series(pd.NA, index=df.index)).map(normalise_doi)
    dataset_id = df.get(
        "dataset_id",
        pd.Series(pd.NA, index=df.index),
    ).map(normalise_identifier)
    source_id = df.get(
        "source_id",
        pd.Series("unknown_source", index=df.index),
    ).map(normalise_identifier)
    record_id = df.get(
        "record_id",
        pd.Series([f"row_{i}" for i in range(len(df))], index=df.index),
    ).map(normalise_identifier)

    groups = pd.Series(index=df.index, dtype="string")
    groups.loc[doi.ne("")] = "doi:" + doi.loc[doi.ne("")]

    missing = groups.isna()
    groups.loc[missing & dataset_id.ne("")] = (
        "dataset:" + dataset_id.loc[missing & dataset_id.ne("")]
    )

    missing = groups.isna()
    groups.loc[missing & source_id.ne("")] = (
        "source:" + source_id.loc[missing & source_id.ne("")]
    )
    groups = groups.fillna("record:" + record_id)

    return groups


def load_experiment_frame(
    dataset_path: str | Path,
    target: str,
    target_bounds: tuple[float, float] | None,
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, low_memory=False)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()

    if target_bounds is not None:
        lower, upper = target_bounds
        df = df.loc[df[target].between(lower, upper, inclusive="both")].copy()

    df["evaluation_group_id"] = build_evaluation_groups(df)
    return df.reset_index(drop=True)


def select_usable_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    min_non_missing: int | None = None,
) -> tuple[list[str], list[str]]:
    numeric = []
    categorical = []
    required_non_missing = (
        min_non_missing
        if min_non_missing is not None
        else max(20, math.ceil(len(df) * 0.01))
    )

    for column in numeric_features:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if (
            values.notna().sum() >= required_non_missing
            and values.nunique(dropna=True) > 1
        ):
            numeric.append(column)

    for column in categorical_features:
        if column not in df.columns:
            continue
        values = df[column].astype("string").str.strip().replace("", pd.NA)
        if (
            values.notna().sum() >= required_non_missing
            and values.nunique(dropna=True) > 1
        ):
            categorical.append(column)

    return numeric, categorical


def clean_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    result = df.copy()

    for column in numeric_features:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in categorical_features:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .replace("", "missing")
            .fillna("missing")
        )

    return result


def make_one_hot_encoder(sparse: bool = True) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    sparse: bool = True,
) -> ColumnTransformer:
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="missing",
                            ),
                        ),
                        ("encoder", make_one_hot_encoder(sparse=sparse)),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns found.")

    return ColumnTransformer(transformers=transformers)


def split_development_and_test(
    df: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
    test_groups: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_groups is not None:
        in_test = df["evaluation_group_id"].astype(str).isin(test_groups)
        if not in_test.any() or in_test.all():
            raise ValueError("Supplied final holdout groups produce an empty split.")
        return (
            df.loc[~in_test].reset_index(drop=True),
            df.loc[in_test].reset_index(drop=True),
        )

    groups = df["evaluation_group_id"]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    development_index, test_index = next(splitter.split(df, groups=groups))

    return (
        df.iloc[development_index].reset_index(drop=True),
        df.iloc[test_index].reset_index(drop=True),
    )


def select_final_holdout_groups(df: pd.DataFrame) -> set[str]:
    _, test_df = split_development_and_test(df)
    return set(test_df["evaluation_group_id"].astype(str))


def make_group_folds(
    development_df: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = development_df["evaluation_group_id"]
    unique_groups = groups.nunique(dropna=True)

    if unique_groups < n_splits:
        raise ValueError(
            f"Need at least {n_splits} groups for GroupKFold; found {unique_groups}."
        )

    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(splitter.split(development_df, groups=groups))


def assert_disjoint_groups(*frames: pd.DataFrame) -> None:
    group_sets = [
        set(frame["evaluation_group_id"].dropna().astype(str))
        for frame in frames
    ]

    for left_index in range(len(group_sets)):
        for right_index in range(left_index + 1, len(group_sets)):
            overlap = group_sets[left_index] & group_sets[right_index]
            if overlap:
                raise AssertionError(
                    f"Evaluation group leakage detected: {sorted(overlap)[:3]}"
                )


def catboost_frame(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = clean_features(df, numeric_features, categorical_features)

    if numeric_medians is None:
        numeric_medians = {}
        for column in numeric_features:
            median = pd.to_numeric(result[column], errors="coerce").median()
            numeric_medians[column] = 0.0 if pd.isna(median) else float(median)

    for column in numeric_features:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(
            numeric_medians[column]
        )

    return result[numeric_features + categorical_features], numeric_medians
