from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from am_mvt.modelling.model_explanation import (
    grouped_error_rows,
    permutation_importance_rows,
    sensitivity_rows,
)
from am_mvt.optimisation.testing_matrix import (
    MATRIX_COLUMNS,
    build_reduced_testing_matrix,
)


def make_bundle_and_frame():
    rng = np.random.default_rng(42)
    signal = np.linspace(0, 10, 120)
    noise = rng.normal(size=120)
    target = 4 * signal + rng.normal(scale=0.2, size=120)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "noise": noise,
            "alloy_family": ["Ti alloy"] * 60 + ["Al alloy"] * 60,
            "target": target,
        }
    )
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
    ).fit(frame[["signal", "noise"]], target)
    bundle = {
        "kind": "sklearn",
        "model": model,
        "model_key": "test_model",
        "target": "target",
        "mode": "process_only",
        "candidate": "random_forest",
        "numeric_features": ["signal", "noise"],
        "categorical_features": [],
    }
    return bundle, frame


def test_permutation_importance_ranks_signal_above_noise():
    bundle, frame = make_bundle_and_frame()
    result = permutation_importance_rows(
        bundle,
        frame,
        repeats=3,
    )
    importance = result.set_index("feature")[
        "permutation_mae_increase_mean"
    ]
    assert importance["signal"] > importance["noise"]
    assert result["importance_fraction"].sum() == pytest.approx(1.0)


def test_grouped_errors_require_minimum_rows():
    bundle, frame = make_bundle_and_frame()
    frame.loc[0:2, "surface_condition"] = "rare"
    result = grouped_error_rows(bundle, frame, min_group_rows=5)
    assert "rare" not in set(result["group_value"])
    assert {"Ti alloy", "Al alloy"} <= set(result["group_value"])


def test_numeric_sensitivity_reports_direction():
    bundle, frame = make_bundle_and_frame()
    frame["ved_J_mm3"] = frame["signal"]
    bundle["numeric_features"].append("ved_J_mm3")
    bundle["model"] = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
    ).fit(frame[bundle["numeric_features"]], frame["target"])
    result = sensitivity_rows(bundle, frame)
    ved = result.loc[result["feature"].eq("ved_J_mm3")]
    assert len(ved) == 5
    assert set(ved["observed_direction"]) == {"increasing"}


def test_testing_matrix_keeps_sparse_risk_validation():
    summary = pd.DataFrame(
        [
            {
                "target": target,
                "route": "ordinary_regression",
                "selected": True,
                "candidate": "test_model",
                "test_r2": 0.6,
                "test_mae": 1.0,
            }
            for target in [
                "uts_MPa",
                "yield_strength_MPa",
                "elongation_percent",
                "youngs_modulus_GPa",
                "log10_fatigue_life_cycles",
            ]
        ]
    )
    relationships = pd.DataFrame(
        [
            {
                "relationship_id": relationship_id,
                "evidence": "test evidence",
            }
            for relationship_id in [
                "stress_to_fatigue",
                "orientation_to_properties",
                "surface_to_fatigue",
                "defect_to_fatigue",
                "uts_to_yield",
                "strength_to_elongation",
                "heat_treatment_to_properties",
            ]
        ]
    )
    matrix = build_reduced_testing_matrix(summary, relationships)
    assert list(matrix.columns) == MATRIX_COLUMNS
    sparse = matrix.loc[
        matrix["supporting_features"].str.contains(
            "defect_type|surface_condition|heat_treatment",
            regex=True,
        )
    ]
    assert sparse["needs_validation"].all()
    assert sparse["coverage_risk"].isin(["high", "very_high"]).all()
