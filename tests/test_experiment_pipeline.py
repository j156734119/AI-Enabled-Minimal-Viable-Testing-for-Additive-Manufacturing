from __future__ import annotations

import json
import inspect

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from am_mvt.modelling.basquin import HierarchicalBasquin
from am_mvt.modelling.build_views import build_model_views
from am_mvt.modelling.experiment_config import (
    MEASURED_PROPERTY_FEATURES,
    get_experiment_config,
    get_training_profile,
    iter_task_mode_targets,
)
from am_mvt.modelling.experiment_data import (
    assert_disjoint_groups,
    build_evaluation_groups,
    make_group_folds,
    select_final_holdout_groups,
    split_development_and_test,
)
from am_mvt.modelling.experiment_inference import (
    domain_warnings,
    predict_scenarios,
)
from am_mvt.modelling.experiment_training import (
    filter_valid_fatigue_loading,
    get_model_candidates,
    make_aft_bounds,
    run_experiment_suite,
    write_run_configuration,
)


def test_process_only_does_not_use_measured_properties():
    for model_key, target in [
        ("model1_uts", "uts_MPa"),
        ("model2_sn_fatigue", "log10_fatigue_life_cycles"),
        ("model3_elongation_yield", "elongation_percent"),
        ("model3_elongation_yield", "yield_strength_MPa"),
        ("model4_elastic_modulus", "youngs_modulus_GPa"),
    ]:
        config = get_experiment_config(model_key, target, "process_only")
        assert not set(config["numeric_features"]) & set(
            MEASURED_PROPERTY_FEATURES
        )


def test_master_dataset_builds_all_four_modelling_views(tmp_path):
    master_path = tmp_path / "master.csv"
    pd.DataFrame(
        [
            {
                "source_id": "paper_static",
                "dataset_id": "static_1",
                "record_id": "static_row",
                "task_type": "static_tensile",
                "alloy": "Ti-6Al-4V",
                "uts_MPa": 1000.0,
                "yield_strength_MPa": 900.0,
                "elongation_percent": 10.0,
                "youngs_modulus_GPa": 110.0,
            },
            {
                "source_id": "paper_fatigue",
                "dataset_id": "fatigue_1",
                "record_id": "fatigue_row",
                "task_type": "sn_fatigue",
                "source_sheet": "S-N",
                "alloy": "Ti-6Al-4V",
                "stress_amplitude_MPa": 300.0,
                "fatigue_life_cycles": 100000.0,
                "runout": False,
            },
        ]
    ).to_csv(master_path, index=False)

    views = build_model_views(master_path)

    assert len(views["model1_uts"]) == 1
    assert len(views["model2_sn_fatigue"]) == 1
    assert len(views["model3_elongation_yield"]) == 1
    assert len(views["model4_elastic_modulus"]) == 1
    assert views["model2_sn_fatigue"].iloc[0][
        "log10_fatigue_life_cycles"
    ] == pytest.approx(5.0)
    for view in views.values():
        assert "failure_mode" not in view.columns
        assert "fracture_origin" not in view.columns


def test_fast_profile_uses_one_light_catboost_candidate():
    candidates = get_model_candidates("fast")
    catboost_names = [
        name
        for name, candidate in candidates.items()
        if candidate["kind"] == "catboost"
    ]

    assert list(candidates) == [
        "dummy_mean",
        "dummy_median",
        "alloy_family_median",
        "ridge",
        "random_forest",
        "xgboost",
        "catboost_light",
    ]
    assert catboost_names == ["catboost_light"]
    assert candidates["random_forest"]["model"].n_estimators == 120
    assert candidates["xgboost"]["model"].n_estimators == 200
    assert candidates["catboost_light"]["params"]["iterations"] == 300
    assert candidates["catboost_light"]["early_stopping_rounds"] == 30


def test_standard_profile_retains_all_catboost_candidates():
    candidates = get_model_candidates("standard")
    catboost_names = {
        name
        for name, candidate in candidates.items()
        if candidate["kind"] == "catboost"
    }
    assert catboost_names == {
        "catboost_d4",
        "catboost_d6",
        "catboost_d8",
        "catboost_robust",
    }


def test_fast_defaults_to_process_only_and_three_folds():
    signature = inspect.signature(run_experiment_suite)
    assert signature.parameters["profile"].default == "fast"
    assert signature.parameters["mode"].default == "process_only"
    assert signature.parameters["n_splits"].default is None
    assert get_training_profile("fast")["default_cv_folds"] == 3
    assert get_training_profile("standard")["default_cv_folds"] == 5
    assert len(list(iter_task_mode_targets("process_only"))) == 5
    assert len(list(iter_task_mode_targets("all"))) == 10


def test_run_configuration_records_fast_profile_and_mode(tmp_path):
    write_run_configuration(
        tmp_path,
        run_name="cpu_fast_test",
        profile="fast",
        n_splits=3,
        mode="process_only",
    )
    payload = json.loads((tmp_path / "run_config.json").read_text())

    assert payload["profile"] == "fast"
    assert payload["mode_selection"] == "process_only"
    assert payload["prediction_modes"] == ["process_only"]
    assert payload["profile_parameters"]["basquin_residual_catboost"] == (
        "catboost_light"
    )
    assert payload["profile_parameters"]["aft_boost_rounds"] == 400
    assert len(payload["task_configs"]) == 5


def test_doi_has_priority_for_evaluation_groups():
    frame = pd.DataFrame(
        {
            "doi": [
                "10.1/example",
                "https://doi.org/10.1/example",
                None,
            ],
            "source_id": ["a", "b", "c"],
            "dataset_id": ["1", "2", "3"],
            "record_id": ["r1", "r2", "r3"],
        }
    )
    groups = build_evaluation_groups(frame)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[2] == "dataset:3"


def test_fatigue_reduced_testing_uses_only_planned_measured_properties():
    config = get_experiment_config(
        "model2_sn_fatigue",
        "log10_fatigue_life_cycles",
        "reduced_testing",
    )
    measured = set(config["numeric_features"]) & set(MEASURED_PROPERTY_FEATURES)
    assert measured == {"uts_MPa", "yield_strength_MPa", "elongation_percent"}


def test_holdout_groups_are_disjoint():
    frame = pd.DataFrame(
        {
            "evaluation_group_id": np.repeat(
                [f"group_{index}" for index in range(20)],
                2,
            )
        }
    )
    development, test = split_development_and_test(frame)
    assert_disjoint_groups(development, test)


def test_group_kfold_never_splits_an_evaluation_group():
    frame = pd.DataFrame(
        {
            "evaluation_group_id": np.repeat(
                [f"group_{index}" for index in range(10)],
                3,
            )
        }
    )
    for train_index, validation_index in make_group_folds(frame, n_splits=5):
        assert_disjoint_groups(
            frame.iloc[train_index],
            frame.iloc[validation_index],
        )


def test_route_subsets_reuse_the_same_final_holdout_groups():
    frame = pd.DataFrame(
        {
            "evaluation_group_id": np.repeat(
                [f"group_{index}" for index in range(20)],
                2,
            ),
            "runout": [False, True] * 20,
        }
    )
    holdout_groups = select_final_holdout_groups(frame)
    failures = frame.loc[frame["runout"].eq(False)].reset_index(drop=True)
    development, test = split_development_and_test(
        failures,
        test_groups=holdout_groups,
    )

    assert set(test["evaluation_group_id"]) == holdout_groups
    assert_disjoint_groups(development, test)


def test_runout_becomes_right_censored():
    frame = pd.DataFrame(
        {
            "fatigue_life_cycles": [1000.0, 2000.0],
            "runout": [False, True],
        }
    )
    lower, upper, event = make_aft_bounds(frame)
    assert lower.tolist() == [1000.0, 2000.0]
    assert upper[0] == 1000.0
    assert np.isinf(upper[1])
    assert event.tolist() == [True, False]


def test_invalid_or_unit_contaminated_stress_is_excluded():
    frame = pd.DataFrame(
        {"stress_amplitude_MPa": [np.nan, 0.003, 100.0, 3500.0]}
    )
    filtered = filter_valid_fatigue_loading(frame)
    assert filtered["stress_amplitude_MPa"].tolist() == [100.0]


def test_basquin_slope_is_negative_and_prediction_is_monotonic():
    stress = np.array([100, 150, 200, 250, 300, 350] * 20, dtype=float)
    life = 8.5 - 2.0 * np.log10(stress)
    frame = pd.DataFrame(
        {
            "stress_amplitude_MPa": stress,
            "log10_fatigue_life_cycles": life,
            "runout": False,
            "alloy_family": "Ti alloy",
            "r_ratio": 0.1,
        }
    )
    model = HierarchicalBasquin(
        family_r_min_rows=10,
        family_min_rows=10,
        global_min_rows=10,
    ).fit(frame)
    scan = pd.DataFrame(
        {
            "stress_amplitude_MPa": [100, 200, 300],
            "alloy_family": ["Ti alloy"] * 3,
            "r_ratio": [0.1] * 3,
        }
    )
    predictions = model.predict(scan)
    assert model.global_curve is not None
    assert model.global_curve.slope < 0
    assert np.all(np.diff(predictions) <= 0)


def test_prediction_warnings_cover_missing_unknown_and_range():
    frame = pd.DataFrame(
        {
            "laser_power_W": [500],
            "alloy": ["unseen_alloy"],
        }
    )
    warnings = domain_warnings(
        frame,
        numeric_features=["laser_power_W", "scan_speed_mm_s"],
        categorical_features=["alloy"],
        domain={
            "numeric_ranges": {
                "laser_power_W": {"min": 100, "max": 400},
                "scan_speed_mm_s": {"min": 200, "max": 2000},
            },
            "categorical_values": {"alloy": ["Ti-6Al-4V"]},
        },
    )[0]
    assert "above_training_range:laser_power_W" in warnings
    assert "missing_column:scan_speed_mm_s" in warnings
    assert "unknown_category:alloy" in warnings


def test_batch_prediction_reports_missing_and_out_of_range_inputs(tmp_path):
    run_dir = tmp_path / "run"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    model = DummyRegressor(strategy="mean").fit(
        pd.DataFrame({"laser_power_W": [100.0, 200.0]}),
        [400.0, 600.0],
    )
    bundle = {
        "kind": "sklearn",
        "model": model,
        "numeric_features": ["laser_power_W"],
        "categorical_features": [],
        "conformal_q90": 50.0,
        "feature_domain": {
            "numeric_ranges": {
                "laser_power_W": {"min": 100.0, "max": 200.0}
            },
            "categorical_values": {},
        },
    }
    artifact = model_dir / "model.joblib"
    joblib.dump(bundle, artifact)
    (run_dir / "model_registry.json").write_text(
        json.dumps(
            [
                {
                    "model_key": "model1_uts",
                    "target": "uts_MPa",
                    "mode": "process_only",
                    "route": "ordinary_regression",
                    "candidate": "dummy_mean",
                    "artifact": "models/model.joblib",
                }
            ]
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "scenarios.csv"
    output_path = tmp_path / "predictions.csv"
    pd.DataFrame({"laser_power_W": [300.0]}).to_csv(input_path, index=False)

    predict_scenarios(run_dir, input_path, output_path, mode="process_only")
    result = pd.read_csv(output_path)

    assert result.loc[0, "prediction"] == 500.0
    assert result.loc[0, "prediction_lower_90"] == 450.0
    assert "above_training_range:laser_power_W" in result.loc[0, "warnings"]
