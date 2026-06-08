from __future__ import annotations

from copy import deepcopy

from am_mvt.config import get_path


PROCESS_NUMERIC_FEATURES = [
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "porosity_percent",
    "relative_density_percent",
]

PROCESS_CATEGORICAL_FEATURES = [
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
]

FATIGUE_LOADING_FEATURES = [
    "stress_amplitude_MPa",
    "max_stress_MPa",
    "r_ratio",
    "frequency_Hz",
    "test_temperature_C",
]

MEASURED_PROPERTY_FEATURES = [
    "yield_strength_MPa",
    "uts_MPa",
    "elongation_percent",
    "youngs_modulus_GPa",
    "hardness_HV",
]


BASE_TASKS = {
    "model1_uts": {
        "dataset_path": get_path("data", "processed", "view_model1_uts.csv"),
        "targets": ["uts_MPa"],
        "process_numeric": PROCESS_NUMERIC_FEATURES,
        "process_categorical": PROCESS_CATEGORICAL_FEATURES,
        "minimum_rows": 20,
        "target_bounds": {"uts_MPa": (20.0, 4000.0)},
    },
    "model2_sn_fatigue": {
        "dataset_path": get_path(
            "data",
            "processed",
            "view_model2_sn_fatigue.csv",
        ),
        "targets": ["log10_fatigue_life_cycles"],
        "process_numeric": PROCESS_NUMERIC_FEATURES + FATIGUE_LOADING_FEATURES,
        "process_categorical": PROCESS_CATEGORICAL_FEATURES,
        "minimum_rows": 30,
        "target_bounds": {"log10_fatigue_life_cycles": (0.0, 12.0)},
    },
    "model3_elongation_yield": {
        "dataset_path": get_path(
            "data",
            "processed",
            "view_model3_elongation_yield.csv",
        ),
        "targets": ["elongation_percent", "yield_strength_MPa"],
        "process_numeric": PROCESS_NUMERIC_FEATURES,
        "process_categorical": PROCESS_CATEGORICAL_FEATURES,
        "minimum_rows": 20,
        "target_bounds": {
            "elongation_percent": (0.0, 150.0),
            "yield_strength_MPa": (10.0, 3500.0),
        },
    },
    "model4_elastic_modulus": {
        "dataset_path": get_path(
            "data",
            "processed",
            "view_model4_elastic_modulus.csv",
        ),
        "targets": ["youngs_modulus_GPa"],
        "process_numeric": PROCESS_NUMERIC_FEATURES,
        "process_categorical": PROCESS_CATEGORICAL_FEATURES,
        "minimum_rows": 15,
        "target_bounds": {"youngs_modulus_GPa": (1.0, 500.0)},
    },
}


CATBOOST_CANDIDATES = {
    "catboost_d4": {
        "iterations": 600,
        "depth": 4,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
    },
    "catboost_d6": {
        "iterations": 800,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 7.0,
    },
    "catboost_d8": {
        "iterations": 700,
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 10.0,
    },
    "catboost_robust": {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.025,
        "l2_leaf_reg": 15.0,
        "loss_function": "Huber:delta=1.0",
    },
}

FAST_CATBOOST_CANDIDATES = {
    "catboost_light": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 7.0,
    }
}

TRAINING_PROFILES = {
    "fast": {
        "default_cv_folds": 3,
        "random_forest_estimators": 120,
        "xgboost_estimators": 200,
        "catboost_candidates": FAST_CATBOOST_CANDIDATES,
        "catboost_early_stopping_rounds": 30,
        "basquin_residual_catboost": "catboost_light",
        "aft_boost_rounds": 400,
        "aft_early_stopping_rounds": 30,
    },
    "standard": {
        "default_cv_folds": 5,
        "random_forest_estimators": 200,
        "xgboost_estimators": 300,
        "catboost_candidates": CATBOOST_CANDIDATES,
        "catboost_early_stopping_rounds": 50,
        "basquin_residual_catboost": "catboost_d6",
        "aft_boost_rounds": 1200,
        "aft_early_stopping_rounds": 50,
    },
}


def get_training_profile(profile: str) -> dict[str, object]:
    if profile not in TRAINING_PROFILES:
        raise ValueError(f"Unknown training profile: {profile}")
    return deepcopy(TRAINING_PROFILES[profile])


def get_experiment_config(
    model_key: str,
    target: str,
    mode: str,
) -> dict[str, object]:
    if model_key not in BASE_TASKS:
        raise ValueError(f"Unknown model key: {model_key}")

    if mode not in {"process_only", "reduced_testing"}:
        raise ValueError(f"Unknown prediction mode: {mode}")

    config = deepcopy(BASE_TASKS[model_key])

    if target not in config["targets"]:
        raise ValueError(f"Target {target} is not configured for {model_key}")

    numeric_features = list(config["process_numeric"])
    categorical_features = list(config["process_categorical"])

    if mode == "reduced_testing":
        assisted_features = (
            ["uts_MPa", "yield_strength_MPa", "elongation_percent"]
            if model_key == "model2_sn_fatigue"
            else MEASURED_PROPERTY_FEATURES
        )
        numeric_features.extend(
            feature for feature in assisted_features if feature != target
        )

    config["numeric_features"] = list(dict.fromkeys(numeric_features))
    config["categorical_features"] = categorical_features
    config["mode"] = mode
    config["diagnostic_only"] = mode == "reduced_testing" and model_key != (
        "model2_sn_fatigue"
    )

    return config


def selected_modes(mode: str) -> list[str]:
    if mode == "all":
        return ["process_only", "reduced_testing"]
    if mode in {"process_only", "reduced_testing"}:
        return [mode]
    raise ValueError(f"Unknown prediction mode selection: {mode}")


def iter_task_mode_targets(mode: str = "all"):
    for model_key, config in BASE_TASKS.items():
        for target in config["targets"]:
            for selected_mode in selected_modes(mode):
                yield model_key, target, selected_mode
