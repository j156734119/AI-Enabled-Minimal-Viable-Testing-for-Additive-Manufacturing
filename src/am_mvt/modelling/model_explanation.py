from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from am_mvt.modelling.experiment_config import get_experiment_config
from am_mvt.modelling.experiment_data import (
    load_experiment_frame,
    select_final_holdout_groups,
    split_development_and_test,
)
from am_mvt.modelling.experiment_inference import predict_ordinary
from am_mvt.modelling.experiment_metrics import regression_metrics
from am_mvt.modelling.experiment_training import (
    filter_valid_fatigue_loading,
    normalise_runout,
)


GROUP_COLUMNS = [
    "alloy_family",
    "am_process",
    "build_orientation",
    "surface_condition",
]

SENSITIVITY_FEATURES = [
    "ved_J_mm3",
    "porosity_percent",
    "relative_density_percent",
    "stress_amplitude_MPa",
    "build_orientation",
]


def load_registry(run_dir: Path) -> list[dict[str, Any]]:
    return json.loads(
        (run_dir / "model_registry.json").read_text(encoding="utf-8")
    )


def ordinary_entries(run_dir: Path, mode: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in load_registry(run_dir)
        if entry["route"] == "ordinary_regression" and entry["mode"] == mode
    ]


def evaluation_frames(
    model_key: str,
    target: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = get_experiment_config(model_key, target, mode)
    frame = load_experiment_frame(
        config["dataset_path"],
        target,
        config["target_bounds"].get(target),
    )
    final_holdout_groups = None

    if model_key == "model2_sn_fatigue":
        frame = filter_valid_fatigue_loading(frame)
        final_holdout_groups = select_final_holdout_groups(frame)
        runout = normalise_runout(frame["runout"])
        frame = frame.loc[runout.eq(False)].reset_index(drop=True)

    _, test_df = split_development_and_test(
        frame,
        test_groups=final_holdout_groups,
    )
    return frame, test_df


def permutation_importance_rows(
    bundle: dict[str, Any],
    test_df: pd.DataFrame,
    *,
    repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    target = str(bundle["target"])
    features = list(bundle["numeric_features"]) + list(
        bundle["categorical_features"]
    )
    baseline_predictions = predict_ordinary(bundle, test_df)
    baseline_mae = regression_metrics(
        test_df[target],
        baseline_predictions,
    )["mae"]
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []

    for feature in features:
        if feature not in test_df.columns:
            continue

        repeat_increases = []
        values = test_df[feature].to_numpy(copy=True)

        for _ in range(repeats):
            permuted = test_df.copy()
            permuted[feature] = values[rng.permutation(len(values))]
            predictions = predict_ordinary(bundle, permuted)
            permuted_mae = regression_metrics(
                permuted[target],
                predictions,
            )["mae"]
            repeat_increases.append(permuted_mae - baseline_mae)

        rows.append(
            {
                "model_key": bundle["model_key"],
                "target": target,
                "mode": bundle["mode"],
                "candidate": bundle["candidate"],
                "feature": feature,
                "feature_type": (
                    "numeric"
                    if feature in bundle["numeric_features"]
                    else "categorical"
                ),
                "test_rows": len(test_df),
                "non_missing_count": int(test_df[feature].notna().sum()),
                "coverage_fraction": float(test_df[feature].notna().mean()),
                "baseline_mae": baseline_mae,
                "permutation_mae_increase_mean": float(
                    np.mean(repeat_increases)
                ),
                "permutation_mae_increase_std": float(
                    np.std(repeat_increases, ddof=0)
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    positive = result["permutation_mae_increase_mean"].clip(lower=0)
    total = float(positive.sum())
    result["importance_fraction"] = positive / total if total else 0.0
    return result.sort_values(
        "permutation_mae_increase_mean",
        ascending=False,
    ).reset_index(drop=True)


def grouped_error_rows(
    bundle: dict[str, Any],
    test_df: pd.DataFrame,
    *,
    min_group_rows: int = 5,
) -> pd.DataFrame:
    target = str(bundle["target"])
    predictions = predict_ordinary(bundle, test_df)
    evaluated = test_df.copy()
    evaluated["_prediction"] = predictions
    rows: list[dict[str, Any]] = []

    for group_column in GROUP_COLUMNS:
        if group_column not in evaluated.columns:
            continue

        for group_value, group in evaluated.groupby(group_column, dropna=True):
            if len(group) < min_group_rows:
                continue

            metrics = regression_metrics(group[target], group["_prediction"])
            rows.append(
                {
                    "model_key": bundle["model_key"],
                    "target": target,
                    "mode": bundle["mode"],
                    "group_column": group_column,
                    "group_value": group_value,
                    "rows": len(group),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def coverage_rows(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    features = list(bundle["numeric_features"]) + list(
        bundle["categorical_features"]
    )
    rows = []

    for feature in features:
        if feature not in frame.columns:
            continue

        non_missing = frame[feature].notna()
        rows.append(
            {
                "model_key": bundle["model_key"],
                "target": bundle["target"],
                "mode": bundle["mode"],
                "feature": feature,
                "feature_type": (
                    "numeric"
                    if feature in bundle["numeric_features"]
                    else "categorical"
                ),
                "rows": len(frame),
                "non_missing_count": int(non_missing.sum()),
                "coverage_fraction": float(non_missing.mean()),
                "unique_values": int(frame[feature].nunique(dropna=True)),
                "source_count": int(
                    frame.loc[non_missing, "source_id"].nunique(dropna=True)
                )
                if "source_id" in frame
                else 0,
                "group_count": int(
                    frame.loc[
                        non_missing,
                        "evaluation_group_id",
                    ].nunique(dropna=True)
                ),
            }
        )

    return pd.DataFrame(rows)


def sensitivity_rows(
    bundle: dict[str, Any],
    test_df: pd.DataFrame,
    *,
    max_rows: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    sample = test_df.sample(
        n=min(len(test_df), max_rows),
        random_state=random_state,
    ).reset_index(drop=True)
    baseline = float(np.mean(predict_ordinary(bundle, sample)))
    rows: list[dict[str, Any]] = []

    for feature in SENSITIVITY_FEATURES:
        if feature not in sample.columns:
            continue

        non_missing = sample[feature].dropna()

        if feature in bundle["numeric_features"]:
            numeric = pd.to_numeric(non_missing, errors="coerce").dropna()

            if len(numeric) < 20 or numeric.nunique() < 3:
                continue

            values = np.unique(
                np.quantile(numeric, [0.1, 0.25, 0.5, 0.75, 0.9])
            )
            sensitivity_type = "numeric_quantile"
        elif feature in bundle["categorical_features"]:
            counts = non_missing.astype(str).value_counts()
            values = counts.loc[counts >= 5].head(6).index.to_numpy()
            sensitivity_type = "categorical_level"

            if not len(values):
                continue
        else:
            continue

        feature_rows = []

        for order, value in enumerate(values):
            changed = sample.copy()
            changed[feature] = value
            prediction = float(np.mean(predict_ordinary(bundle, changed)))
            feature_rows.append(
                {
                    "model_key": bundle["model_key"],
                    "target": bundle["target"],
                    "mode": bundle["mode"],
                    "feature": feature,
                    "sensitivity_type": sensitivity_type,
                    "value_order": order,
                    "feature_value": value,
                    "baseline_mean_prediction": baseline,
                    "mean_prediction": prediction,
                    "prediction_change": prediction - baseline,
                    "sample_rows": len(sample),
                    "feature_coverage_fraction": float(
                        test_df[feature].notna().mean()
                    ),
                    "interpretation_warning": (
                        "One-feature model response, not a causal effect; "
                        "sparse coverage and correlated inputs may change direction."
                    ),
                }
            )

        if sensitivity_type == "numeric_quantile" and len(feature_rows) >= 2:
            first = feature_rows[0]["mean_prediction"]
            last = feature_rows[-1]["mean_prediction"]
            direction = (
                "increasing"
                if last > first
                else "decreasing"
                if last < first
                else "flat"
            )
        else:
            direction = "categorical_comparison"

        for row in feature_rows:
            row["observed_direction"] = direction
            rows.append(row)

    return pd.DataFrame(rows)


def correlation_evidence() -> dict[str, dict[str, float]]:
    frame = pd.read_csv(
        get_experiment_config(
            "model3_elongation_yield",
            "elongation_percent",
            "process_only",
        )["dataset_path"],
        low_memory=False,
    )
    pairs = {
        "uts_yield": ("uts_MPa", "yield_strength_MPa"),
        "uts_elongation": ("uts_MPa", "elongation_percent"),
        "yield_elongation": (
            "yield_strength_MPa",
            "elongation_percent",
        ),
    }
    result: dict[str, dict[str, float]] = {}

    for key, columns in pairs.items():
        paired = frame[list(columns)].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        result[key] = {
            "rows": int(len(paired)),
            "spearman": float(paired.corr(method="spearman").iloc[0, 1]),
        }

    return result


def relationship_evidence_rows(
    run_dir: Path,
    importance: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    correlations = correlation_evidence()
    physical_path = run_dir / "tables" / "physical_checks.csv"
    physical = pd.read_csv(physical_path) if physical_path.exists() else pd.DataFrame()
    stress_supported = bool(
        not physical.empty
        and physical["all_curve_slopes_negative"].fillna(False).all()
        and physical["stress_scan_monotonic_nonincreasing"].fillna(False).all()
    )

    def coverage_for(target: str, feature: str) -> float:
        selected = coverage.loc[
            coverage["target"].eq(target) & coverage["feature"].eq(feature),
            "coverage_fraction",
        ]
        return float(selected.max()) if len(selected) else 0.0

    def importance_for(target: str, feature: str) -> float:
        selected = importance.loc[
            importance["target"].eq(target) & importance["feature"].eq(feature),
            "permutation_mae_increase_mean",
        ]
        return float(selected.max()) if len(selected) else 0.0

    orientation_coverage = min(
        coverage_for(target, "build_orientation")
        for target in [
            "uts_MPa",
            "yield_strength_MPa",
            "elongation_percent",
            "log10_fatigue_life_cycles",
        ]
    )
    orientation_importance = max(
        importance_for(target, "build_orientation")
        for target in [
            "uts_MPa",
            "yield_strength_MPa",
            "elongation_percent",
            "log10_fatigue_life_cycles",
        ]
    )
    orientation_status = (
        "model_and_data_support"
        if orientation_coverage >= 0.5 and orientation_importance > 0
        else "current_data_insufficient"
    )

    return pd.DataFrame(
        [
            {
                "relationship_id": "process_to_defect",
                "relationship": "process parameters -> melt-pool behaviour -> defects",
                "status": "literature_mechanism_only",
                "evidence": "No melt-pool variable or defect prediction target is available.",
                "recommended_use": "Literature discussion only; do not claim model validation.",
            },
            {
                "relationship_id": "defect_to_static",
                "relationship": "porosity/LoF -> lower static properties",
                "status": "current_data_insufficient",
                "evidence": (
                    f"UTS defect_type coverage={coverage_for('uts_MPa', 'defect_type'):.3f}; "
                    f"porosity coverage={coverage_for('uts_MPa', 'porosity_percent'):.3f}."
                ),
                "recommended_use": "Retain validation tests in defect-sensitive regions.",
            },
            {
                "relationship_id": "defect_to_fatigue",
                "relationship": "defect geometry/location -> fatigue life",
                "status": "current_data_insufficient",
                "evidence": (
                    "Defect size, position, and sharpness are not structured; "
                    f"defect_type coverage={coverage_for('log10_fatigue_life_cycles', 'defect_type'):.3f}."
                ),
                "recommended_use": "Literature-supported risk note; no model reduction.",
            },
            {
                "relationship_id": "surface_to_fatigue",
                "relationship": "surface condition -> fatigue life",
                "status": "current_data_insufficient",
                "evidence": (
                    "Fatigue surface_condition coverage="
                    f"{coverage_for('log10_fatigue_life_cycles', 'surface_condition'):.3f}."
                ),
                "recommended_use": "Keep as-built and machined/polished validation conditions.",
            },
            {
                "relationship_id": "heat_treatment_to_properties",
                "relationship": "heat treatment -> static/fatigue properties",
                "status": "current_data_insufficient",
                "evidence": (
                    f"UTS coverage={coverage_for('uts_MPa', 'heat_treatment'):.3f}; "
                    "regimes are heterogeneous."
                ),
                "recommended_use": "Analyse by alloy and treatment; avoid a universal direction.",
            },
            {
                "relationship_id": "orientation_to_properties",
                "relationship": "build orientation -> tensile/fatigue anisotropy",
                "status": orientation_status,
                "evidence": (
                    f"Minimum target coverage={orientation_coverage:.3f}; "
                    f"maximum holdout MAE increase={orientation_importance:.4g}."
                ),
                "recommended_use": "Preserve representative orientation tests.",
            },
            {
                "relationship_id": "uts_to_yield",
                "relationship": "UTS is positively associated with yield strength",
                "status": "model_and_data_support",
                "evidence": (
                    f"Paired rows={correlations['uts_yield']['rows']}; "
                    f"Spearman={correlations['uts_yield']['spearman']:.3f}."
                ),
                "recommended_use": "Report as an association, not interchangeability.",
            },
            {
                "relationship_id": "strength_to_elongation",
                "relationship": "strength is often inversely associated with elongation",
                "status": "model_and_data_support",
                "evidence": (
                    f"UTS/elongation Spearman={correlations['uts_elongation']['spearman']:.3f}; "
                    "the association is weak and alloy-dependent."
                ),
                "recommended_use": "Report conditionally and stratify by alloy family.",
            },
            {
                "relationship_id": "stress_to_fatigue",
                "relationship": "higher stress amplitude -> shorter fatigue life",
                "status": (
                    "model_and_data_support"
                    if stress_supported
                    else "current_data_insufficient"
                ),
                "evidence": (
                    "All fitted Basquin slopes are negative and stress scans are "
                    "monotonic." if stress_supported else "Physical checks missing."
                ),
                "recommended_use": "Use as the strongest fatigue testing relationship.",
            },
        ]
    )


def plot_feature_importance(df: pd.DataFrame, output_path: Path) -> None:
    targets = list(df["target"].drop_duplicates())
    panel_height = 300
    width = 1000
    height = panel_height * len(targets)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.title{font-size:18px;font-weight:bold}.label{font-size:12px}'
        '.axis{font-size:11px;fill:#555}</style>',
    ]

    for panel_index, target in enumerate(targets):
        selected = (
            df.loc[df["target"].eq(target)]
            .nlargest(10, "permutation_mae_increase_mean")
            .sort_values("permutation_mae_increase_mean")
        )
        top = panel_index * panel_height
        parts.append(
            f'<text class="title" x="20" y="{top + 28}">'
            f"{escape(str(target))}</text>"
        )
        max_value = max(
            float(selected["permutation_mae_increase_mean"].max()),
            1e-12,
        )

        for row_index, row in enumerate(selected.itertuples(index=False)):
            y = top + 52 + row_index * 22
            value = max(
                float(row.permutation_mae_increase_mean),
                0.0,
            )
            bar_width = 550 * value / max_value
            parts.append(
                f'<text class="label" x="20" y="{y + 12}">'
                f"{escape(str(row.feature))}</text>"
            )
            parts.append(
                f'<rect x="235" y="{y}" width="{bar_width:.2f}" '
                'height="15" fill="#3366cc"/>'
            )
            parts.append(
                f'<text class="axis" x="{245 + bar_width:.2f}" '
                f'y="{y + 12}">{value:.4g}</text>'
            )

        parts.append(
            f'<text class="axis" x="235" y="{top + 286}">'
            "Holdout MAE increase after permutation</text>"
        )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def plot_sensitivity(df: pd.DataFrame, output_path: Path) -> None:
    numeric = df.loc[df["sensitivity_type"].eq("numeric_quantile")].copy()

    if numeric.empty:
        return

    priority = {
        "stress_amplitude_MPa": 0,
        "porosity_percent": 1,
        "ved_J_mm3": 2,
        "relative_density_percent": 3,
    }
    numeric["_plot_priority"] = numeric["feature"].map(priority).fillna(99)
    group_keys = (
        numeric[["target", "feature", "_plot_priority"]]
        .drop_duplicates()
        .sort_values(["_plot_priority", "target"])
        .head(12)
    )
    groups = [
        (
            (row.target, row.feature),
            numeric.loc[
                numeric["target"].eq(row.target)
                & numeric["feature"].eq(row.feature)
            ],
        )
        for row in group_keys.itertuples(index=False)
    ]
    columns = 2
    panel_width = 520
    panel_height = 260
    rows = int(np.ceil(len(groups) / columns))
    width = columns * panel_width
    height = rows * panel_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.title{font-size:14px;font-weight:bold}.axis{font-size:10px;fill:#555}'
        '</style>',
    ]

    for index, ((target, feature), group) in enumerate(groups):
        panel_x = (index % columns) * panel_width
        panel_y = (index // columns) * panel_height
        x_values = pd.to_numeric(
            group["feature_value"],
            errors="coerce",
        ).to_numpy(dtype=float)
        y_values = pd.to_numeric(
            group["mean_prediction"],
            errors="coerce",
        ).to_numpy(dtype=float)
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)
        points = []

        for x_value, y_value in zip(x_values, y_values, strict=False):
            x = panel_x + 75 + 390 * (x_value - x_min) / x_span
            y = panel_y + 205 - 145 * (y_value - y_min) / y_span
            points.append((x, y))

        parts.append(
            f'<text class="title" x="{panel_x + 20}" y="{panel_y + 25}">'
            f"{escape(str(target))}: {escape(str(feature))}</text>"
        )
        parts.append(
            f'<line x1="{panel_x + 75}" y1="{panel_y + 205}" '
            f'x2="{panel_x + 470}" y2="{panel_y + 205}" stroke="#777"/>'
        )
        parts.append(
            f'<line x1="{panel_x + 75}" y1="{panel_y + 55}" '
            f'x2="{panel_x + 75}" y2="{panel_y + 205}" stroke="#777"/>'
        )
        parts.append(
            '<polyline fill="none" stroke="#3366cc" stroke-width="2" points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            + '"/>'
        )

        for x, y in points:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#3366cc"/>'
            )

        parts.append(
            f'<text class="axis" x="{panel_x + 75}" y="{panel_y + 225}">'
            f"{x_min:.3g}</text>"
        )
        parts.append(
            f'<text class="axis" x="{panel_x + 440}" y="{panel_y + 225}">'
            f"{x_max:.3g}</text>"
        )
        parts.append(
            f'<text class="axis" x="{panel_x + 18}" y="{panel_y + 65}">'
            f"{y_max:.3g}</text>"
        )
        parts.append(
            f'<text class="axis" x="{panel_x + 18}" y="{panel_y + 205}">'
            f"{y_min:.3g}</text>"
        )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def run_model_explanation(
    run_dir: str | Path,
    *,
    mode: str = "process_only",
    repeats: int = 5,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    table_dir = run_dir / "tables"
    figure_dir = run_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    importance_parts = []
    error_parts = []
    coverage_parts = []
    sensitivity_parts = []

    for entry in ordinary_entries(run_dir, mode):
        bundle = joblib.load(run_dir / entry["artifact"])
        frame, test_df = evaluation_frames(
            entry["model_key"],
            entry["target"],
            entry["mode"],
        )
        importance_parts.append(
            permutation_importance_rows(
                bundle,
                test_df,
                repeats=repeats,
            )
        )
        error_parts.append(grouped_error_rows(bundle, test_df))
        coverage_parts.append(coverage_rows(bundle, frame))
        sensitivity_parts.append(sensitivity_rows(bundle, test_df))

    importance = pd.concat(importance_parts, ignore_index=True)
    errors = pd.concat(error_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    relationships = relationship_evidence_rows(
        run_dir,
        importance,
        coverage,
    )
    outputs = {
        "feature_importance": table_dir / "feature_importance.csv",
        "grouped_error_analysis": table_dir / "grouped_error_analysis.csv",
        "variable_coverage": table_dir / "variable_coverage.csv",
        "sensitivity_analysis": table_dir / "sensitivity_analysis.csv",
        "relationship_evidence": table_dir / "relationship_evidence.csv",
        "feature_importance_figure": figure_dir / "feature_importance.svg",
        "sensitivity_figure": figure_dir / "sensitivity_analysis.svg",
    }

    for frame, key in [
        (importance, "feature_importance"),
        (errors, "grouped_error_analysis"),
        (coverage, "variable_coverage"),
        (sensitivity, "sensitivity_analysis"),
        (relationships, "relationship_evidence"),
    ]:
        frame.to_csv(outputs[key], index=False, encoding="utf-8-sig")

    plot_feature_importance(importance, outputs["feature_importance_figure"])
    plot_sensitivity(sensitivity, outputs["sensitivity_figure"])
    return outputs
