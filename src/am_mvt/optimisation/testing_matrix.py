from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from am_mvt.config import get_path
from am_mvt.modelling.experiment_inference import predict_scenarios


MATRIX_COLUMNS = [
    "priority",
    "alloy_family",
    "am_process",
    "build_orientation",
    "surface_condition",
    "test_type",
    "target_property",
    "recommended_test_condition",
    "reason",
    "supporting_features",
    "model_evidence",
    "coverage_risk",
    "confidence_level",
    "recommendation_action",
    "n_rows",
    "source_count",
    "group_count",
    "test_r2",
    "prediction_interval_width",
    "shap_support",
    "needs_validation",
]


def selected_metric(summary: pd.DataFrame, target: str) -> pd.Series:
    selected_mask = (
        summary["selected"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .eq("true")
    )
    selected = summary.loc[
        summary["target"].eq(target)
        & summary["route"].eq("ordinary_regression")
        & selected_mask
    ]
    return selected.iloc[0] if len(selected) else pd.Series(dtype=object)


def evidence_text(relationships: pd.DataFrame, relationship_id: str) -> str:
    selected = relationships.loc[
        relationships["relationship_id"].eq(relationship_id),
        "evidence",
    ]
    return str(selected.iloc[0]) if len(selected) else "Evidence unavailable."


def build_reduced_testing_matrix(
    summary: pd.DataFrame,
    relationships: pd.DataFrame,
    combination_coverage: pd.DataFrame | None = None,
    importance_comparison: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fatigue = selected_metric(summary, "log10_fatigue_life_cycles")
    uts = selected_metric(summary, "uts_MPa")
    yield_row = selected_metric(summary, "yield_strength_MPa")
    elongation = selected_metric(summary, "elongation_percent")
    modulus = selected_metric(summary, "youngs_modulus_GPa")

    def metric_line(row: pd.Series) -> str:
        if row.empty:
            return "Model metric unavailable."
        return (
            f"{row.get('candidate')} holdout R2={row.get('test_r2'):.3f}, "
            f"MAE={row.get('test_mae'):.3f}."
        )

    rows = [
        {
            "priority": 1,
            "alloy_family": "each qualified alloy family",
            "am_process": "each represented AM process",
            "build_orientation": "representative orientations",
            "surface_condition": "controlled within comparison",
            "test_type": "S-N fatigue",
            "target_property": "fatigue life",
            "recommended_test_condition": (
                "Retain multiple stress-amplitude levels and relevant R-ratios; "
                "do not reduce each condition to one stress level."
            ),
            "reason": (
                "Stress amplitude is the strongest physically verified fatigue "
                "relationship, while exact-life prediction remains moderate."
            ),
            "supporting_features": "stress_amplitude_MPa;r_ratio;alloy_family",
            "model_evidence": (
                f"{evidence_text(relationships, 'stress_to_fatigue')} "
                f"{metric_line(fatigue)}"
            ),
            "coverage_risk": "medium",
            "confidence_level": "high_direction_moderate_prediction",
            "needs_validation": True,
        },
        {
            "priority": 2,
            "alloy_family": "each qualified alloy family",
            "am_process": "each represented AM process",
            "build_orientation": "at least principal build orientations",
            "surface_condition": "fixed within orientation comparison",
            "test_type": "tensile and S-N fatigue",
            "target_property": "UTS;yield strength;elongation;fatigue life",
            "recommended_test_condition": (
                "Preserve orientation contrasts rather than pooling all "
                "orientations into one representative condition."
            ),
            "reason": "Orientation coverage is high and anisotropy remains a qualification risk.",
            "supporting_features": "build_orientation;test_direction",
            "model_evidence": evidence_text(
                relationships,
                "orientation_to_properties",
            ),
            "coverage_risk": "medium_due_to_label_aliases",
            "confidence_level": "moderate",
            "needs_validation": True,
        },
        {
            "priority": 3,
            "alloy_family": "material-specific",
            "am_process": "process-specific",
            "build_orientation": "representative orientation",
            "surface_condition": "as-built and machined/polished",
            "test_type": "S-N fatigue",
            "target_property": "fatigue life",
            "recommended_test_condition": (
                "Keep paired surface-condition validation; do not eliminate "
                "as-built testing from the current evidence."
            ),
            "reason": "Surface condition is failure-relevant but current structured coverage is sparse.",
            "supporting_features": "surface_condition;stress_amplitude_MPa",
            "model_evidence": evidence_text(
                relationships,
                "surface_to_fatigue",
            ),
            "coverage_risk": "high",
            "confidence_level": "low_model_high_domain_risk",
            "needs_validation": True,
        },
        {
            "priority": 4,
            "alloy_family": "material-specific",
            "am_process": "process-specific",
            "build_orientation": "representative orientation",
            "surface_condition": "controlled",
            "test_type": "tensile and S-N fatigue",
            "target_property": "static properties;fatigue life",
            "recommended_test_condition": (
                "Retain defect-rich and nominal conditions where available; "
                "record defect size, location, and morphology."
            ),
            "reason": "Defect geometry is high risk but is not sufficiently structured for reduction.",
            "supporting_features": "porosity_percent;defect_type",
            "model_evidence": evidence_text(
                relationships,
                "defect_to_fatigue",
            ),
            "coverage_risk": "very_high",
            "confidence_level": "insufficient_for_reduction",
            "needs_validation": True,
        },
        {
            "priority": 5,
            "alloy_family": "material-specific",
            "am_process": "process-specific",
            "build_orientation": "representative orientation",
            "surface_condition": "controlled",
            "test_type": "tensile",
            "target_property": "UTS;yield strength;elongation",
            "recommended_test_condition": (
                "Use UTS and yield jointly for consistency checks, while "
                "retaining elongation as a separate response."
            ),
            "reason": "UTS and yield are strongly associated; elongation has a weaker alloy-dependent trade-off.",
            "supporting_features": "uts_MPa;yield_strength_MPa;elongation_percent",
            "model_evidence": (
                f"{evidence_text(relationships, 'uts_to_yield')} "
                f"{evidence_text(relationships, 'strength_to_elongation')} "
                f"{metric_line(uts)} {metric_line(yield_row)} "
                f"{metric_line(elongation)}"
            ),
            "coverage_risk": "medium",
            "confidence_level": "high_for_uts_yield_moderate_for_elongation",
            "needs_validation": True,
        },
        {
            "priority": 6,
            "alloy_family": "material-specific",
            "am_process": "process-specific",
            "build_orientation": "principal orientation",
            "surface_condition": "controlled",
            "test_type": "elastic modulus",
            "target_property": "Young's modulus",
            "recommended_test_condition": (
                "Use a smaller representative modulus set only within covered "
                "alloy/process domains; retain validation for new domains."
            ),
            "reason": "The modulus model is useful, but source diversity is lower than for other static targets.",
            "supporting_features": "alloy_family;am_process;build_orientation",
            "model_evidence": metric_line(modulus),
            "coverage_risk": "medium_high",
            "confidence_level": "moderate",
            "needs_validation": True,
        },
        {
            "priority": 7,
            "alloy_family": "material-specific",
            "am_process": "process-specific",
            "build_orientation": "matched",
            "surface_condition": "matched",
            "test_type": "tensile and S-N fatigue",
            "target_property": "all current targets",
            "recommended_test_condition": (
                "Keep untreated and treated validation pairs for each distinct "
                "heat-treatment regime; do not assume a universal benefit."
            ),
            "reason": "Heat-treatment labels and coverage are too heterogeneous for general reduction.",
            "supporting_features": "heat_treatment;alloy_family",
            "model_evidence": evidence_text(
                relationships,
                "heat_treatment_to_properties",
            ),
            "coverage_risk": "high",
            "confidence_level": "insufficient_for_reduction",
            "needs_validation": True,
        },
    ]
    principle_actions = {
        1: "retain_core",
        2: "retain_core",
        3: "retain_validation",
        4: "collect_more_data",
        5: "retain_core",
        6: "retain_validation",
        7: "retain_validation",
    }
    for row in rows:
        row.update(
            {
                "recommendation_action": principle_actions[row["priority"]],
                "n_rows": np.nan,
                "source_count": np.nan,
                "group_count": np.nan,
                "test_r2": np.nan,
                "prediction_interval_width": np.nan,
                "shap_support": "principle_level",
            }
        )

    condition_rows: list[dict[str, object]] = []
    if combination_coverage is not None and not combination_coverage.empty:
        comparison = (
            importance_comparison
            if importance_comparison is not None
            else pd.DataFrame()
        )
        target_labels = {
            "uts_MPa": ("tensile", "UTS"),
            "yield_strength_MPa": ("tensile", "yield strength"),
            "elongation_percent": ("tensile", "elongation"),
            "youngs_modulus_GPa": ("elastic modulus", "Young's modulus"),
            "log10_fatigue_life_cycles": ("S-N fatigue", "fatigue life"),
        }
        for target, target_group in combination_coverage.groupby("target"):
            metric = selected_metric(summary, str(target))
            if metric.empty:
                continue
            test_r2 = float(metric.get("test_r2", np.nan))
            radius = float(metric.get("conformal_q90", np.nan))
            interval_width = 2 * radius if np.isfinite(radius) else np.nan
            if {
                "target",
                "feature",
                "evidence_stability",
            } <= set(comparison.columns):
                supported = comparison.loc[
                    comparison["target"].eq(target)
                    & comparison["evidence_stability"].eq(
                        "supported_by_both"
                    ),
                    "feature",
                ].dropna().astype(str).tolist()
            else:
                supported = []
            supported_text = ";".join(supported[:8]) or "no_dual_support"
            test_type, property_label = target_labels.get(
                str(target),
                ("mechanical test", str(target)),
            )
            for coverage_row in target_group.nlargest(20, "rows").itertuples(
                index=False
            ):
                n_rows = int(coverage_row.rows)
                source_count = int(coverage_row.source_count)
                group_count = int(coverage_row.group_count)
                coverage_level = str(coverage_row.coverage_level)
                dual_support = bool(supported)
                combination_known = all(
                    str(
                        getattr(coverage_row, column, "missing")
                    ).strip().lower()
                    not in {"", "missing", "nan", "<na>"}
                    for column in [
                        "alloy_family",
                        "am_process",
                        "build_orientation",
                        "surface_condition",
                    ]
                )
                reducible = (
                    coverage_level == "adequate"
                    and source_count >= 3
                    and group_count >= 3
                    and np.isfinite(test_r2)
                    and test_r2 >= 0.5
                    and np.isfinite(interval_width)
                    and dual_support
                    and combination_known
                )
                if reducible:
                    action = "candidate_for_reduction"
                    confidence = "moderate_model_supported"
                    risk = "low_to_medium"
                elif coverage_level == "sparse":
                    action = "collect_more_data"
                    confidence = "insufficient"
                    risk = "high"
                else:
                    action = "retain_validation"
                    confidence = "limited"
                    risk = "medium_to_high"
                condition_rows.append(
                    {
                        "priority": len(condition_rows) + 10,
                        "alloy_family": getattr(
                            coverage_row,
                            "alloy_family",
                            "missing",
                        ),
                        "am_process": getattr(
                            coverage_row,
                            "am_process",
                            "missing",
                        ),
                        "build_orientation": getattr(
                            coverage_row,
                            "build_orientation",
                            "missing",
                        ),
                        "surface_condition": getattr(
                            coverage_row,
                            "surface_condition",
                            "missing",
                        ),
                        "test_type": test_type,
                        "target_property": property_label,
                        "recommended_test_condition": (
                            "Use this represented combination as a candidate "
                            "condition for a reduced matrix."
                            if reducible
                            else "Retain validation or collect additional data "
                            "before reducing this condition."
                        ),
                        "reason": (
                            f"Combination coverage={coverage_level}; "
                            f"holdout R2={test_r2:.3f}."
                        ),
                        "supporting_features": supported_text,
                        "model_evidence": (
                            f"{metric.get('candidate')} holdout R2={test_r2:.3f}; "
                            f"90% interval width={interval_width:.3g}."
                        ),
                        "coverage_risk": risk,
                        "confidence_level": confidence,
                        "recommendation_action": action,
                        "n_rows": n_rows,
                        "source_count": source_count,
                        "group_count": group_count,
                        "test_r2": test_r2,
                        "prediction_interval_width": interval_width,
                        "shap_support": supported_text,
                        "needs_validation": not reducible,
                    }
                )

    return pd.DataFrame(rows + condition_rows, columns=MATRIX_COLUMNS)


def generate_testing_matrix(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    scenario_input: str | Path | None = None,
    scenario_output: str | Path | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    summary = pd.read_csv(run_dir / "tables" / "experiment_summary.csv")
    relationship_path = run_dir / "tables" / "relationship_evidence.csv"

    if not relationship_path.exists():
        raise FileNotFoundError(
            "Run scripts/07_explain_models.py before Step 08."
        )

    relationships = pd.read_csv(relationship_path)
    combination_path = run_dir / "tables" / "combination_coverage.csv"
    comparison_path = (
        run_dir / "tables" / "feature_importance_comparison.csv"
    )
    combination_coverage = (
        pd.read_csv(combination_path)
        if combination_path.exists()
        else pd.DataFrame()
    )
    importance_comparison = (
        pd.read_csv(comparison_path)
        if comparison_path.exists()
        else pd.DataFrame()
    )
    matrix = build_reduced_testing_matrix(
        summary,
        relationships,
        combination_coverage=combination_coverage,
        importance_comparison=importance_comparison,
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else get_path("outputs", "tables", "reduced_testing_matrix.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, index=False, encoding="utf-8-sig")
    outputs = {"testing_matrix": output_path}

    if scenario_input is not None:
        scenario_output = (
            Path(scenario_output)
            if scenario_output is not None
            else run_dir / "example_scenario_predictions.csv"
        )
        predict_scenarios(
            run_dir,
            scenario_input,
            scenario_output,
            mode="process_only",
        )
        outputs["scenario_predictions"] = scenario_output

    return outputs
