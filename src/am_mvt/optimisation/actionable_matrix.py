from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from am_mvt.config import get_path, load_config
from am_mvt.modelling.fatigue_protocol import protocolise_fatigue_data
from am_mvt.optimisation.domain_readiness import (
    build_alloy_process_domain_readiness,
    client_target_template,
    domain_priority_shortlist,
)


STATIC_TARGETS = ["uts_MPa", "yield_strength_MPa"]
STATIC_AUXILIARY_TARGETS = ["elongation_percent", "youngs_modulus_GPa"]
FATIGUE_TARGET = "log10_fatigue_life_cycles"

STATIC_GROUP_COLUMNS = [
    "alloy",
    "alloy_family",
    "am_process",
    "build_orientation",
    "surface_condition",
    "heat_treatment",
]

FATIGUE_GROUP_COLUMNS = [
    "alloy",
    "alloy_family",
    "am_process",
    "build_orientation",
    "surface_condition",
    "heat_treatment",
    "r_ratio",
    "test_temperature_C",
    "frequency_Hz",
]

PROCESS_NUMERIC_COLUMNS = [
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "porosity_percent",
    "relative_density_percent",
]

CRITICAL_STATIC_FEATURES = [
    "alloy",
    "am_process",
    "build_orientation",
    "surface_condition",
    "laser_power_W",
    "scan_speed_mm_s",
    "layer_thickness_um",
]

ACTIONS = {
    "retain_core",
    "retain_validation",
    "pilot_validation",
    "pilot_reduction_candidate",
    "candidate_for_reduction",
    "collect_more_data",
    "not_assessable",
}


@dataclass(frozen=True)
class ActionableMatrixConfig:
    static_budgets: tuple[int, ...] = (24, 36, 48)
    fatigue_budgets: tuple[int, ...] = (30, 45, 60)
    static_replicates: int = 3
    fatigue_stress_levels: int = 5
    fatigue_replicates_per_level: int = 3
    static_green_rows: int = 100
    static_amber_rows: int = 30
    fatigue_green_rows: int = 150
    fatigue_amber_rows: int = 50
    min_sources_green: int = 3
    min_groups_green: int = 5
    min_sources_amber: int = 2
    min_groups_amber: int = 3
    min_condition_rows: int = 20
    min_condition_sources: int = 2
    min_condition_groups: int = 3
    min_critical_coverage: float = 0.70
    static_reduction_oof_r2: float = 0.50
    static_pilot_oof_r2: float = 0.35
    static_max_local_nmae: float = 0.30
    static_min_interval_coverage: float = 0.85
    static_max_interval_width_iqr: float = 0.75
    fatigue_reduction_oof_r2: float = 0.40
    fatigue_pilot_oof_r2: float = 0.20
    fatigue_min_interval_coverage: float = 0.85
    fatigue_max_log_half_width: float = 0.30
    fatigue_min_route_rank_correlation: float = 0.60
    max_source_dominance: float = 0.60
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "uncertainty": 0.30,
            "coverage_gap": 0.25,
            "local_error": 0.20,
            "domain_risk": 0.15,
            "diversity": 0.10,
        }
    )


def load_actionable_matrix_config() -> ActionableMatrixConfig:
    values = dict(load_config().get("testing_matrix", {}))
    for key in ["static_budgets", "fatigue_budgets"]:
        if key in values:
            values[key] = tuple(int(value) for value in values[key])
    return ActionableMatrixConfig(**values)


def _clean_group_values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
        if column in {"r_ratio", "test_temperature_C", "frequency_Hz"}:
            numeric = pd.to_numeric(result[column], errors="coerce")
            if column == "r_ratio":
                result[column] = numeric.round(2)
            else:
                result[column] = (numeric / 5).round() * 5
            result[column] = result[column].astype("string").fillna("missing")
        elif column == "build_orientation":
            orientation = (
                result[column]
                .astype("string")
                .str.strip()
                .str.replace(r"\s*(?:deg|degree|degrees|°)\s*$", "", regex=True)
            )
            numeric_orientation = pd.to_numeric(orientation, errors="coerce")
            numeric_mask = numeric_orientation.notna()
            orientation.loc[numeric_mask] = numeric_orientation.loc[numeric_mask].map(
                lambda value: f"{float(value):g}"
            )
            result[column] = orientation.replace("", "missing").fillna("missing")
        else:
            result[column] = (
                result[column]
                .astype("string")
                .str.strip()
                .replace("", "missing")
                .fillna("missing")
            )
    return result


def _selected_summary(summary: pd.DataFrame, target: str) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    mode = (
        summary["mode"]
        if "mode" in summary
        else pd.Series("process_only", index=summary.index)
    )
    selected = summary.loc[
        summary["target"].eq(target)
        & summary["route"].eq("ordinary_regression")
        & mode.eq("process_only")
        & summary["selected"]
        .astype("string")
        .str.lower()
        .eq("true")
        .fillna(False)
    ]
    return selected.iloc[0] if len(selected) else pd.Series(dtype=object)


def _safe_float(value: Any, default: float = np.nan) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else default


def _target_iqr(frame: pd.DataFrame) -> float:
    values = pd.to_numeric(frame["y_true"], errors="coerce").dropna()
    if values.empty:
        return np.nan
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    return iqr if iqr > 0 else float(values.std(ddof=0))


def _medoid_row(group: pd.DataFrame) -> pd.Series:
    numeric_columns = [
        column
        for column in PROCESS_NUMERIC_COLUMNS
        if column in group
        and pd.to_numeric(group[column], errors="coerce").notna().any()
    ]
    if not numeric_columns:
        return group.sort_index().iloc[0]
    numeric = group[numeric_columns].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    scale = numeric.quantile(0.75) - numeric.quantile(0.25)
    scale = scale.replace(0, 1).fillna(1)
    distance = (((numeric.fillna(medians) - medians) / scale) ** 2).sum(axis=1)
    return group.loc[distance.sort_values(kind="stable").index[0]]


def _source_dominance(group: pd.DataFrame) -> float:
    if "source_id" not in group or group["source_id"].dropna().empty:
        return 1.0
    counts = group["source_id"].astype("string").value_counts(dropna=True)
    return float(counts.iloc[0] / counts.sum()) if len(counts) else 1.0


def _distinct_count(group: pd.DataFrame, candidates: list[str]) -> int:
    for column in candidates:
        if column in group:
            return int(group[column].nunique(dropna=True))
    return 0


def _domain_group(frame: pd.DataFrame, alloy: Any, process: Any) -> pd.DataFrame:
    return frame.loc[frame["alloy"].eq(alloy) & frame["am_process"].eq(process)]


def _coverage_fraction(group: pd.DataFrame, columns: list[str]) -> float:
    available = [column for column in columns if column in group]
    if not available:
        return 0.0
    fractions = []
    for column in available:
        values = group[column]
        normalised = values.astype("string").str.strip().str.lower()
        present = values.notna() & ~normalised.isin(
            ["", "missing", "nan", "<na>", "none"]
        )
        fractions.append(float(present.mean()))
    return float(np.mean(fractions)) if fractions else 0.0


def _evidence_tier(
    *,
    rows: int,
    sources: int,
    groups: int,
    critical_coverage: float,
    green_rows: int,
    amber_rows: int,
    config: ActionableMatrixConfig,
) -> str:
    if (
        rows >= green_rows
        and sources >= config.min_sources_green
        and groups >= config.min_groups_green
        and critical_coverage >= config.min_critical_coverage
    ):
        return "green"
    if (
        rows >= amber_rows
        and sources >= config.min_sources_amber
        and groups >= config.min_groups_amber
    ):
        return "amber"
    return "red"


def _metric_components(
    group: pd.DataFrame,
    target_frame: pd.DataFrame,
    summary_row: pd.Series,
) -> dict[str, float]:
    iqr = _target_iqr(target_frame)
    local_mae = float(pd.to_numeric(group["abs_error"], errors="coerce").mean())
    local_nmae = local_mae / iqr if np.isfinite(iqr) and iqr > 0 else np.nan
    interval_coverage = float(
        group.get("interval_hit_90", pd.Series(dtype=float)).astype("boolean").mean()
    )
    half_width = _safe_float(
        group.get("conformal_q90", pd.Series(dtype=float)).median()
    )
    interval_width_iqr = (
        (2 * half_width / iqr)
        if np.isfinite(half_width) and np.isfinite(iqr) and iqr > 0
        else np.nan
    )
    return {
        "oof_r2": _safe_float(summary_row.get("oof_r2")),
        "local_mae": local_mae,
        "local_nmae": local_nmae,
        "interval_coverage_90": interval_coverage,
        "interval_half_width": half_width,
        "interval_width_over_iqr": interval_width_iqr,
    }


def _component_scores(
    *,
    tier: str,
    rows: int,
    sources: int,
    groups: int,
    local_nmae: float,
    interval_width: float,
    diversity: float = 0.0,
    config: ActionableMatrixConfig,
) -> dict[str, float]:
    uncertainty = float(
        np.clip(interval_width if np.isfinite(interval_width) else 1, 0, 1)
    )
    coverage_strength = min(rows / 100, sources / 3, groups / 5, 1)
    coverage_gap = float(1 - coverage_strength)
    local_error = (
        float(np.clip(local_nmae / 0.5, 0, 1)) if np.isfinite(local_nmae) else 1.0
    )
    domain_risk = {"green": 0.2, "amber": 0.6, "red": 1.0}[tier]
    weights = config.score_weights
    score = 100 * (
        weights["uncertainty"] * uncertainty
        + weights["coverage_gap"] * coverage_gap
        + weights["local_error"] * local_error
        + weights["domain_risk"] * domain_risk
        + weights["diversity"] * diversity
    )
    return {
        "uncertainty_score": uncertainty,
        "coverage_gap_score": coverage_gap,
        "local_error_score": local_error,
        "domain_risk_score": domain_risk,
        "diversity_score": diversity,
        "information_value_score": float(score),
    }


def _importance_supported(importance: pd.DataFrame, target: str) -> bool:
    if importance.empty or not {"target", "evidence_stability"} <= set(importance):
        return False
    return bool(
        importance.loc[
            importance["target"].eq(target),
            "evidence_stability",
        ]
        .eq("supported_by_both")
        .any()
    )


def build_static_condition_evidence(
    oof: pd.DataFrame,
    summary: pd.DataFrame,
    importance: pd.DataFrame | None = None,
    *,
    config: ActionableMatrixConfig | None = None,
) -> pd.DataFrame:
    config = config or ActionableMatrixConfig()
    importance = importance if importance is not None else pd.DataFrame()
    selected = oof.loc[
        oof["target"].isin(STATIC_TARGETS)
        & oof["mode"].eq("process_only")
        & oof["route"].eq("ordinary_regression")
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    selected = _clean_group_values(selected, STATIC_GROUP_COLUMNS)
    target_frames = {
        target: selected.loc[selected["target"].eq(target)].copy()
        for target in STATIC_TARGETS
    }
    rows: list[dict[str, Any]] = []
    for values, raw_group in selected.groupby(STATIC_GROUP_COLUMNS, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        medoid = _medoid_row(raw_group)
        metrics_by_target: dict[str, dict[str, float]] = {}
        target_counts = []
        source_counts = []
        group_counts = []
        stable_targets = []
        for target in STATIC_TARGETS:
            target_group = raw_group.loc[raw_group["target"].eq(target)]
            if target_group.empty:
                continue
            target_counts.append(len(target_group))
            source_counts.append(_distinct_count(target_group, ["source_id"]))
            group_counts.append(
                _distinct_count(
                    target_group,
                    ["evaluation_group_id", "modelling_group_id", "source_id"],
                )
            )
            metrics_by_target[target] = _metric_components(
                target_group,
                target_frames[target],
                _selected_summary(summary, target),
            )
            stable_targets.append(_importance_supported(importance, target))
        if not metrics_by_target:
            continue
        observed_primary_targets = sorted(metrics_by_target)
        missing_primary_targets = [
            target for target in STATIC_TARGETS if target not in metrics_by_target
        ]
        condition_rows = min(target_counts)
        condition_sources = min(source_counts)
        condition_groups = min(group_counts)
        domain = _domain_group(selected, values[0], values[2])
        domain_target_rows = []
        domain_source_counts = []
        domain_group_counts = []
        for target in STATIC_TARGETS:
            target_domain = domain.loc[domain["target"].eq(target)]
            if target_domain.empty:
                continue
            domain_target_rows.append(len(target_domain))
            domain_source_counts.append(_distinct_count(target_domain, ["source_id"]))
            domain_group_counts.append(
                _distinct_count(
                    target_domain,
                    ["evaluation_group_id", "modelling_group_id", "source_id"],
                )
            )
        domain_rows = min(domain_target_rows) if domain_target_rows else 0
        domain_sources = min(domain_source_counts) if domain_source_counts else 0
        domain_groups = min(domain_group_counts) if domain_group_counts else 0
        critical_coverage = _coverage_fraction(domain, CRITICAL_STATIC_FEATURES)
        tier = _evidence_tier(
            rows=domain_rows,
            sources=domain_sources,
            groups=domain_groups,
            critical_coverage=critical_coverage,
            green_rows=config.static_green_rows,
            amber_rows=config.static_amber_rows,
            config=config,
        )
        min_oof_r2 = min(metric["oof_r2"] for metric in metrics_by_target.values())
        max_local_nmae = max(
            metric["local_nmae"] for metric in metrics_by_target.values()
        )
        min_interval_coverage = min(
            metric["interval_coverage_90"] for metric in metrics_by_target.values()
        )
        max_interval_width = max(
            metric["interval_width_over_iqr"] for metric in metrics_by_target.values()
        )
        known_condition = all(str(value).lower() != "missing" for value in values)
        gates = {
            "gate_green_domain": tier == "green",
            "gate_condition_rows": condition_rows >= config.min_condition_rows,
            "gate_condition_sources": condition_sources >= config.min_condition_sources,
            "gate_condition_groups": condition_groups >= config.min_condition_groups,
            "gate_model_quality": min_oof_r2 >= config.static_reduction_oof_r2,
            "gate_local_error": max_local_nmae <= config.static_max_local_nmae,
            "gate_interval_coverage": min_interval_coverage
            >= config.static_min_interval_coverage,
            "gate_interval_width": max_interval_width
            <= config.static_max_interval_width_iqr,
            "gate_known_condition": known_condition,
            "gate_not_ood": True,
            "gate_importance_stability": bool(stable_targets) and all(stable_targets),
            "gate_primary_target_coverage": not missing_primary_targets,
        }
        blockers = [name for name, passed in gates.items() if not passed]
        if all(gates.values()):
            action = "candidate_for_reduction"
        elif (
            min_oof_r2 >= config.static_pilot_oof_r2
            and gates["gate_condition_rows"]
            and gates["gate_condition_sources"]
            and gates["gate_primary_target_coverage"]
            and known_condition
        ):
            action = "pilot_reduction_candidate"
        elif tier in {"green", "amber"}:
            action = "retain_validation"
        elif known_condition:
            action = "collect_more_data"
        else:
            action = "not_assessable"
        scores = _component_scores(
            tier=tier,
            rows=condition_rows,
            sources=condition_sources,
            groups=condition_groups,
            local_nmae=max_local_nmae,
            interval_width=max_interval_width,
            diversity=1 - len(raw_group) / max(len(domain), 1),
            config=config,
        )
        row = {
            "condition_id": "static::" + "::".join(map(str, values)),
            "matrix_type": "static_tensile",
            **dict(zip(STATIC_GROUP_COLUMNS, values, strict=False)),
            "test_type": "tensile",
            "medoid_record_id": medoid.get("record_id", ""),
            "medoid_source_id": medoid.get("source_id", ""),
            "medoid_source_file": medoid.get("source_file", ""),
            "primary_targets": ";".join(STATIC_TARGETS),
            "observed_primary_targets": ";".join(observed_primary_targets),
            "missing_primary_targets": ";".join(missing_primary_targets),
            "auxiliary_targets": ";".join(STATIC_AUXILIARY_TARGETS),
            "n_rows": condition_rows,
            "source_count": condition_sources,
            "group_count": condition_groups,
            "domain_n_rows": domain_rows,
            "domain_source_count": domain_sources,
            "domain_group_count": domain_groups,
            "critical_feature_coverage": critical_coverage,
            "source_dominance": _source_dominance(raw_group),
            "ood_status": "observed_medoid_condition",
            "evidence_tier": tier,
            "oof_r2_min": min_oof_r2,
            "local_nmae_max": max_local_nmae,
            "interval_coverage_90_min": min_interval_coverage,
            "interval_width_over_iqr_max": max_interval_width,
            "recommendation_action": action,
            "decision_blocker": ";".join(blockers),
            "planned_replicates": config.static_replicates,
            **gates,
            **scores,
        }
        for column in PROCESS_NUMERIC_COLUMNS + [
            "machine_model",
            "scan_strategy",
            "post_processing",
        ]:
            if column in medoid:
                row[column] = medoid[column]
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["information_value_score", "condition_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _fatigue_route_support(
    oof: pd.DataFrame,
    summary: pd.DataFrame,
    physical_checks: pd.DataFrame,
    config: ActionableMatrixConfig,
) -> tuple[int, bool, float, bool]:
    fatigue = summary.loc[summary["target"].eq(FATIGUE_TARGET)].copy()
    metric_columns = [
        column for column in ["oof_r2", "cv_r2_mean", "test_r2"] if column in fatigue
    ]
    supported_routes = 0
    for route, group in fatigue.groupby("route"):
        if any(
            pd.to_numeric(group[column], errors="coerce").notna().any()
            for column in metric_columns
        ):
            supported_routes += 1
    monotonic = False
    if (
        not physical_checks.empty
        and "stress_scan_monotonic_nonincreasing" in physical_checks
    ):
        monotonic = bool(
            physical_checks["stress_scan_monotonic_nonincreasing"]
            .astype("boolean")
            .fillna(False)
            .all()
        )
    route_oof = oof.loc[
        oof["target"].eq(FATIGUE_TARGET)
        & oof["mode"].eq("process_only")
        & oof["y_pred"].notna()
    ].copy()
    if not route_oof.empty:
        supported_routes = max(supported_routes, int(route_oof["route"].nunique()))
    key_columns = [
        column
        for column in [
            "record_id",
            "source_id",
            "evaluation_group_id",
            "stress_amplitude_MPa",
        ]
        if column in route_oof
    ]
    correlations: list[float] = []
    if key_columns and route_oof["route"].nunique() >= 2:
        route_oof["_match_key"] = (
            route_oof[key_columns]
            .astype("string")
            .agg(
                "::".join,
                axis=1,
            )
        )
        route_oof["_duplicate_order"] = route_oof.groupby(
            ["route", "_match_key"],
            dropna=False,
        ).cumcount()
        pivot = route_oof.pivot_table(
            index=["_match_key", "_duplicate_order"],
            columns="route",
            values="y_pred",
            aggfunc="first",
        )
        for left_index, left in enumerate(pivot.columns):
            for right in pivot.columns[left_index + 1 :]:
                paired = pivot[[left, right]].dropna()
                if len(paired) >= 5:
                    correlations.append(
                        float(paired.corr(method="spearman").iloc[0, 1])
                    )
    minimum_correlation = min(correlations) if correlations else np.nan
    agreement = (
        len(correlations) >= 1
        and minimum_correlation >= config.fatigue_min_route_rank_correlation
    )
    return supported_routes, monotonic, minimum_correlation, agreement


def _nearest_observed_levels(values: pd.Series, count: int) -> list[float]:
    observed = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
    if len(observed) == 0:
        return []
    quantiles = np.linspace(0.20, 0.80, count)
    targets = np.quantile(observed, quantiles)
    if len(observed) < count:
        # Four observed levels are sufficient evidence for a five-level pilot
        # block; interpolate inside the observed range instead of under-filling
        # the physical specimen budget.
        return sorted({float(value) for value in targets})
    selected = []
    for target in targets:
        level = float(observed[np.argmin(np.abs(observed - target))])
        if level not in selected:
            selected.append(level)
    if len(selected) < min(count, len(observed)):
        for level in observed:
            value = float(level)
            if value not in selected:
                selected.append(value)
            if len(selected) == min(count, len(observed)):
                break
    return sorted(selected)


def build_fatigue_condition_evidence(
    oof: pd.DataFrame,
    summary: pd.DataFrame,
    physical_checks: pd.DataFrame | None = None,
    fatigue_frame: pd.DataFrame | None = None,
    *,
    config: ActionableMatrixConfig | None = None,
) -> pd.DataFrame:
    config = config or ActionableMatrixConfig()
    physical_checks = physical_checks if physical_checks is not None else pd.DataFrame()
    oof_fatigue = oof.loc[
        oof["target"].eq(FATIGUE_TARGET)
        & oof["mode"].eq("process_only")
        & oof["route"].eq("ordinary_regression")
    ].copy()
    if oof_fatigue.empty:
        return pd.DataFrame()
    source_frame = (
        fatigue_frame.copy()
        if fatigue_frame is not None and not fatigue_frame.empty
        else oof_fatigue.copy()
    )
    if fatigue_frame is not None and not fatigue_frame.empty:
        source_frame = protocolise_fatigue_data(source_frame)
        source_frame = source_frame.loc[
            source_frame["fatigue_protocol"].eq("e466_conventional")
            & ~source_frame["stress_consistency_status"].eq("review_required")
        ].copy()
    source_frame = _clean_group_values(source_frame, FATIGUE_GROUP_COLUMNS)
    oof_fatigue = _clean_group_values(oof_fatigue, FATIGUE_GROUP_COLUMNS)
    summary_row = _selected_summary(summary, FATIGUE_TARGET)
    (
        route_count,
        monotonic,
        route_rank_correlation,
        route_agreement,
    ) = _fatigue_route_support(oof, summary, physical_checks, config)
    rows: list[dict[str, Any]] = []
    for values, block in source_frame.groupby(FATIGUE_GROUP_COLUMNS, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        mask = pd.Series(True, index=oof_fatigue.index)
        for column, value in zip(FATIGUE_GROUP_COLUMNS, values, strict=False):
            mask &= oof_fatigue[column].eq(value)
        local_oof = oof_fatigue.loc[mask]
        if local_oof.empty:
            continue
        metrics = _metric_components(local_oof, oof_fatigue, summary_row)
        block_rows = len(block)
        source_count = _distinct_count(block, ["source_id"])
        group_count = _distinct_count(
            block,
            ["evaluation_group_id", "modelling_group_id", "source_id"],
        )
        stress_levels = _nearest_observed_levels(
            block.get("stress_amplitude_MPa", pd.Series(dtype=float)),
            config.fatigue_stress_levels,
        )
        domain = _domain_group(source_frame, values[0], values[2])
        domain_rows = len(domain)
        domain_sources = _distinct_count(domain, ["source_id"])
        domain_groups = _distinct_count(
            domain,
            ["evaluation_group_id", "modelling_group_id", "source_id"],
        )
        runout_coverage = (
            float(domain["runout"].notna().mean()) if "runout" in domain else 0.0
        )
        r_known = str(values[6]).lower() != "missing"
        critical_coverage = float(np.mean([runout_coverage, 1.0 if r_known else 0.0]))
        tier = _evidence_tier(
            rows=domain_rows,
            sources=domain_sources,
            groups=domain_groups,
            critical_coverage=critical_coverage,
            green_rows=config.fatigue_green_rows,
            amber_rows=config.fatigue_amber_rows,
            config=config,
        )
        enough_stress_levels = (
            len(
                pd.to_numeric(
                    block.get("stress_amplitude_MPa", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .dropna()
                .unique()
            )
            >= 4
        )
        source_dominance = _source_dominance(domain)
        gates = {
            "gate_green_domain": tier == "green",
            "gate_model_quality": metrics["oof_r2"] >= config.fatigue_reduction_oof_r2,
            "gate_interval_coverage": metrics["interval_coverage_90"]
            >= config.fatigue_min_interval_coverage,
            "gate_interval_half_width": metrics["interval_half_width"]
            <= config.fatigue_max_log_half_width,
            "gate_stress_levels": enough_stress_levels,
            "gate_runout_coverage": runout_coverage >= 0.80,
            "gate_route_agreement": route_agreement,
            "gate_monotonicity": monotonic,
            "gate_source_diversity": source_dominance <= config.max_source_dominance,
            "gate_known_condition": all(
                str(value).lower() != "missing" for value in values
            ),
            "gate_not_ood": True,
        }
        unresolved_condition_fields = [
            column
            for column, value in zip(FATIGUE_GROUP_COLUMNS, values, strict=False)
            if str(value).lower() == "missing"
        ]
        planning_readiness_score = 1 - (
            len(unresolved_condition_fields) / len(FATIGUE_GROUP_COLUMNS)
        )
        blockers = [name for name, passed in gates.items() if not passed]
        if all(gates.values()):
            action = "candidate_for_reduction"
        elif (
            metrics["oof_r2"] >= config.fatigue_pilot_oof_r2
            and gates["gate_stress_levels"]
        ):
            action = "pilot_validation"
        elif tier in {"green", "amber"} and gates["gate_stress_levels"]:
            action = "retain_validation"
        elif gates["gate_stress_levels"]:
            action = "collect_more_data"
        else:
            action = "not_assessable"
        scores = _component_scores(
            tier=tier,
            rows=block_rows,
            sources=source_count,
            groups=group_count,
            local_nmae=metrics["local_nmae"],
            interval_width=min(metrics["interval_half_width"] / 0.6, 1)
            if np.isfinite(metrics["interval_half_width"])
            else 1,
            diversity=1 - len(block) / max(len(domain), 1),
            config=config,
        )
        rows.append(
            {
                "condition_id": "fatigue::" + "::".join(map(str, values)),
                "matrix_type": "fatigue_sn_block",
                **dict(zip(FATIGUE_GROUP_COLUMNS, values, strict=False)),
                "test_type": "S-N fatigue",
                "primary_targets": FATIGUE_TARGET,
                "stress_levels_MPa": ";".join(f"{value:g}" for value in stress_levels),
                "observed_stress_min_MPa": min(stress_levels)
                if stress_levels
                else np.nan,
                "observed_stress_max_MPa": max(stress_levels)
                if stress_levels
                else np.nan,
                "observed_stress_level_count": int(
                    pd.to_numeric(
                        block.get("stress_amplitude_MPa", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    .dropna()
                    .nunique()
                ),
                "planned_stress_levels": config.fatigue_stress_levels,
                "replicates_per_level": config.fatigue_replicates_per_level,
                "planned_specimens": config.fatigue_stress_levels
                * config.fatigue_replicates_per_level,
                "n_rows": block_rows,
                "source_count": source_count,
                "group_count": group_count,
                "domain_n_rows": domain_rows,
                "domain_source_count": domain_sources,
                "domain_group_count": domain_groups,
                "runout_coverage": runout_coverage,
                "source_dominance": source_dominance,
                "ood_status": "observed_sn_block",
                "supported_route_count": route_count,
                "route_rank_correlation_min": route_rank_correlation,
                "evidence_tier": tier,
                "oof_r2": metrics["oof_r2"],
                "local_nmae": metrics["local_nmae"],
                "interval_coverage_90": metrics["interval_coverage_90"],
                "interval_half_width_log10": metrics["interval_half_width"],
                "recommendation_action": action,
                "decision_blocker": ";".join(blockers),
                "condition_completion_required": bool(
                    unresolved_condition_fields
                ),
                "unresolved_condition_count": len(unresolved_condition_fields),
                "unresolved_condition_fields": ";".join(
                    unresolved_condition_fields
                ),
                "planning_readiness_score": planning_readiness_score,
                **gates,
                **scores,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["information_value_score", "condition_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _coverage_tokens(row: pd.Series, matrix_type: str) -> set[str]:
    domain = f"domain::{row.get('alloy')}::{row.get('am_process')}"
    tokens = {domain}
    orientation = str(row.get("build_orientation", "missing"))
    surface = str(row.get("surface_condition", "missing"))
    if orientation.lower() != "missing":
        tokens.add(f"orientation::{domain}::{orientation}")
    if surface.lower() != "missing":
        tokens.add(f"surface::{domain}::{surface}")
    if matrix_type == "fatigue":
        ratio = str(row.get("r_ratio", "missing"))
        if ratio.lower() != "missing":
            tokens.add(f"r_ratio::{domain}::{ratio}")
    return tokens


def _greedy_indices(
    candidates: pd.DataFrame,
    mandatory_tokens: set[str],
    max_conditions: int,
    *,
    stop_when_covered: bool,
) -> tuple[list[int], set[str]]:
    selected_indices: list[int] = []
    uncovered = set(mandatory_tokens)
    remaining = set(candidates.index)
    while remaining and len(selected_indices) < max_conditions:
        if stop_when_covered and not uncovered:
            break
        best_index = None
        best_key: tuple[float, float, str] | None = None
        selected_tokens = (
            set().union(*(candidates.loc[item, "_tokens"] for item in selected_indices))
            if selected_indices
            else set()
        )
        for index in remaining:
            row = candidates.loc[index]
            new_coverage = len(row["_tokens"] & uncovered)
            diversity = len(row["_tokens"] - selected_tokens) / max(
                len(row["_tokens"]), 1
            )
            planning_readiness = _safe_float(
                row.get("planning_readiness_score", 1.0),
                1.0,
            )
            key = (
                float(
                    new_coverage * 100
                    + row["information_value_score"]
                    + 10 * diversity
                    + 20 * planning_readiness
                ),
                float(row["information_value_score"]),
                str(row["condition_id"]),
            )
            if best_key is None or key > best_key:
                best_index = index
                best_key = key
        if best_index is None:
            break
        selected_indices.append(best_index)
        uncovered -= candidates.loc[best_index, "_tokens"]
        remaining.remove(best_index)
    return selected_indices, uncovered


def select_budget_plan(
    evidence: pd.DataFrame,
    *,
    budget: int,
    specimens_per_condition: int,
    matrix_type: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if evidence.empty:
        return evidence.copy(), {
            "matrix_type": matrix_type,
            "budget_specimens": budget,
            "selected_conditions": 0,
            "selected_specimens": 0,
            "minimum_feasible_budget": 0,
            "budget_feasible": False,
            "uncovered_constraints": "no_candidate_conditions",
        }
    candidates = evidence.loc[
        ~evidence["recommendation_action"].eq("not_assessable")
    ].copy()
    if candidates.empty:
        return candidates, {
            "matrix_type": matrix_type,
            "budget_specimens": budget,
            "selected_conditions": 0,
            "selected_specimens": 0,
            "minimum_feasible_budget": 0,
            "budget_feasible": False,
            "uncovered_constraints": "no_assessable_conditions",
        }
    candidates["_tokens"] = candidates.apply(
        lambda row: _coverage_tokens(row, matrix_type), axis=1
    )
    mandatory_tokens: set[str] = set()
    for tokens in candidates.loc[
        candidates["evidence_tier"].isin(["green", "amber"]), "_tokens"
    ]:
        mandatory_tokens.update(tokens)
    max_conditions = budget // specimens_per_condition
    selected_indices, uncovered = _greedy_indices(
        candidates,
        mandatory_tokens,
        max_conditions,
        stop_when_covered=False,
    )
    selected = candidates.loc[selected_indices].copy()
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["budget_specimens"] = budget
    selected["planned_specimens"] = specimens_per_condition
    selected["selection_action"] = "retain_core"
    selected["plan_purpose"] = "evidence_validation_only"
    selected["client_target_status"] = "not_provided"
    selected = selected.drop(columns=["_tokens"])
    if mandatory_tokens:
        minimum_indices, minimum_uncovered = _greedy_indices(
            candidates,
            mandatory_tokens,
            len(candidates),
            stop_when_covered=True,
        )
        minimum_conditions = len(minimum_indices)
        minimum_feasible_budget = (
            minimum_conditions * specimens_per_condition
            if not minimum_uncovered
            else np.nan
        )
    else:
        minimum_feasible_budget = specimens_per_condition
    summary = {
        "matrix_type": matrix_type,
        "budget_specimens": budget,
        "selected_conditions": len(selected),
        "selected_specimens": len(selected) * specimens_per_condition,
        "minimum_feasible_budget": minimum_feasible_budget,
        "budget_feasible": not uncovered,
        "uncovered_constraints": ";".join(sorted(uncovered)),
    }
    return selected, summary


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _matched_step07_rows(table: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if table.empty:
        return table
    matched = table.copy()
    if "mode" in matched:
        matched = matched.loc[matched["mode"].eq("process_only")]
    for column in [
        "alloy",
        "alloy_family",
        "am_process",
        "build_orientation",
        "surface_condition",
    ]:
        if column in matched and column in row:
            matched = matched.loc[matched[column].astype("string").eq(str(row[column]))]
    return matched


def attach_step07_traceability(
    evidence: pd.DataFrame,
    *,
    combination_coverage: pd.DataFrame,
    grouped_errors: pd.DataFrame,
    b2_diagnostics: pd.DataFrame,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    if evidence.empty:
        return evidence
    traced_rows = []
    for _, evidence_row in evidence.iterrows():
        targets = str(evidence_row.get("primary_targets", "")).split(";")
        combination = _matched_step07_rows(combination_coverage, evidence_row)
        if "target" in combination:
            combination = combination.loc[combination["target"].isin(targets)]
        levels = (
            combination.get(
                "coverage_level",
                pd.Series(dtype="string"),
            )
            .dropna()
            .astype("string")
        )
        level_rank = {"sparse": 0, "limited": 1, "adequate": 2}
        worst_level = (
            min(levels, key=lambda value: level_rank.get(str(value), -1))
            if len(levels)
            else "unavailable"
        )

        grouped = grouped_errors.copy()
        if not grouped.empty and "target" in grouped:
            grouped = grouped.loc[grouped["target"].isin(targets)]
        if not grouped.empty and {"group_column", "group_value"} <= set(grouped):
            grouped = grouped.loc[
                grouped.apply(
                    lambda item: str(evidence_row.get(item["group_column"], ""))
                    == str(item["group_value"]),
                    axis=1,
                )
            ]

        b2 = _matched_step07_rows(b2_diagnostics, evidence_row)
        if "target" in b2:
            b2 = b2.loc[b2["target"].isin(targets)]
        stability = {
            target: _importance_supported(importance, target) for target in targets
        }
        traced = evidence_row.to_dict()
        traced.update(
            {
                "step07_combination_coverage": str(worst_level),
                "step07_combination_rows_min": _safe_float(
                    combination.get("rows", pd.Series(dtype=float)).min()
                ),
                "step07_grouped_error_matches": len(grouped),
                "step07_grouped_error_r2_min": _safe_float(
                    grouped.get("r2", pd.Series(dtype=float)).min()
                ),
                "step07_b2_diagnostic_matches": len(b2),
                "step07_b2_r2_min": _safe_float(
                    b2.get("r2", pd.Series(dtype=float)).min()
                ),
                "step07_importance_stability": ";".join(
                    f"{target}:{str(supported).lower()}"
                    for target, supported in stability.items()
                ),
            }
        )
        traced_rows.append(traced)
    return pd.DataFrame(traced_rows)


def _matrix_change_log(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    matrix_type: str,
) -> pd.DataFrame:
    current_actions = (
        current[["condition_id", "recommendation_action"]].rename(
            columns={"recommendation_action": "action_after"}
        )
        if not current.empty
        else pd.DataFrame(columns=["condition_id", "action_after"])
    )
    previous_actions = (
        previous[["condition_id", "recommendation_action"]].rename(
            columns={"recommendation_action": "action_before"}
        )
        if not previous.empty
        and {"condition_id", "recommendation_action"} <= set(previous)
        else pd.DataFrame(columns=["condition_id", "action_before"])
    )
    changed = previous_actions.merge(current_actions, on="condition_id", how="outer")
    changed["matrix_type"] = matrix_type
    changed["change_reason"] = np.where(
        changed["action_before"].fillna("").eq(changed["action_after"].fillna("")),
        "unchanged",
        "evidence_or_gate_update",
    )
    return changed


def generate_actionable_testing_matrix(
    run_dir: str | Path,
    *,
    config: ActionableMatrixConfig | None = None,
    fatigue_frame_path: str | Path | None = None,
) -> dict[str, Path]:
    config = config or load_actionable_matrix_config()
    run_dir = Path(run_dir)
    table_dir = run_dir / "tables"
    summary = pd.read_csv(table_dir / "experiment_summary.csv", low_memory=False)
    oof = pd.read_csv(table_dir / "oof_predictions.csv", low_memory=False)
    importance = _read_optional(table_dir / "feature_importance_comparison.csv")
    physical_checks = _read_optional(table_dir / "physical_checks.csv")
    combination_coverage = _read_optional(table_dir / "combination_coverage.csv")
    grouped_errors = _read_optional(table_dir / "grouped_error_analysis.csv")
    b2_diagnostics = _read_optional(table_dir / "b2_combination_diagnostics.csv")
    fatigue_path = (
        Path(fatigue_frame_path)
        if fatigue_frame_path
        else get_path("data", "processed", "view_model2_sn_fatigue.csv")
    )
    fatigue_frame = _read_optional(fatigue_path)

    domain_readiness = build_alloy_process_domain_readiness(oof, summary)
    domain_readiness_path = table_dir / "alloy_process_domain_readiness.csv"
    domain_readiness.to_csv(
        domain_readiness_path,
        index=False,
        encoding="utf-8-sig",
    )
    shortlist = domain_priority_shortlist(domain_readiness)
    shortlist_path = table_dir / "domain_priority_shortlist.csv"
    shortlist.to_csv(shortlist_path, index=False, encoding="utf-8-sig")
    target_template_path = table_dir / "client_target_template.csv"
    client_target_template().to_csv(
        target_template_path,
        index=False,
        encoding="utf-8-sig",
    )

    static_path = table_dir / "condition_evidence_static.csv"
    fatigue_evidence_path = table_dir / "condition_evidence_fatigue.csv"
    previous_static = _read_optional(static_path)
    previous_fatigue = _read_optional(fatigue_evidence_path)
    static = build_static_condition_evidence(
        oof,
        summary,
        importance,
        config=config,
    )
    fatigue = build_fatigue_condition_evidence(
        oof,
        summary,
        physical_checks,
        fatigue_frame,
        config=config,
    )
    static = attach_step07_traceability(
        static,
        combination_coverage=combination_coverage,
        grouped_errors=grouped_errors,
        b2_diagnostics=b2_diagnostics,
        importance=importance,
    )
    fatigue = attach_step07_traceability(
        fatigue,
        combination_coverage=combination_coverage,
        grouped_errors=grouped_errors,
        b2_diagnostics=b2_diagnostics,
        importance=importance,
    )
    static.to_csv(static_path, index=False, encoding="utf-8-sig")
    fatigue.to_csv(fatigue_evidence_path, index=False, encoding="utf-8-sig")

    outputs: dict[str, Path] = {
        "alloy_process_domain_readiness": domain_readiness_path,
        "domain_priority_shortlist": shortlist_path,
        "client_target_template": target_template_path,
        "condition_evidence_static": static_path,
        "condition_evidence_fatigue": fatigue_evidence_path,
    }
    summary_rows: list[dict[str, Any]] = []
    summary_rows.append(
        {
            "summary_kind": "client_target_status",
            "client_target_status": "not_provided",
            "matrix_scope": "domain_readiness_and_evidence_validation_only",
        }
    )
    if not domain_readiness.empty:
        for status, count in domain_readiness["domain_status"].value_counts().items():
            summary_rows.append(
                {
                    "matrix_type": "alloy_process_domain",
                    "summary_kind": "domain_readiness_count",
                    "domain_status": status,
                    "count": int(count),
                }
            )
    for budget in config.static_budgets:
        plan, plan_summary = select_budget_plan(
            static,
            budget=budget,
            specimens_per_condition=config.static_replicates,
            matrix_type="static",
        )
        path = table_dir / f"selected_static_plan_{budget}.csv"
        plan.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[f"selected_static_plan_{budget}"] = path
        summary_rows.append(plan_summary)
    fatigue_specimens = (
        config.fatigue_stress_levels * config.fatigue_replicates_per_level
    )
    for budget in config.fatigue_budgets:
        plan, plan_summary = select_budget_plan(
            fatigue,
            budget=budget,
            specimens_per_condition=fatigue_specimens,
            matrix_type="fatigue",
        )
        path = table_dir / f"selected_fatigue_plan_{budget}.csv"
        plan.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[f"selected_fatigue_plan_{budget}"] = path
        summary_rows.append(plan_summary)

    for matrix_type, evidence in [("static", static), ("fatigue", fatigue)]:
        if evidence.empty:
            continue
        for action, count in evidence["recommendation_action"].value_counts().items():
            summary_rows.append(
                {
                    "matrix_type": matrix_type,
                    "summary_kind": "classification_count",
                    "recommendation_action": action,
                    "count": int(count),
                }
            )
    matrix_summary_path = table_dir / "matrix_summary.csv"
    pd.DataFrame(summary_rows).to_csv(
        matrix_summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    outputs["matrix_summary"] = matrix_summary_path

    change_log = pd.concat(
        [
            _matrix_change_log(previous_static, static, "static"),
            _matrix_change_log(previous_fatigue, fatigue, "fatigue"),
        ],
        ignore_index=True,
    )
    change_path = table_dir / "matrix_change_log.csv"
    change_log.to_csv(change_path, index=False, encoding="utf-8-sig")
    outputs["matrix_change_log"] = change_path

    metadata_path = table_dir / "actionable_matrix_config.json"
    metadata_path.write_text(
        json.dumps(config.__dict__, indent=2),
        encoding="utf-8",
    )
    outputs["matrix_config"] = metadata_path
    return outputs
