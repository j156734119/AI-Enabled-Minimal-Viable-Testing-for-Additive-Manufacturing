from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from am_mvt.cleaning.normalise_alloys import apply_alloy_normalisation


TARGET_SPECS = {
    "uts": ("uts_MPa", 30, 15, 0.35),
    "yield_strength": ("yield_strength_MPa", 30, 15, 0.35),
    "elongation": ("elongation_percent", 30, 15, 0.35),
    "youngs_modulus": ("youngs_modulus_GPa", 30, 15, 0.35),
    "fatigue_life": ("log10_fatigue_life_cycles", 50, 30, 0.20),
}

DOMAIN_FEATURES = [
    "build_orientation",
    "surface_condition",
    "heat_treatment",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
]

FATIGUE_FEATURES = [
    "stress_amplitude_MPa",
    "r_ratio",
    "frequency_Hz",
    "test_temperature_C",
    "runout",
]


@dataclass(frozen=True)
class DomainReadinessConfig:
    min_sources: int = 2
    min_groups: int = 3
    ready_feature_coverage: float = 0.70
    pilot_feature_coverage: float = 0.40
    shortlist_size: int = 12


def normalise_process_domain(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() == "missing":
        return pd.NA
    compact = "".join(character for character in text.lower() if character.isalnum())
    aliases = {
        "lpbf": "L-PBF",
        "lbpbf": "L-PBF",
        "pbflb": "L-PBF",
        "pbflbm": "L-PBF",
        "laserpowderbedfusion": "L-PBF",
        "laserpowderbedfusionlpbf": "L-PBF",
        "selectivelasermelting": "L-PBF",
        "selectivelasermeltingslm": "L-PBF",
        "slm": "L-PBF",
        "epbf": "E-PBF",
        "ebpbf": "E-PBF",
        "electronbeammelting": "E-PBF",
        "waam": "WAAM",
        "wirearcadditivemanufacturingwaam": "WAAM",
    }
    return aliases.get(compact, text)


def _present(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.notna()
    text = series.astype("string").str.strip().str.lower()
    return series.notna() & ~text.isin(["", "missing", "nan", "none"])


def _coverage(frame: pd.DataFrame, columns: list[str]) -> float:
    available = [column for column in columns if column in frame]
    if not available or frame.empty:
        return 0.0
    return float(np.mean([_present(frame[column]).mean() for column in available]))


def _distinct(frame: pd.DataFrame, candidates: list[str]) -> int:
    for column in candidates:
        if column in frame:
            return int(frame[column].dropna().astype("string").nunique())
    return 0


def _r2(frame: pd.DataFrame) -> float:
    valid = frame[["y_true", "y_pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) < 5:
        return np.nan
    residual = float(np.square(valid["y_true"] - valid["y_pred"]).sum())
    total = float(np.square(valid["y_true"] - valid["y_true"].mean()).sum())
    return 1 - residual / total if total > 0 else np.nan


def _selected_global_r2(summary: pd.DataFrame, target: str) -> float:
    if summary.empty:
        return np.nan
    selected = summary.loc[
        summary["target"].eq(target)
        & summary["route"].eq("ordinary_regression")
        & summary.get("mode", "process_only").eq("process_only")
        & summary["selected"].astype("string").str.lower().eq("true")
    ]
    if selected.empty:
        return np.nan
    return float(pd.to_numeric(selected.iloc[0].get("oof_r2"), errors="coerce"))


def _normalise_domains(frame: pd.DataFrame) -> pd.DataFrame:
    result = apply_alloy_normalisation(frame)
    if "am_process" not in result:
        result["am_process"] = pd.NA
    result["am_process_original"] = result["am_process"]
    result["am_process"] = result["am_process"].map(normalise_process_domain)
    result = result.loc[result["alloy"].notna() & result["am_process"].notna()].copy()
    result["domain_id"] = (
        result["alloy"].astype("string")
        + "::"
        + result["am_process"].astype("string")
    )
    return result


def build_alloy_process_domain_readiness(
    oof: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    config: DomainReadinessConfig | None = None,
) -> pd.DataFrame:
    config = config or DomainReadinessConfig()
    ordinary = oof.loc[
        oof["target"].isin([spec[0] for spec in TARGET_SPECS.values()])
        & oof["route"].eq("ordinary_regression")
        & oof.get("mode", "process_only").eq("process_only")
    ].copy()
    ordinary = _normalise_domains(ordinary)
    if ordinary.empty:
        return pd.DataFrame()

    global_r2 = {
        target: _selected_global_r2(summary, target)
        for target, *_ in TARGET_SPECS.values()
    }
    rows: list[dict[str, Any]] = []
    for (alloy, process), domain in ordinary.groupby(
        ["alloy", "am_process"],
        dropna=False,
    ):
        row: dict[str, Any] = {
            "domain_id": f"{alloy}::{process}",
            "alloy": alloy,
            "alloy_family": ";".join(
                sorted(domain.get("alloy_family", pd.Series(dtype="string")).dropna().astype(str).unique())
            ),
            "am_process": process,
            "process_variants": ";".join(
                sorted(domain["am_process_original"].dropna().astype(str).unique())
            ),
            "condition_feature_coverage": _coverage(domain, DOMAIN_FEATURES),
        }
        fatigue_domain = domain.loc[
            domain["target"].eq(TARGET_SPECS["fatigue_life"][0])
        ]
        row["fatigue_loading_coverage"] = _coverage(
            fatigue_domain,
            FATIGUE_FEATURES,
        )
        target_tiers: list[str] = []
        target_model_gates: list[bool] = []
        present_targets = 0
        ready_targets = 0
        pilot_targets = 0
        row_scores: list[float] = []
        source_scores: list[float] = []
        for prefix, (target, ready_rows, pilot_rows, min_r2) in TARGET_SPECS.items():
            target_frame = domain.loc[domain["target"].eq(target)]
            n_rows = len(target_frame)
            sources = _distinct(target_frame, ["source_id"])
            groups = _distinct(
                target_frame,
                ["evaluation_group_id", "modelling_group_id", "source_id"],
            )
            local_r2 = _r2(target_frame)
            model_r2 = global_r2[target]
            model_gate = bool(np.isfinite(model_r2) and model_r2 >= min_r2)
            if n_rows >= ready_rows and sources >= config.min_sources and groups >= config.min_groups:
                tier = "ready"
                ready_targets += 1
            elif n_rows >= pilot_rows and groups >= 2:
                tier = "pilot"
                pilot_targets += 1
            elif n_rows > 0:
                tier = "sparse"
            else:
                tier = "absent"
            if n_rows > 0:
                present_targets += 1
            target_tiers.append(tier)
            target_model_gates.append(model_gate)
            row_scores.append(min(n_rows / ready_rows, 1.0))
            source_scores.append(min(sources / config.min_sources, 1.0))
            row.update(
                {
                    f"{prefix}_rows": n_rows,
                    f"{prefix}_sources": sources,
                    f"{prefix}_groups": groups,
                    f"{prefix}_tier": tier,
                    f"{prefix}_global_oof_r2": model_r2,
                    f"{prefix}_local_oof_r2": local_r2,
                    f"{prefix}_model_gate": model_gate,
                }
            )

        all_ready = ready_targets == len(TARGET_SPECS)
        all_models = all(target_model_gates)
        feature_ready = (
            row["condition_feature_coverage"] >= config.ready_feature_coverage
            and row["fatigue_loading_coverage"] >= config.ready_feature_coverage
        )
        if all_ready and all_models and feature_ready:
            status = "ready_for_targeted_matrix"
        elif (
            present_targets == len(TARGET_SPECS)
            and ready_targets + pilot_targets >= 3
            and row["condition_feature_coverage"] >= config.pilot_feature_coverage
        ):
            status = "pilot_ready"
        elif present_targets >= 2:
            status = "collect_more_data"
        else:
            status = "not_supported"

        blockers = []
        if present_targets < len(TARGET_SPECS):
            blockers.append("missing_target_coverage")
        if not all_models:
            blockers.append("model_quality")
        if row["condition_feature_coverage"] < config.ready_feature_coverage:
            blockers.append("condition_feature_coverage")
        if row["fatigue_loading_coverage"] < config.ready_feature_coverage:
            blockers.append("fatigue_loading_coverage")
        if not all(tier == "ready" for tier in target_tiers):
            blockers.append("row_source_group_evidence")

        model_score = float(np.mean(target_model_gates))
        feature_score = float(
            np.mean(
                [
                    row["condition_feature_coverage"],
                    row["fatigue_loading_coverage"],
                ]
            )
        )
        readiness_score = 100 * (
            0.30 * float(np.mean(row_scores))
            + 0.25 * float(np.mean(source_scores))
            + 0.25 * feature_score
            + 0.20 * model_score
        )
        row.update(
            {
                "targets_present": present_targets,
                "targets_ready": ready_targets,
                "targets_pilot_or_better": ready_targets + pilot_targets,
                "domain_status": status,
                "decision_blocker": ";".join(blockers),
                "readiness_score": readiness_score,
                "client_target_status": "not_provided",
            }
        )
        rows.append(row)

    rank = {
        "ready_for_targeted_matrix": 0,
        "pilot_ready": 1,
        "collect_more_data": 2,
        "not_supported": 3,
    }
    result = pd.DataFrame(rows)
    result["_status_rank"] = result["domain_status"].map(rank).fillna(4)
    result = result.sort_values(
        ["_status_rank", "readiness_score", "domain_id"],
        ascending=[True, False, True],
    ).drop(columns="_status_rank")
    return result.reset_index(drop=True)


def domain_priority_shortlist(
    readiness: pd.DataFrame,
    *,
    size: int = 12,
) -> pd.DataFrame:
    if readiness.empty:
        return readiness.copy()
    eligible = readiness.loc[~readiness["domain_status"].eq("not_supported")].copy()
    return eligible.head(size).reset_index(drop=True)


def client_target_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": target,
                "display_name": display,
                "unit": unit,
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "required": True,
                "target_status": "awaiting_client_input",
            }
            for target, display, unit in [
                ("uts_MPa", "UTS", "MPa"),
                ("yield_strength_MPa", "Yield strength", "MPa"),
                ("elongation_percent", "Elongation", "%"),
                ("youngs_modulus_GPa", "Young's modulus", "GPa"),
                ("fatigue_life_cycles", "Fatigue life", "cycles"),
            ]
        ]
    )
