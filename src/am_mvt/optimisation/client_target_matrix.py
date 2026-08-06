from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from am_mvt.modelling.experiment_inference import prediction_rows
from am_mvt.optimisation.domain_readiness import _normalise_domains


STATIC_TARGETS = [
    "uts_MPa",
    "yield_strength_MPa",
    "elongation_percent",
    "youngs_modulus_GPa",
]

PROCESS_COLUMNS = [
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
]

E466_STRESS_LEVELS_MPA = [80.0, 95.0, 110.0, 125.0, 140.0]
E466_REFERENCE_STRESS_MPA = 110.0
INDUSTRY_LIFE_LOWER = 10_000_000.0
INDUSTRY_LIFE_UPPER = 20_000_000.0


def _target_map(targets: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(row["target"]): row.to_dict() for _, row in targets.iterrows()}


def _point_gate(value: float, target: dict[str, Any]) -> bool:
    lower = pd.to_numeric(target.get("lower_bound"), errors="coerce")
    upper = pd.to_numeric(target.get("upper_bound"), errors="coerce")
    return bool(
        np.isfinite(value)
        and (not np.isfinite(lower) or value >= lower)
        and (not np.isfinite(upper) or value <= upper)
    )


def _robust_gate(
    lower_prediction: float, upper_prediction: float, target: dict[str, Any]
) -> bool:
    lower = pd.to_numeric(target.get("lower_bound"), errors="coerce")
    upper = pd.to_numeric(target.get("upper_bound"), errors="coerce")
    return bool(
        np.isfinite(lower_prediction)
        and np.isfinite(upper_prediction)
        and (not np.isfinite(lower) or lower_prediction >= lower)
        and (not np.isfinite(upper) or upper_prediction <= upper)
    )


def _static_candidates(oof: pd.DataFrame, alloy: str, process: str) -> pd.DataFrame:
    normalised = _normalise_domains(oof)
    candidates = normalised.loc[
        normalised["target"].eq("uts_MPa")
        & normalised["route"].eq("ordinary_regression")
        & normalised["mode"].eq("process_only")
        & normalised["alloy"].eq(alloy)
        & normalised["am_process"].eq(process)
    ].copy()
    for column in PROCESS_COLUMNS:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates = candidates.dropna(subset=PROCESS_COLUMNS)
    orientation = candidates["build_orientation"].astype("string").str.strip()
    candidates = candidates.loc[
        orientation.notna() & ~orientation.str.lower().isin(["", "missing", "nan"])
    ].copy()
    candidates = candidates.drop_duplicates(
        subset=["build_orientation", *PROCESS_COLUMNS],
    ).reset_index(drop=True)
    return candidates


def _prediction_wide(
    run_dir: Path,
    candidates: pd.DataFrame,
    targets: list[str],
) -> pd.DataFrame:
    predictions = prediction_rows(run_dir, candidates, "process_only")
    predictions = predictions.loc[
        predictions["route"].eq("ordinary_regression")
        & predictions["target"].isin(targets)
    ].copy()
    pieces = []
    for target in targets:
        selected = predictions.loc[predictions["target"].eq(target)].set_index(
            "input_row"
        )
        pieces.append(
            selected[
                ["prediction", "prediction_lower_90", "prediction_upper_90", "warnings"]
            ].rename(
                columns={
                    "prediction": f"predicted_{target}",
                    "prediction_lower_90": f"{target}_lower_90",
                    "prediction_upper_90": f"{target}_upper_90",
                    "warnings": f"{target}_warnings",
                }
            )
        )
    return pd.concat(pieces, axis=1).reset_index(drop=True)


def _select_representative_static(
    candidates: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    numeric = candidates[PROCESS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    scale = numeric.std(ddof=0).replace(0, 1).fillna(1)
    standardised = (numeric - numeric.mean()) / scale
    remaining = list(candidates.index)
    selected: list[int] = []
    while remaining and len(selected) < count:
        best_index = None
        best_key = None
        for index in remaining:
            row = candidates.loc[index]
            gate_score = float(row["point_target_gate_count"]) / len(STATIC_TARGETS)
            if selected:
                distances = np.sqrt(
                    np.square(
                        standardised.loc[selected].to_numpy()
                        - standardised.loc[index].to_numpy()
                    ).sum(axis=1)
                )
                diversity = float(np.min(distances))
                orientation_bonus = float(
                    str(row["build_orientation"])
                    not in set(
                        candidates.loc[selected, "build_orientation"].astype(str)
                    )
                )
            else:
                diversity = 1.0
                orientation_bonus = 1.0
            key = (
                60 * gate_score + 25 * min(diversity, 2) / 2 + 15 * orientation_bonus,
                float(row["point_target_gate_count"]),
                -int(index),
            )
            if best_key is None or key > best_key:
                best_index = index
                best_key = key
        selected.append(int(best_index))
        remaining.remove(int(best_index))
    return candidates.loc[selected].reset_index(drop=True)


def _industry_plausibility(predicted_cycles: float) -> str:
    if not np.isfinite(predicted_cycles):
        return "not_assessable"
    if predicted_cycles < INDUSTRY_LIFE_LOWER:
        return "below_industry_reference"
    if predicted_cycles <= INDUSTRY_LIFE_UPPER:
        return "within_industry_reference"
    return "above_industry_reference"


def _known_condition_values(series: pd.Series) -> set[str]:
    values = series.astype("string").str.strip().str.lower()
    return set(
        values.loc[
            values.notna()
            & ~values.str.lower().isin(["", "missing", "nan", "none", "unknown"])
        ].astype(str)
    )


def _normalise_requested_surface(value: object) -> str:
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "machine": "machined",
        "turned": "machined",
        "turned/machined": "machined",
        "polish": "polished",
        "as-manufactured": "as-built",
    }
    return aliases.get(text, text)


def _normalise_requested_heat_treatment(value: object) -> str:
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if text in {"stress-relieved", "stress-relief", "t6", "aged", "annealed"}:
        return "heat-treated"
    if text in {"as-manufactured", "as-built", "none", "nht"}:
        return "no-heat-treatment"
    return text


def _representative_fatigue_condition(
    oof: pd.DataFrame,
    alloy: str,
    process: str,
    orientation: str,
    fatigue_target: dict[str, Any],
) -> pd.Series:
    domain = oof.loc[
        oof["target"].eq("log10_fatigue_life_cycles")
        & oof["route"].eq("xgboost_aft")
        & oof["mode"].eq("process_only")
        & oof["alloy"].astype(str).eq(alloy)
        & oof["am_process"].astype(str).eq(process)
    ].drop_duplicates(subset=["record_id"])
    if domain.empty:
        raise ValueError(f"No fatigue evidence rows for {alloy} x {process}.")

    orientation_text = domain["build_orientation"].astype("string").str.strip()
    orientation_match = orientation_text.eq(str(orientation))
    if orientation_match.any():
        domain = domain.loc[orientation_match].copy()

    requested_surface = _normalise_requested_surface(
        fatigue_target["execution_surface_condition"]
    )
    requested_heat = _normalise_requested_heat_treatment(
        fatigue_target["execution_heat_treatment"]
    )
    surface_match = (
        domain["surface_condition"].astype("string").str.lower().eq(requested_surface)
    )
    heat_match = (
        domain["heat_treatment"].astype("string").str.lower().eq(requested_heat)
    )
    if (surface_match & heat_match).any():
        domain = domain.loc[surface_match & heat_match].copy()
    elif surface_match.any():
        domain = domain.loc[surface_match].copy()

    reference = {
        "stress_amplitude_MPa": float(fatigue_target["reference_stress_amplitude_MPa"]),
        "r_ratio": float(fatigue_target["r_ratio"]),
        "frequency_Hz": float(fatigue_target["frequency_Hz"]),
        "test_temperature_C": float(fatigue_target["test_temperature_C"]),
    }
    scales = {
        "stress_amplitude_MPa": 30.0,
        "r_ratio": 0.5,
        "frequency_Hz": 50.0,
        "test_temperature_C": 50.0,
    }
    distance = pd.Series(0.0, index=domain.index)
    missing_penalty = pd.Series(0.0, index=domain.index)
    for column, target_value in reference.items():
        values = pd.to_numeric(domain[column], errors="coerce")
        distance += ((values.fillna(target_value) - target_value) / scales[column]) ** 2
        missing_penalty += values.isna().astype(float)
    domain = domain.assign(
        _representative_distance=np.sqrt(distance) + missing_penalty
    )
    return domain.sort_values(
        ["_representative_distance", "dataset_id", "record_id"]
    ).iloc[0]


def _fatigue_condition_support(
    oof: pd.DataFrame,
    alloy: str,
    process: str,
    surface_condition: str,
    heat_treatment: str,
) -> tuple[bool, str]:
    domain = oof.loc[
        oof["target"].eq("log10_fatigue_life_cycles")
        & oof["route"].eq("xgboost_aft")
        & oof["mode"].eq("process_only")
        & oof["alloy"].astype(str).eq(alloy)
        & oof["am_process"].astype(str).eq(process)
    ]
    if domain.empty:
        return False, "not_assessable_insufficient_domain_evidence"
    known_surface = _known_condition_values(
        domain.get("surface_condition", pd.Series(dtype="string"))
    )
    known_heat = _known_condition_values(
        domain.get("heat_treatment", pd.Series(dtype="string"))
    )
    missing = []
    requested_surface = _normalise_requested_surface(surface_condition)
    requested_heat = _normalise_requested_heat_treatment(heat_treatment)
    if not known_surface or requested_surface not in known_surface:
        missing.append("surface_condition")
    if not known_heat or requested_heat not in known_heat:
        missing.append("heat_treatment")
    if missing:
        return (
            False,
            "not_assessable_surface_heat_treatment_unsupported:" + ",".join(missing),
        )
    return True, ""


def _fatigue_route_agreement(predictions: pd.DataFrame) -> pd.Series:
    route_values = predictions.pivot_table(
        index="input_row",
        columns="route",
        values="predicted_fatigue_life_cycles",
        aggfunc="first",
    )
    available = [
        route
        for route in ("xgboost_aft", "basquin_only", "ordinary_regression")
        if route in route_values
    ]
    if len(available) < 2:
        return pd.Series("insufficient_routes", index=route_values.index)
    logs = np.log10(route_values[available].clip(lower=1.0))
    spread = logs.max(axis=1) - logs.min(axis=1)
    return pd.Series(
        np.where(spread.le(0.50), "agree_within_0.5_log10", "route_disagreement"),
        index=route_values.index,
    )


def _write_fatigue_promotion_check(
    run_dir: Path,
    fatigue: pd.DataFrame,
) -> tuple[Path, str]:
    output_path = run_dir / "tables" / "fatigue_model_promotion.csv"
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        status = "not_assessable_missing_run_configuration"
        pd.DataFrame([{"promotion_status": status}]).to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        return output_path, status
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = [Path(path) for path in config.get("composed_from", [])]
    legacy_summary_path = (
        sources[0] / "tables" / "experiment_summary.csv" if sources else None
    )
    current_summary_path = run_dir / "tables" / "experiment_summary.csv"
    if legacy_summary_path is None or not legacy_summary_path.exists():
        status = "not_assessable_missing_legacy_baseline"
        pd.DataFrame([{"promotion_status": status}]).to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        return output_path, status

    def select_aft(path: Path) -> pd.Series:
        table = pd.read_csv(path, low_memory=False)
        rows = table.loc[
            table["route"].eq("xgboost_aft")
            & table["mode"].eq("process_only")
            & table["fold"].astype(str).eq("summary")
        ]
        if rows.empty:
            raise ValueError(f"Missing process-only AFT summary: {path}")
        return rows.iloc[-1]

    legacy = select_aft(legacy_summary_path)
    current = select_aft(current_summary_path)
    legacy_loss = float(legacy["cv_aft_nloglik_mean"])
    current_loss = float(current["cv_aft_nloglik_mean"])
    legacy_c = float(legacy["cv_harrell_c_index_mean"])
    current_c = float(current["cv_harrell_c_index_mean"])
    calibration_path = run_dir / "tables" / "fatigue_threshold_calibration.csv"
    calibration = pd.read_csv(calibration_path, low_memory=False)
    calibrated_rows = calibration.loc[calibration["calibration_available"].eq(True)]
    calibration_improved = bool(
        (
            pd.to_numeric(calibrated_rows["calibrated_brier"], errors="coerce")
            < pd.to_numeric(calibrated_rows["raw_brier"], errors="coerce")
        ).any()
    )
    gates = {
        "gate_grouped_oof_aft_loss_improved": current_loss < legacy_loss,
        "gate_c_index_drop_within_0_02": current_c >= legacy_c - 0.02,
        "gate_threshold_calibration_improved": calibration_improved,
        "gate_matrix_monotonicity_100_percent": bool(
            fatigue["monotonicity"].eq("passed").all()
        ),
    }
    promoted = all(gates.values())
    status = "promoted_protocol_aware_aft" if promoted else "candidate_not_promoted"
    pd.DataFrame(
        [
            {
                "promotion_status": status,
                "legacy_cv_aft_nloglik": legacy_loss,
                "new_cv_aft_nloglik": current_loss,
                "legacy_cv_harrell_c_index": legacy_c,
                "new_cv_harrell_c_index": current_c,
                **gates,
                "comparison_scope_note": (
                    "Legacy baseline used mixed frequency regimes; the new score "
                    "also reflects protocol correction and is not a pure algorithm "
                    "ablation."
                ),
            }
        ]
    ).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path, status


def generate_client_target_matrix(
    run_dir: str | Path,
    target_path: str | Path,
    *,
    static_budget: int = 24,
    fatigue_budget: int = 30,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    table_dir = run_dir / "tables"
    targets = pd.read_csv(target_path)
    cases = targets["case_id"].dropna().unique()
    if len(cases) != 1:
        raise ValueError("A client target file must contain exactly one case_id.")
    alloy = str(targets.iloc[0]["alloy"])
    process = str(targets.iloc[0]["am_process"])
    target_by_name = _target_map(targets)
    oof = pd.read_csv(table_dir / "oof_predictions.csv", low_memory=False)

    static = _static_candidates(oof, alloy, process)
    if static.empty:
        raise ValueError(f"No observed process candidates for {alloy} x {process}.")
    predictions = _prediction_wide(run_dir, static, STATIC_TARGETS)
    static = pd.concat([static.reset_index(drop=True), predictions], axis=1)
    static["point_target_gate_count"] = 0
    static["robust_target_gate_count"] = 0
    for target in STATIC_TARGETS:
        point_column = f"gate_point_{target}"
        robust_column = f"gate_robust90_{target}"
        static[point_column] = static[f"predicted_{target}"].map(
            lambda value: _point_gate(float(value), target_by_name[target])
        )
        static[robust_column] = static.apply(
            lambda row: _robust_gate(
                float(row[f"{target}_lower_90"]),
                float(row[f"{target}_upper_90"]),
                target_by_name[target],
            ),
            axis=1,
        )
        static["point_target_gate_count"] += static[point_column].astype(int)
        static["robust_target_gate_count"] += static[robust_column].astype(int)

    selected_static = _select_representative_static(
        static,
        static_budget // 3,
    )
    selected_static["condition_id"] = [
        f"{cases[0]}::static::{index + 1:02d}" for index in range(len(selected_static))
    ]
    selected_static["execution_surface_condition"] = targets.iloc[0][
        "execution_surface_condition"
    ]
    selected_static["execution_heat_treatment"] = targets.iloc[0][
        "execution_heat_treatment"
    ]
    for target in STATIC_TARGETS:
        selected_static[f"client_{target}_lower_bound"] = target_by_name[target][
            "lower_bound"
        ]
        selected_static[f"client_{target}_upper_bound"] = target_by_name[target][
            "upper_bound"
        ]
    selected_static["planned_replicates"] = 3
    selected_static["planned_specimens"] = 3
    selected_static["selection_role"] = np.where(
        selected_static["point_target_gate_count"].eq(len(STATIC_TARGETS)),
        "target_match_validation",
        "target_boundary_validation",
    )
    selected_static["matrix_purpose"] = "client_target_pilot_matrix"
    selected_static[
        "evidence_warning"
    ] = "surface_and_heat_treatment_are_client_fixed_but_sparse_in_training_data"

    fatigue_target = target_by_name["fatigue_life_cycles"]
    reference_stress = E466_REFERENCE_STRESS_MPA
    stress_levels = np.asarray(E466_STRESS_LEVELS_MPA, dtype=float)
    block_orientations = ["0", "90"][: max(1, fatigue_budget // 15)]
    fatigue_rows = []
    for block_index, orientation_value in enumerate(block_orientations, start=1):
        representative = _representative_fatigue_condition(
            oof,
            alloy,
            process,
            orientation_value,
            fatigue_target,
        )
        base = representative.to_dict()
        base["surface_condition"] = _normalise_requested_surface(
            fatigue_target["execution_surface_condition"]
        )
        base["heat_treatment"] = _normalise_requested_heat_treatment(
            fatigue_target["execution_heat_treatment"]
        )
        if base["heat_treatment"] == "heat-treated":
            base["material_state"] = "heat-treated"
        elif base["heat_treatment"] == "no-heat-treatment":
            base["material_state"] = (
                "as-manufactured"
                if base["surface_condition"] == "as-built"
                else "surface-processed-no-heat-treatment"
            )
        for stress in stress_levels:
            fatigue_rows.append(
                {
                    **base,
                    "build_orientation": orientation_value,
                    "stress_amplitude_MPa": float(stress),
                    "r_ratio": float(fatigue_target["r_ratio"]),
                    "test_temperature_C": float(fatigue_target["test_temperature_C"]),
                    "frequency_Hz": float(fatigue_target["frequency_Hz"]),
                    "stress_definition": "amplitude",
                    "fatigue_protocol": "e466_conventional",
                    "representative_fatigue_dataset_id": representative.get(
                        "dataset_id"
                    ),
                    "representative_fatigue_record_id": representative.get(
                        "record_id"
                    ),
                    "fatigue_block_id": f"{cases[0]}::fatigue::{block_index:02d}",
                }
            )
    fatigue = pd.DataFrame(fatigue_rows)
    all_fatigue_predictions = prediction_rows(run_dir, fatigue, "process_only")
    all_fatigue_predictions = all_fatigue_predictions.loc[
        all_fatigue_predictions["target"].eq("log10_fatigue_life_cycles")
    ].copy()
    fatigue_predictions = all_fatigue_predictions.loc[
        all_fatigue_predictions["route"].eq("xgboost_aft")
    ].sort_values("input_row")
    if len(fatigue_predictions) != len(fatigue):
        raise ValueError("The run does not contain a complete E466 XGBoost-AFT route.")
    fatigue["median_fatigue_life_cycles"] = fatigue_predictions[
        "predicted_fatigue_life_cycles"
    ].to_numpy()
    fatigue["life_quantile_10_cycles"] = fatigue_predictions[
        "prediction_lower_90_cycles"
    ].to_numpy()
    fatigue["life_quantile_90_cycles"] = fatigue_predictions[
        "prediction_upper_90_cycles"
    ].to_numpy()
    for column in [
        "life_quantile_20_cycles",
        "life_quantile_50_cycles",
        "life_quantile_80_cycles",
        "probability_reach_10m",
        "probability_reach_20m",
        "calibration_level_10m",
        "calibration_level_20m",
        "probability_order_adjustment",
        "fatigue_model_level",
        "fatigue_model_domain_key",
        "aft_distribution",
        "aft_scale",
    ]:
        fatigue[column] = fatigue_predictions[column].to_numpy()
    fatigue["model_warnings"] = fatigue_predictions["warnings"].to_numpy()
    fatigue["reference_target_applies"] = np.isclose(
        fatigue["stress_amplitude_MPa"],
        reference_stress,
    )
    fatigue["client_reference_stress_amplitude_MPa"] = reference_stress
    fatigue["external_reference_lower_cycles"] = INDUSTRY_LIFE_LOWER
    fatigue["external_reference_upper_cycles"] = INDUSTRY_LIFE_UPPER
    fatigue["execution_surface_condition"] = fatigue_target[
        "execution_surface_condition"
    ]
    fatigue["execution_heat_treatment"] = fatigue_target["execution_heat_treatment"]
    fatigue["replicates_per_level"] = 3
    fatigue["planned_specimens"] = 3
    frequency = pd.to_numeric(fatigue["frequency_Hz"], errors="coerce")
    fatigue["machine_hours_to_10m_per_specimen"] = (
        INDUSTRY_LIFE_LOWER / frequency / 3600.0
    )
    fatigue["machine_hours_to_20m_per_specimen"] = (
        INDUSTRY_LIFE_UPPER / frequency / 3600.0
    )
    fatigue["planned_machine_hours_to_20m"] = (
        fatigue["machine_hours_to_20m_per_specimen"] * fatigue["planned_specimens"]
    )
    fatigue["matrix_purpose"] = "astm_e466_style_hcf_validation_matrix"
    fatigue["prediction_interval"] = "q10_to_q90_survival_distribution"
    support_ok, support_blocker = _fatigue_condition_support(
        oof,
        alloy,
        process,
        str(fatigue_target["execution_surface_condition"]),
        str(fatigue_target["execution_heat_treatment"]),
    )
    fatigue["condition_fields_supported"] = support_ok
    fatigue["route_agreement"] = (
        _fatigue_route_agreement(all_fatigue_predictions)
        .reindex(range(len(fatigue)))
        .fillna("insufficient_routes")
        .to_numpy()
    )
    fatigue["monotonicity"] = "not_checked"
    for _, indices in fatigue.groupby("fatigue_block_id").groups.items():
        ordered = fatigue.loc[indices].sort_values("stress_amplitude_MPa")
        monotonic = all(
            np.all(np.diff(ordered[column].to_numpy(dtype=float)) <= 1e-12)
            for column in [
                "median_fatigue_life_cycles",
                "probability_reach_10m",
                "probability_reach_20m",
            ]
        )
        fatigue.loc[indices, "monotonicity"] = "passed" if monotonic else "failed"
    fatigue["external_plausibility_status"] = fatigue["median_fatigue_life_cycles"].map(
        _industry_plausibility
    )
    fatigue["decision_blocker"] = support_blocker
    fatigue.loc[
        fatigue["fatigue_model_level"].eq("not_assessable"),
        "decision_blocker",
    ] = "not_assessable_insufficient_domain_evidence"
    fatigue.loc[
        fatigue["monotonicity"].eq("failed"),
        "decision_blocker",
    ] = fatigue.loc[
        fatigue["monotonicity"].eq("failed"),
        "decision_blocker",
    ].map(
        lambda value: ";".join(filter(None, [value, "aft_monotonicity_failed"]))
    )
    for mask, reason in [
        (fatigue["route_agreement"].eq("route_disagreement"), "route_disagreement"),
        (
            fatigue["external_plausibility_status"].eq("below_industry_reference"),
            "prediction_below_industry_reference",
        ),
    ]:
        fatigue.loc[mask, "decision_blocker"] = fatigue.loc[
            mask,
            "decision_blocker",
        ].map(lambda value: ";".join(filter(None, [value, reason])))
    fatigue["decision_status"] = np.where(
        fatigue["decision_blocker"].astype(str).eq(""),
        "retain_validation",
        "not_assessable",
    )

    case_prefix = str(cases[0])
    static_path = table_dir / f"client_case_{case_prefix}_static_24.csv"
    fatigue_path = table_dir / f"client_case_{case_prefix}_fatigue_30.csv"
    summary_path = table_dir / f"client_case_{case_prefix}_summary.csv"
    benchmark_path = table_dir / "fatigue_external_benchmark_check.csv"
    selected_static.to_csv(static_path, index=False, encoding="utf-8-sig")
    promotion_path, promotion_status = _write_fatigue_promotion_check(
        run_dir,
        fatigue,
    )
    fatigue["aft_route_promotion_status"] = promotion_status
    fatigue.to_csv(fatigue_path, index=False, encoding="utf-8-sig")
    reference_rows = fatigue.loc[fatigue["reference_target_applies"]].copy()
    benchmark = reference_rows[
        [
            "build_orientation",
            "fatigue_protocol",
            "stress_definition",
            "stress_amplitude_MPa",
            "r_ratio",
            "frequency_Hz",
            "median_fatigue_life_cycles",
            "probability_reach_10m",
            "probability_reach_20m",
            "life_quantile_10_cycles",
            "life_quantile_90_cycles",
            "external_plausibility_status",
            "fatigue_model_level",
            "decision_blocker",
        ]
    ].copy()
    benchmark.insert(0, "benchmark_id", "eos_like_alsi10mg_lpbf")
    benchmark["alloy"] = "AlSi10Mg"
    benchmark["am_process"] = "L-PBF"
    benchmark["benchmark_surface_condition"] = "turned/machined"
    benchmark["benchmark_heat_treatment"] = "as-manufactured"
    benchmark["benchmark_cycles"] = INDUSTRY_LIFE_UPPER
    benchmark["benchmark_role"] = "external_audit_only_not_used_for_training"
    benchmark["condition_match_note"] = (
        "EOS-like condition is separate from the client machined and "
        "stress-relieved execution condition"
    )
    benchmark.to_csv(benchmark_path, index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(
        [
            {
                "case_id": case_prefix,
                "alloy": alloy,
                "am_process": process,
                "domain_status": "pilot_ready",
                "static_conditions": len(selected_static),
                "static_specimens": int(selected_static["planned_specimens"].sum()),
                "fatigue_blocks": fatigue["fatigue_block_id"].nunique(),
                "fatigue_stress_levels_per_block": len(stress_levels),
                "fatigue_specimens": int(fatigue["planned_specimens"].sum()),
                "client_target_status": "example_target_provided",
                "matrix_status": "pilot_matrix_requires_physical_confirmation",
                "primary_blocker": support_blocker,
                "fatigue_protocol": "e466_conventional",
                "fatigue_reference_stress_amplitude_MPa": reference_stress,
                "fatigue_external_reference_cycles": "10000000-20000000",
                "fatigue_aft_route_status": promotion_status,
                "fatigue_total_machine_hours_to_20m": float(
                    fatigue["planned_machine_hours_to_20m"].sum()
                ),
            }
        ]
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {
        "static_matrix": static_path,
        "fatigue_matrix": fatigue_path,
        "case_summary": summary_path,
        "fatigue_external_benchmark": benchmark_path,
        "fatigue_model_promotion": promotion_path,
    }
