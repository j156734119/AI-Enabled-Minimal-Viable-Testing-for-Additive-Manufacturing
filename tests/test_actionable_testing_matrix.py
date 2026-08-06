from __future__ import annotations

import numpy as np
import pandas as pd

from am_mvt.optimisation.actionable_matrix import (
    ActionableMatrixConfig,
    _clean_group_values,
    build_fatigue_condition_evidence,
    build_static_condition_evidence,
    generate_actionable_testing_matrix,
    select_budget_plan,
)
from am_mvt.optimisation.domain_readiness import (
    build_alloy_process_domain_readiness,
    normalise_process_domain,
)


def test_numeric_build_orientation_has_one_canonical_group_value():
    frame = pd.DataFrame({"build_orientation": [0, 0.0, "0 deg", "0°", pd.NA]})

    result = _clean_group_values(frame, ["build_orientation"])

    assert result["build_orientation"].tolist() == [
        "0",
        "0",
        "0",
        "0",
        "missing",
    ]


def selected_summary(target: str, oof_r2: float) -> dict[str, object]:
    return {
        "target": target,
        "route": "ordinary_regression",
        "mode": "process_only",
        "candidate": "random_forest",
        "selected": True,
        "oof_r2": oof_r2,
    }


def static_oof() -> pd.DataFrame:
    rows = []
    for target, base in [("uts_MPa", 900.0), ("yield_strength_MPa", 800.0)]:
        for index in range(120):
            condition = index % 4
            actual = base + index
            rows.append(
                {
                    "target": target,
                    "mode": "process_only",
                    "route": "ordinary_regression",
                    "source_id": f"source_{index % 3}",
                    "evaluation_group_id": f"group_{index % 6}",
                    "record_id": f"{target}_{index}",
                    "alloy": "Ti-6Al-4V",
                    "alloy_family": "titanium",
                    "am_process": "LPBF",
                    "build_orientation": f"O{condition}",
                    "surface_condition": "machined",
                    "heat_treatment": "stress_relief",
                    "laser_power_W": 180 + (index % 9) * 10,
                    "scan_speed_mm_s": 900 + (index % 7) * 20,
                    "layer_thickness_um": 30,
                    "y_true": actual,
                    "y_pred": actual - 2,
                    "abs_error": 2.0,
                    "interval_hit_90": True,
                    "conformal_q90": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_static_matrix_uses_observed_medoid_and_counts_tensile_once():
    oof = static_oof()
    summary = pd.DataFrame(
        [
            selected_summary("uts_MPa", 0.70),
            selected_summary("yield_strength_MPa", 0.65),
        ]
    )
    importance = pd.DataFrame(
        [
            {
                "target": target,
                "evidence_stability": "supported_by_both",
            }
            for target in ["uts_MPa", "yield_strength_MPa"]
        ]
    )
    evidence = build_static_condition_evidence(oof, summary, importance)

    assert set(evidence["evidence_tier"]) == {"green"}
    assert set(evidence["recommendation_action"]) == {"candidate_for_reduction"}
    assert evidence["test_type"].eq("tensile").all()
    assert evidence["planned_replicates"].eq(3).all()
    observed_powers = set(oof["laser_power_W"])
    assert set(evidence["laser_power_W"]) <= observed_powers


def test_domain_readiness_is_built_at_alloy_process_level_for_five_targets():
    rows = []
    targets = [
        "uts_MPa",
        "yield_strength_MPa",
        "elongation_percent",
        "youngs_modulus_GPa",
        "log10_fatigue_life_cycles",
    ]
    for target_index, target in enumerate(targets):
        for index in range(60):
            actual = 100.0 + target_index * 10 + index
            rows.append(
                {
                    "target": target,
                    "mode": "process_only",
                    "route": "ordinary_regression",
                    "selected": True,
                    "source_id": f"source_{index % 3}",
                    "evaluation_group_id": f"group_{index % 6}",
                    "record_id": f"{target}_{index}",
                    "alloy": "Ti64",
                    "alloy_family": "Ti alloy",
                    "am_process": "Laser Powder Bed Fusion",
                    "build_orientation": "vertical",
                    "surface_condition": "machined",
                    "heat_treatment": "stress relieved",
                    "laser_power_W": 200,
                    "scan_speed_mm_s": 1000,
                    "hatch_spacing_um": 100,
                    "layer_thickness_um": 30,
                    "stress_amplitude_MPa": 400,
                    "r_ratio": 0.1,
                    "frequency_Hz": 20,
                    "test_temperature_C": 25,
                    "runout": False,
                    "y_true": actual,
                    "y_pred": actual - 1,
                }
            )
    summary = pd.DataFrame(
        [selected_summary(target, 0.70) for target in targets]
    )

    readiness = build_alloy_process_domain_readiness(
        pd.DataFrame(rows),
        summary,
    )

    assert len(readiness) == 1
    assert readiness.iloc[0]["alloy"] == "Ti-6Al-4V"
    assert readiness.iloc[0]["am_process"] == "L-PBF"
    assert readiness.iloc[0]["targets_present"] == 5
    assert readiness.iloc[0]["domain_status"] == "ready_for_targeted_matrix"
    assert normalise_process_domain("SLM") == "L-PBF"


def test_static_condition_exposes_missing_primary_target_evidence():
    oof = static_oof()
    oof = oof.loc[
        ~(
            oof["target"].eq("yield_strength_MPa")
            & oof["build_orientation"].eq("O0")
        )
    ]
    summary = pd.DataFrame(
        [
            selected_summary("uts_MPa", 0.70),
            selected_summary("yield_strength_MPa", 0.65),
        ]
    )

    evidence = build_static_condition_evidence(oof, summary)
    uts_only = evidence.loc[evidence["build_orientation"].eq("O0")].iloc[0]

    assert not uts_only["gate_primary_target_coverage"]
    assert uts_only["observed_primary_targets"] == "uts_MPa"
    assert uts_only["missing_primary_targets"] == "yield_strength_MPa"
    assert "gate_primary_target_coverage" in uts_only["decision_blocker"]


def fatigue_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = []
    oof_rows = []
    stresses = [250, 300, 350, 400, 450, 500]
    for index in range(180):
        stress = stresses[index % len(stresses)]
        orientation = f"O{index // 45}"
        actual = 7.0 - stress / 200.0
        base = {
            "source_id": f"source_{index % 3}",
            "evaluation_group_id": f"group_{index % 6}",
            "record_id": f"fatigue_{index}",
            "alloy": "Ti-6Al-4V",
            "alloy_family": "titanium",
            "am_process": "LPBF",
            "build_orientation": orientation,
            "surface_condition": "machined",
            "heat_treatment": "stress_relief",
            "r_ratio": 0.1,
            "test_temperature_C": 20,
            "frequency_Hz": 20,
            "stress_amplitude_MPa": stress,
            "runout": index % 5 == 0,
            "log10_fatigue_life_cycles": actual,
        }
        source_rows.append(base)
        for route, offset in [
            ("ordinary_regression", 0.02),
            ("basquin_only", 0.01),
        ]:
            oof_rows.append(
                {
                    **base,
                    "target": "log10_fatigue_life_cycles",
                    "mode": "process_only",
                    "route": route,
                    "y_true": actual,
                    "y_pred": actual - offset,
                    "abs_error": offset,
                    "interval_hit_90": True,
                    "conformal_q90": 0.20,
                }
            )
    return pd.DataFrame(oof_rows), pd.DataFrame(source_rows)


def test_weak_fatigue_model_generates_validation_blocks_not_reduction():
    oof, source = fatigue_frames()
    summary = pd.DataFrame(
        [
            selected_summary("log10_fatigue_life_cycles", 0.265),
            {
                "target": "log10_fatigue_life_cycles",
                "route": "basquin_only",
                "mode": "process_only",
                "candidate": "basquin_only",
                "selected": False,
                "cv_r2_mean": 0.30,
            },
        ]
    )
    checks = pd.DataFrame(
        [
            {
                "route": "basquin_only",
                "stress_scan_monotonic_nonincreasing": True,
            }
        ]
    )
    evidence = build_fatigue_condition_evidence(
        oof,
        summary,
        checks,
        source,
    )

    assert len(evidence) == 4
    assert evidence["recommendation_action"].eq("pilot_validation").all()
    assert evidence["planned_specimens"].eq(15).all()
    assert evidence["stress_levels_MPa"].str.split(";").map(len).eq(5).all()
    assert evidence["gate_route_agreement"].all()
    assert not evidence["gate_model_quality"].any()


def test_fatigue_pilot_keeps_missing_conditions_as_explicit_blockers():
    oof, source = fatigue_frames()
    source["surface_condition"] = pd.NA
    source["heat_treatment"] = pd.NA
    oof["surface_condition"] = pd.NA
    oof["heat_treatment"] = pd.NA
    summary = pd.DataFrame(
        [selected_summary("log10_fatigue_life_cycles", 0.265)]
    )

    evidence = build_fatigue_condition_evidence(oof, summary, fatigue_frame=source)

    assert evidence["recommendation_action"].eq("pilot_validation").all()
    assert evidence["condition_completion_required"].all()
    assert evidence["unresolved_condition_fields"].eq(
        "surface_condition;heat_treatment"
    ).all()
    assert evidence["decision_blocker"].str.contains("gate_known_condition").all()


def test_four_observed_fatigue_levels_expand_to_five_planned_levels():
    oof, source = fatigue_frames()
    source = source.loc[source["stress_amplitude_MPa"].isin([250, 300, 400, 500])]
    oof = oof.loc[oof["record_id"].isin(source["record_id"])]
    summary = pd.DataFrame(
        [selected_summary("log10_fatigue_life_cycles", 0.265)]
    )

    evidence = build_fatigue_condition_evidence(oof, summary, fatigue_frame=source)

    planned_level_counts = evidence["stress_levels_MPa"].str.split(";").map(len)
    assert planned_level_counts.eq(5).all()
    assert evidence["planned_specimens"].eq(15).all()


def test_budget_plans_use_specimen_counts_and_complete_sn_blocks():
    evidence = pd.DataFrame(
        [
            {
                "condition_id": f"fatigue::{index}",
                "alloy": "Ti-6Al-4V",
                "am_process": "LPBF",
                "build_orientation": f"O{index}",
                "surface_condition": "machined",
                "r_ratio": "0.1",
                "evidence_tier": "green",
                "recommendation_action": "pilot_validation",
                "information_value_score": float(100 - index),
            }
            for index in range(4)
        ]
    )
    for budget, expected_blocks in [(30, 2), (45, 3), (60, 4)]:
        plan, summary = select_budget_plan(
            evidence,
            budget=budget,
            specimens_per_condition=15,
            matrix_type="fatigue",
        )
        assert len(plan) == expected_blocks
        assert summary["selected_specimens"] == budget


def test_static_budget_does_not_charge_four_properties_per_specimen():
    evidence = pd.DataFrame(
        [
            {
                "condition_id": f"static::{index}",
                "alloy": "316L",
                "am_process": "LPBF",
                "build_orientation": f"O{index}",
                "surface_condition": "machined",
                "evidence_tier": "green",
                "recommendation_action": "retain_validation",
                "information_value_score": float(100 - index),
            }
            for index in range(20)
        ]
    )
    plan, summary = select_budget_plan(
        evidence,
        budget=24,
        specimens_per_condition=3,
        matrix_type="static",
    )
    assert len(plan) == 8
    assert summary["selected_specimens"] == 24
    assert np.all(plan["planned_specimens"].eq(3))


def test_synthetic_run_writes_static_and_fatigue_artifacts(tmp_path):
    run_dir = tmp_path / "synthetic_run"
    tables = run_dir / "tables"
    tables.mkdir(parents=True)
    fatigue_oof, fatigue_source = fatigue_frames()
    oof = pd.concat([static_oof(), fatigue_oof], ignore_index=True)
    summary = pd.DataFrame(
        [
            selected_summary("uts_MPa", 0.70),
            selected_summary("yield_strength_MPa", 0.65),
            selected_summary("log10_fatigue_life_cycles", 0.265),
            {
                "target": "log10_fatigue_life_cycles",
                "route": "basquin_only",
                "mode": "process_only",
                "candidate": "basquin_only",
                "selected": False,
                "cv_r2_mean": 0.30,
            },
        ]
    )
    importance = pd.DataFrame(
        [
            {
                "target": target,
                "evidence_stability": "supported_by_both",
            }
            for target in ["uts_MPa", "yield_strength_MPa"]
        ]
    )
    checks = pd.DataFrame([{"stress_scan_monotonic_nonincreasing": True}])
    oof.to_csv(tables / "oof_predictions.csv", index=False)
    summary.to_csv(tables / "experiment_summary.csv", index=False)
    importance.to_csv(tables / "feature_importance_comparison.csv", index=False)
    checks.to_csv(tables / "physical_checks.csv", index=False)
    fatigue_source_path = tmp_path / "fatigue.csv"
    fatigue_source.to_csv(fatigue_source_path, index=False)

    outputs = generate_actionable_testing_matrix(
        run_dir,
        config=ActionableMatrixConfig(),
        fatigue_frame_path=fatigue_source_path,
    )

    expected = {
        "alloy_process_domain_readiness",
        "domain_priority_shortlist",
        "client_target_template",
        "condition_evidence_static",
        "condition_evidence_fatigue",
        "selected_static_plan_24",
        "selected_static_plan_36",
        "selected_static_plan_48",
        "selected_fatigue_plan_30",
        "selected_fatigue_plan_45",
        "selected_fatigue_plan_60",
        "matrix_summary",
        "matrix_change_log",
        "matrix_config",
    }
    assert expected <= set(outputs)
    assert all(outputs[name].exists() for name in expected)
