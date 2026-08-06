from __future__ import annotations

import pandas as pd

from am_mvt.optimisation.client_target_matrix import (
    _point_gate,
    _representative_fatigue_condition,
    _robust_gate,
    _select_representative_static,
)


def test_client_target_gates_distinguish_point_and_robust_interval():
    target = {"lower_bound": 340.0, "upper_bound": float("nan")}

    assert _point_gate(360.0, target)
    assert not _robust_gate(300.0, 420.0, target)
    assert _robust_gate(350.0, 420.0, target)


def test_static_selector_respects_budget_and_preserves_orientation_diversity():
    candidates = pd.DataFrame(
        [
            {
                "build_orientation": str(index % 3 * 45),
                "laser_power_W": 300 + index * 10,
                "scan_speed_mm_s": 900 + index * 50,
                "hatch_spacing_um": 130 + index * 5,
                "layer_thickness_um": 30 + index,
                "point_target_gate_count": 4,
            }
            for index in range(10)
        ]
    )

    selected = _select_representative_static(candidates, 6)

    assert len(selected) == 6
    assert selected["build_orientation"].nunique() == 3


def test_fatigue_representative_uses_requested_conditions_and_nearby_loading():
    oof = pd.DataFrame(
        [
            {
                "target": "log10_fatigue_life_cycles",
                "route": "xgboost_aft",
                "mode": "process_only",
                "alloy": "AlSi10Mg",
                "am_process": "L-PBF",
                "build_orientation": "0",
                "surface_condition": "machined",
                "heat_treatment": "heat-treated",
                "stress_amplitude_MPa": 110.0,
                "r_ratio": -1.0,
                "frequency_Hz": 50.0,
                "test_temperature_C": 25.0,
                "dataset_id": "best",
                "record_id": "best-row",
            },
            {
                "target": "log10_fatigue_life_cycles",
                "route": "xgboost_aft",
                "mode": "process_only",
                "alloy": "AlSi10Mg",
                "am_process": "L-PBF",
                "build_orientation": "0",
                "surface_condition": "as-built",
                "heat_treatment": "no-heat-treatment",
                "stress_amplitude_MPa": 80.0,
                "r_ratio": 0.1,
                "frequency_Hz": 20.0,
                "test_temperature_C": 25.0,
                "dataset_id": "other",
                "record_id": "other-row",
            },
        ]
    )
    target = {
        "reference_stress_amplitude_MPa": 110.0,
        "r_ratio": -1.0,
        "frequency_Hz": 50.0,
        "test_temperature_C": 25.0,
        "execution_surface_condition": "machined",
        "execution_heat_treatment": "stress_relieved",
    }

    selected = _representative_fatigue_condition(
        oof,
        "AlSi10Mg",
        "L-PBF",
        "0",
        target,
    )

    assert selected["dataset_id"] == "best"
