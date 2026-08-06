from __future__ import annotations

import pandas as pd

from am_mvt.cleaning.project_schema import standardise_table_to_project_schema
from am_mvt.ingestion.load_fatigue_database import enrich_fatigue_condition_fields
from am_mvt.modelling.experiment_config import get_experiment_config
from am_mvt.modelling.fatigue_protocol import protocolise_fatigue_data


def test_database_condition_fields_are_preserved_and_structured():
    raw = pd.DataFrame(
        {
            "processing sequence and parameters": [
                "HIP 500 C, 2 hour; HT 160 C, 7 hour; SURF machine; SURF polish"
            ],
            "specimens description": [
                "circular cross-section; with drawing; ASTM E466"
            ],
            "critical cross-section size of specimens\n(mm)": ["5"],
            "stress concentration factor of specimens": ["1.0"],
            "fatigue environment": ["air"],
            "fatigue machine": ["MTS 810"],
            "fatigue standard": ["ASTM E466"],
            "load control": ["force"],
            "AM environment": ["Ar"],
        }
    )

    result = enrich_fatigue_condition_fields(
        standardise_table_to_project_schema(raw)
    )

    assert result.loc[0, "surface_condition"] == "machined+polished"
    assert result.loc[0, "heat_treatment"] == "hip+heat-treated"
    assert result.loc[0, "material_state"] == "hip-and-heat-treated"
    assert result.loc[0, "specimen_geometry"] == "circular-cross-section"
    assert result.loc[0, "critical_section_size_mm"] == 5.0
    assert result.loc[0, "stress_concentration_factor"] == "1.0"
    assert result.loc[0, "fatigue_standard"] == "ASTM E466"
    assert result.loc[0, "load_control"] == "force"
    assert result.loc[0, "am_environment"] == "Ar"


def test_compound_section_dimensions_are_retained_but_not_guessed_as_scalar():
    raw = pd.DataFrame(
        {
            "critical cross-section size of specimens\n(mm)": ["50, 10"],
            "specimens description": ["rectangular cross-section"],
        }
    )

    result = enrich_fatigue_condition_fields(
        standardise_table_to_project_schema(raw)
    )

    assert result.loc[0, "critical_section_dimensions_mm"] == "50, 10"
    assert pd.isna(result.loc[0, "critical_section_size_mm"])
    assert result.loc[0, "specimen_geometry"] == "rectangular-cross-section"


def test_displacement_control_is_not_routed_to_e466_force_control():
    frame = pd.DataFrame(
        {
            "frequency_Hz": [50.0, 50.0],
            "load_control": ["force", "displacement"],
            "stress_amplitude_MPa": [110.0, 110.0],
            "fatigue_life_cycles": [1_000_000.0, 1_000_000.0],
            "runout": [False, False],
        }
    )

    result = protocolise_fatigue_data(frame)

    assert result.loc[0, "fatigue_protocol"] == "e466_conventional"
    assert result.loc[1, "fatigue_protocol"] == "non_e466_control"
    assert result.loc[1, "control_mode"] == "displacement_controlled"


def test_fatigue_model_config_includes_recovered_condition_features():
    config = get_experiment_config(
        "model2_sn_fatigue",
        "log10_fatigue_life_cycles",
        "process_only",
    )

    assert "critical_section_size_mm" in config["numeric_features"]
    assert "stress_concentration_factor" in config["numeric_features"]
    assert "specimen_geometry" in config["categorical_features"]
    assert "fatigue_standard" in config["categorical_features"]
    assert "material_state" in config["categorical_features"]
