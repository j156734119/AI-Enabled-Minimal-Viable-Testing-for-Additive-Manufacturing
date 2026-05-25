from __future__ import annotations


STRING_FIELDS = [
    "source_title",
    "doi",
    "source_file",
    "source_sheet",
    "page_or_section",
    "alloy",
    "alloy_family",
    "am_process",
    "machine_model",
    "build_orientation",
    "test_direction",
    "scan_strategy",
    "surface_condition",
    "heat_treatment",
    "post_processing",
    "density_measurement_method",
    "defect_type",
    "residual_stress_indicator",
    "test_type",
    "runout",
    "failure_mode",
    "fracture_origin",
    "evidence_text",
]

NUMBER_FIELDS = [
    "source_year",
    "laser_power_W",
    "scan_speed_mm_s",
    "hatch_spacing_um",
    "layer_thickness_um",
    "ved_J_mm3",
    "layer_rotation_degree",
    "build_plate_temperature_C",
    "porosity_percent",
    "relative_density_percent",
    "test_temperature_C",
    "yield_strength_MPa",
    "uts_MPa",
    "elongation_percent",
    "youngs_modulus_GPa",
    "hardness_HV",
    "stress_amplitude_MPa",
    "max_stress_MPa",
    "strain_amplitude",
    "delta_K_MPa_sqrt_m",
    "da_dN_m_per_cycle",
    "r_ratio",
    "frequency_Hz",
    "fatigue_life_cycles",
    "fatigue_life_h",
    "confidence",
]

BOOLEAN_FIELDS = [
    "needs_human_check",
]


def nullable_string_schema() -> dict:
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "null"},
        ]
    }


def nullable_number_schema() -> dict:
    return {
        "anyOf": [
            {"type": "number"},
            {"type": "null"},
        ]
    }


def nullable_boolean_schema() -> dict:
    return {
        "anyOf": [
            {"type": "boolean"},
            {"type": "null"},
        ]
    }


def make_record_schema() -> dict:
    properties: dict[str, dict] = {}

    for field in STRING_FIELDS:
        properties[field] = nullable_string_schema()

    for field in NUMBER_FIELDS:
        properties[field] = nullable_number_schema()

    for field in BOOLEAN_FIELDS:
        properties[field] = nullable_boolean_schema()

    required_fields = STRING_FIELDS + NUMBER_FIELDS + BOOLEAN_FIELDS

    return {
        "type": "object",
        "properties": properties,
        "required": required_fields,
        "additionalProperties": False,
    }


LLM_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": make_record_schema(),
        }
    },
    "required": ["records"],
    "additionalProperties": False,
}