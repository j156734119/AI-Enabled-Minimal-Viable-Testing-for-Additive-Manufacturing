from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ExtractionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_title: str | None
    doi: str | None
    source_file: str | None
    source_sheet: str | None
    page_or_section: str | None
    alloy: str | None
    alloy_family: str | None
    am_process: str | None
    machine_model: str | None
    build_orientation: str | None
    test_direction: str | None
    scan_strategy: str | None
    surface_condition: str | None
    heat_treatment: str | None
    post_processing: str | None
    density_measurement_method: str | None
    defect_type: str | None
    residual_stress_indicator: str | None
    test_type: str | None
    runout: str | None
    failure_mode: str | None
    fracture_origin: str | None
    evidence_text: str | None
    source_year: float | None
    laser_power_W: float | None
    scan_speed_mm_s: float | None
    hatch_spacing_um: float | None
    layer_thickness_um: float | None
    ved_J_mm3: float | None
    layer_rotation_degree: float | None
    build_plate_temperature_C: float | None
    porosity_percent: float | None
    relative_density_percent: float | None
    test_temperature_C: float | None
    yield_strength_MPa: float | None
    uts_MPa: float | None
    elongation_percent: float | None
    youngs_modulus_GPa: float | None
    hardness_HV: float | None
    stress_amplitude_MPa: float | None
    max_stress_MPa: float | None
    strain_amplitude: float | None
    delta_K_MPa_sqrt_m: float | None
    da_dN_m_per_cycle: float | None
    r_ratio: float | None
    frequency_Hz: float | None
    fatigue_life_cycles: float | None
    fatigue_life_h: float | None
    confidence: float | None
    needs_human_check: bool | None


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ExtractionRecord]


def openai_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


def parse_extraction_response(payload: dict[str, Any]) -> dict[str, Any]:
    return ExtractionResponse.model_validate(payload).model_dump(mode="json")


LLM_EXTRACTION_SCHEMA = openai_json_schema(ExtractionResponse)
