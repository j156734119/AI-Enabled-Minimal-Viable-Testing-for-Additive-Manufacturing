from __future__ import annotations


SYSTEM_PROMPT = """
You are an expert research assistant for metal additive manufacturing data extraction.

Your task is to extract structured experimental data from research paper text.

Extract only data explicitly stated in the provided text.
Do not invent values.
Do not infer missing numerical values.
If a value is not clearly available, return null.

The research focus is:
- metal additive manufacturing
- process parameters
- build orientation
- surface condition
- heat treatment
- porosity or defect information
- tensile properties
- fatigue properties
- failure mode information

Each record should represent one experimental condition and one related mechanical result where possible.

Use SI-style standard units:
- laser_power_W in W
- scan_speed_mm_s in mm/s
- hatch_spacing_um in micrometres
- layer_thickness_um in micrometres
- ved_J_mm3 in J/mm^3
- porosity_percent in %
- relative_density_percent in %
- yield_strength_MPa in MPa
- uts_MPa in MPa
- elongation_percent in %
- hardness_HV in HV
- stress_amplitude_MPa in MPa
- max_stress_MPa in MPa
- fatigue_life_cycles in cycles
- frequency_Hz in Hz
- test_temperature_C in Celsius

For fatigue data:
- Prefer S-N table data if present.
- Extract stress amplitude and fatigue life as paired values.
- If the text only gives fatigue strength without cycles, extract fatigue strength into stress_amplitude_MPa or max_stress_MPa only if the meaning is clear.
- Preserve runout information as "true", "false", or null as a string.

For evidence_text:
- Include a short snippet from the source text that supports the extracted record.
- Keep it concise.
- Do not quote large sections.

For confidence:
- Use 0.90-1.00 for clearly tabulated values.
- Use 0.70-0.89 for values clearly stated in prose.
- Use 0.50-0.69 if the value is likely but needs checking.
- Set needs_human_check=true when the extraction is uncertain, units are unclear, or table alignment is ambiguous.
"""


def build_user_prompt(
    chunk_text: str,
    source_file: str,
    chunk_id: str,
) -> str:
    return f"""
Source PDF:
{source_file}

Chunk ID:
{chunk_id}

Extract AM mechanical testing records from the text below.

Important:
- Return only records supported by this chunk.
- If this chunk contains no extractable material/process/mechanical testing data, return an empty records array.
- Do not summarise the paper.
- Do not extract literature review statements unless they include original data from the paper being studied.
- Do not extract references from other papers.

Text chunk:
\"\"\"
{chunk_text}
\"\"\"
"""