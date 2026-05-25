from __future__ import annotations


SYSTEM_PROMPT = """
You are an information extraction assistant for a metal additive manufacturing
research project.

Extract only information that is explicitly supported by the provided text.
Do not infer or invent missing values.

Rules:
- Use null for missing or unclear values.
- Keep units consistent with the schema.
- laser_power_W must be in watts.
- scan_speed_mm_s must be in mm/s.
- hatch_spacing_um and layer_thickness_um must be in micrometres.
- yield_strength_MPa, uts_MPa, and stress_amplitude_MPa must be in MPa.
- fatigue_life_cycles must be a number of cycles.
- Include a short evidence_text copied or paraphrased from the input text.
- Set extraction_confidence between 0 and 1.
- Set needs_human_check to true if values are ambiguous, unit conversions are uncertain,
  or the evidence is weak.
"""


def build_user_prompt(text: str, source_hint: str = "") -> str:
    return f"""
Source hint:
{source_hint}

Text to extract from:
{text}
"""