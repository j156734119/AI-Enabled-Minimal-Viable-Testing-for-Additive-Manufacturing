from __future__ import annotations

from am_mvt.skill_loader import build_skill_system_prompt

BASE_SYSTEM_PROMPT = """
You are a cautious research assistant extracting structured experimental data
for an MSc dissertation on metal additive manufacturing. Follow the repository
skill exactly and return only the requested structured output.
"""


def build_system_prompt() -> str:
    return build_skill_system_prompt(
        BASE_SYSTEM_PROMPT,
        "evidence-grounded-extraction",
    )


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
