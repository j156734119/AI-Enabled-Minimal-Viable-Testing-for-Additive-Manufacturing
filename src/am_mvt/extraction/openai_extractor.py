from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from am_mvt.config import load_config, load_project_environment
from am_mvt.extraction.prompts import SYSTEM_PROMPT, build_user_prompt
from am_mvt.extraction.schemas import AM_EXTRACTION_SCHEMA


def extract_records_from_text(
    text: str,
    source_hint: str = "",
    model: str | None = None,
) -> dict[str, Any]:
    load_project_environment()
    config = load_config()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_openai_api_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create .env from .env.example and add your key."
        )

    if model is None:
        model = os.getenv("OPENAI_MODEL") or config.get("openai", {}).get(
            "model", "gpt-4o-mini"
        )

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text, source_hint)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "am_extraction_result",
                "schema": AM_EXTRACTION_SCHEMA,
                "strict": True,
            }
        },
        temperature=0,
    )

    output_text = response.output_text
    parsed = json.loads(output_text)

    return parsed


def extract_records_from_file(
    input_path: str | Path,
    output_path: str | Path,
    source_hint: str = "",
    model: str | None = None,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    result = extract_records_from_text(text=text, source_hint=source_hint, model=model)

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path