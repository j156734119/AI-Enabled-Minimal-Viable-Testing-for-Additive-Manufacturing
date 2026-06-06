from __future__ import annotations

import json
import os
import time
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from openai import OpenAI

from am_mvt.extraction.prompts import build_system_prompt, build_user_prompt
from am_mvt.extraction.schemas import LLM_EXTRACTION_SCHEMA


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def get_openai_client() -> OpenAI:
    load_environment()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Set it in PowerShell using: $env:OPENAI_API_KEY='your_api_key'"
        )

    return OpenAI()


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)

    if output_text:
        return output_text

    parts: list[str] = []

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)

            if text:
                parts.append(text)

    return "\n".join(parts)


def extract_records_from_chunk(
    chunk_text: str,
    source_file: str,
    chunk_id: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> dict[str, Any]:
    client = get_openai_client()

    user_prompt = build_user_prompt(
        chunk_text=chunk_text,
        source_file=source_file,
        chunk_id=chunk_id,
    )

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "am_mechanical_data_extraction",
                        "strict": True,
                        "schema": LLM_EXTRACTION_SCHEMA,
                    }
                },
                temperature=0,
            )

            raw_text = extract_output_text(response)
            parsed = json.loads(raw_text)

            if "records" not in parsed:
                parsed["records"] = []

            parsed["_metadata"] = {
                "source_file": source_file,
                "chunk_id": chunk_id,
                "model": model,
                "attempt": attempt,
            }

            return parsed

        except Exception as exc:
            last_error = exc

            if attempt < max_retries:
                time.sleep(retry_sleep_seconds * attempt)
            else:
                break

    return {
        "records": [],
        "_metadata": {
            "source_file": source_file,
            "chunk_id": chunk_id,
            "model": model,
            "error": str(last_error),
        },
    }
