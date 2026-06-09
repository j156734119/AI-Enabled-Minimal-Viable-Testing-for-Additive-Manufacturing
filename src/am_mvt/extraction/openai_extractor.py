from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from am_mvt.extraction.prompts import build_system_prompt, build_user_prompt
from am_mvt.extraction.schemas import LLM_EXTRACTION_SCHEMA, parse_extraction_response
from am_mvt.utils.openai import extract_output_text, get_openai_client


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


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
            parsed = parse_extraction_response(json.loads(raw_text))

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


def extract_records_from_pdf(
    pdf_path: str | Path,
    source_file: str,
    chunk_id: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    client = get_openai_client()
    encoded_pdf = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    prompt = build_user_prompt(
        chunk_text=(
            "The attached PDF has no usable text layer. Read the page images "
            "directly and extract only explicitly reported numerical records. "
            "Retain page-level evidence and leave missing values null."
        ),
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
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_file",
                                "filename": source_file,
                                "file_data": (
                                    "data:application/pdf;base64," + encoded_pdf
                                ),
                            },
                        ],
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
            parsed = parse_extraction_response(
                json.loads(extract_output_text(response))
            )
            parsed["_metadata"] = {
                "source_file": source_file,
                "chunk_id": chunk_id,
                "model": model,
                "attempt": attempt,
                "input_mode": "pdf_vision",
            }
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds * attempt)

    return {
        "records": [],
        "_metadata": {
            "source_file": source_file,
            "chunk_id": chunk_id,
            "model": model,
            "input_mode": "pdf_vision",
            "error": str(last_error),
        },
    }
