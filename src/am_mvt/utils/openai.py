from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from am_mvt.config import get_path


def get_openai_client() -> OpenAI:
    env_path = get_path(".env")
    load_dotenv(env_path if env_path.exists() else None)
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to the project .env file or "
            "set it in the current PowerShell session."
        )
    return OpenAI()


def extract_output_text(response: Any) -> str:
    if output_text := getattr(response, "output_text", None):
        return output_text

    return "\n".join(
        content.text
        for item in (getattr(response, "output", None) or [])
        for content in (getattr(item, "content", None) or [])
        if getattr(content, "text", None)
    )
