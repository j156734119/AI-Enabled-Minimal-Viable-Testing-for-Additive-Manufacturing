from __future__ import annotations

import re


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()