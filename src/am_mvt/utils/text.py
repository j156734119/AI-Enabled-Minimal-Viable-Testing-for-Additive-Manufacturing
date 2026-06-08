from __future__ import annotations

import re

import pandas as pd


def normalise_identifier(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", "", str(value).strip().lower())


def normalise_doi(value: object) -> str:
    text = normalise_identifier(value)
    text = re.sub(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", "", text)
    return text.rstrip(".,;:)]}")
