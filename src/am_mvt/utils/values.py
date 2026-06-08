from __future__ import annotations

from typing import Any

import pandas as pd


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def parse_boolean(value: Any) -> bool | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "runout", "run-out", "survived"}:
        return True
    if text in {"false", "0", "no", "n", "failure", "failed"}:
        return False
    return None
