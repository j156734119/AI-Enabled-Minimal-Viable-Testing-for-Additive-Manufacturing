from __future__ import annotations

import pandas as pd


def normalise_defect_type(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()

    if not text:
        return pd.NA

    if "lack" in text and "fusion" in text:
        return "lack_of_fusion"

    if "lof" == text:
        return "lack_of_fusion"

    if "keyhole" in text:
        return "keyhole_pore"

    if "gas" in text and ("pore" in text or "porosity" in text):
        return "gas_pore"

    if "pore" in text or "porosity" in text:
        return "pore"

    if "crack" in text:
        return "crack"

    if "inclusion" in text:
        return "inclusion"

    return text.replace(" ", "_")


def apply_defect_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "defect_type" in result.columns:
        result["defect_type"] = result["defect_type"].map(normalise_defect_type)

    return result