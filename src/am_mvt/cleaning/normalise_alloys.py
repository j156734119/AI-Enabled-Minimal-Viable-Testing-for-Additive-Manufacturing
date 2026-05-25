from __future__ import annotations

import re

import pandas as pd


def normalise_alloy_name(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    compact = re.sub(r"[\s_\-]", "", text).lower()

    if compact in {"ti64", "ti6al4v", "ti6al4veli", "ti6al4velis"}:
        return "Ti-6Al-4V"

    if compact in {"316l", "ss316l", "316lstainlesssteel"}:
        return "316L stainless steel"

    if compact in {"in718", "inconel718", "alloy718"}:
        return "Inconel 718"

    if compact in {"in625", "inconel625", "alloy625"}:
        return "Inconel 625"

    if compact in {"hastelloyx", "hastelloy-x"}:
        return "Hastelloy X"

    if compact in {"alsi10mg", "alsi10-mg", "al-si10-mg"}:
        return "AlSi10Mg"

    return text


def infer_alloy_family(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()

    if not text:
        return pd.NA

    if text.startswith("ti") or "ti-" in text or "titanium" in text:
        return "Ti alloy"

    if "316" in text or "steel" in text or "ss" in text:
        return "Steel"

    if (
        "inconel" in text
        or "hastelloy" in text
        or "nickel" in text
        or text.startswith("in718")
        or text.startswith("in625")
        or text.startswith("in ")
    ):
        return "Ni alloy"

    if text.startswith("al") or "alsi" in text or "aluminium" in text:
        return "Al alloy"

    return "Other"


def apply_alloy_normalisation(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "alloy" in result.columns:
        result["alloy"] = result["alloy"].map(normalise_alloy_name)

    if "alloy_family" not in result.columns:
        result["alloy_family"] = pd.NA

    missing_family = result["alloy_family"].isna() | (
        result["alloy_family"].astype("string").str.strip() == ""
    )

    result.loc[missing_family, "alloy_family"] = result.loc[
        missing_family, "alloy"
    ].map(infer_alloy_family)

    existing_other = result["alloy_family"].astype("string").str.lower() == "other"
    result.loc[existing_other, "alloy_family"] = result.loc[
        existing_other, "alloy"
    ].map(infer_alloy_family)

    return result