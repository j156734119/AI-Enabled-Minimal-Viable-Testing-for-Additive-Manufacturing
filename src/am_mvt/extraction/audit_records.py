from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from am_mvt.cleaning.project_schema import (
    INPUT_COLUMNS,
    MASTER_COLUMNS,
    OUTPUT_COLUMNS,
)
from am_mvt.config import get_path


AUDIT_STATUSES = {
    "approved",
    "human_review_required",
    "rejected",
}

AUDIT_METHODS = {
    "deterministic",
    "human_review",
}

REQUIRED_EVIDENCE_FIELDS = [
    "source_id",
    "source_file",
    "record_id",
    "evidence_text",
]

USEFUL_FIELDS = [
    column
    for column in INPUT_COLUMNS + OUTPUT_COLUMNS
    if column
    not in {
        "residual_stress_indicator",
        "runout",
        "failure_mode",
        "fracture_origin",
    }
]

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "source_year": (1900, 2100),
    "laser_power_W": (0, 2000),
    "scan_speed_mm_s": (0, 20000),
    "hatch_spacing_um": (0, 1000),
    "layer_thickness_um": (0, 500),
    "ved_J_mm3": (0, 1000),
    "layer_rotation_degree": (-360, 360),
    "build_plate_temperature_C": (-273.15, 2000),
    "porosity_percent": (0, 100),
    "relative_density_percent": (0, 100),
    "test_temperature_C": (-273.15, 3000),
    "yield_strength_MPa": (0, 3500),
    "uts_MPa": (0, 4000),
    "elongation_percent": (0, 150),
    "youngs_modulus_GPa": (0, 500),
    "hardness_HV": (0, 3000),
    "stress_amplitude_MPa": (0, 3000),
    "max_stress_MPa": (0, 4000),
    "strain_amplitude": (0, 10),
    "delta_K_MPa_sqrt_m": (0, 500),
    "da_dN_m_per_cycle": (0, 1),
    "r_ratio": (-10, 1),
    "frequency_Hz": (0, 1_000_000),
    "fatigue_life_cycles": (0, 1e12),
    "fatigue_life_h": (0, 1e9),
}

AUDIT_DECISION_COLUMNS = [
    "source_id",
    "record_id",
    "record_fingerprint",
    "audit_status",
    "audit_reason",
    "audit_method",
    "reviewed_by",
    "reviewed_at",
]

FINGERPRINT_COLUMNS = list(
    dict.fromkeys(
        MASTER_COLUMNS
        + [
            "source_title",
            "doi",
            "page_or_section",
            "evidence_text",
            "confidence",
        ]
    )
)


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

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return None


def normalise_fingerprint_value(value: Any) -> Any:
    if is_missing(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value

    return str(value).strip()


def record_fingerprint(row: pd.Series) -> str:
    payload = {
        column: normalise_fingerprint_value(row.get(column))
        for column in FINGERPRINT_COLUMNS
    }
    serialised = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def find_numeric_issues(row: pd.Series) -> list[str]:
    issues: list[str] = []

    for field, (lower, upper) in NUMERIC_RANGES.items():
        if field not in row or is_missing(row[field]):
            continue

        numeric_value = pd.to_numeric(pd.Series([row[field]]), errors="coerce").iloc[0]

        if pd.isna(numeric_value):
            issues.append(f"{field}:not_numeric")
        elif numeric_value < lower or numeric_value > upper:
            issues.append(f"{field}:outside_{lower}_{upper}")

    return issues


def has_useful_data(row: pd.Series) -> bool:
    return any(field in row and not is_missing(row[field]) for field in USEFUL_FIELDS)


def audit_record(row: pd.Series) -> dict[str, str]:
    reasons: list[str] = []
    missing_fields = [
        field
        for field in REQUIRED_EVIDENCE_FIELDS
        if field not in row or is_missing(row[field])
    ]
    numeric_issues = find_numeric_issues(row)

    if not has_useful_data(row):
        return {
            "audit_status": "rejected",
            "audit_reason": "no_useful_am_or_mechanical_data",
            "audit_method": "deterministic",
        }

    if numeric_issues:
        return {
            "audit_status": "rejected",
            "audit_reason": ";".join(numeric_issues),
            "audit_method": "deterministic",
        }

    if missing_fields:
        reasons.append("missing:" + ",".join(missing_fields))

    confidence = pd.to_numeric(
        pd.Series([row.get("confidence")]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(confidence):
        reasons.append("missing:confidence")
    elif confidence < 0 or confidence > 1:
        return {
            "audit_status": "rejected",
            "audit_reason": "confidence:outside_0_1",
            "audit_method": "deterministic",
        }
    elif confidence < 0.70:
        reasons.append("confidence_below_0.70")

    needs_human_check = parse_boolean(row.get("needs_human_check"))

    if needs_human_check is None:
        reasons.append("needs_human_check:not_boolean")
    elif needs_human_check:
        reasons.append("needs_human_check:true")

    if reasons:
        return {
            "audit_status": "human_review_required",
            "audit_reason": ";".join(reasons),
            "audit_method": "deterministic",
        }

    return {
        "audit_status": "approved",
        "audit_reason": "deterministic_checks_passed",
        "audit_method": "deterministic",
    }


def audit_extracted_records(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if result.empty:
        for column in [
            "record_fingerprint",
            "audit_status",
            "audit_reason",
            "audit_method",
            "reviewed_by",
            "reviewed_at",
        ]:
            result[column] = pd.Series(dtype="string")
        return result

    result["record_fingerprint"] = result.apply(record_fingerprint, axis=1)
    decisions = result.apply(audit_record, axis=1, result_type="expand")

    for column in decisions.columns:
        result[column] = decisions[column]

    result["reviewed_by"] = ""
    result["reviewed_at"] = ""
    return result


def save_extraction_audit(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    input_path = (
        Path(input_path)
        if input_path is not None
        else get_path("data", "interim", "llm_extracted_records.csv")
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else get_path("data", "interim", "llm_extraction_audit_review.csv")
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Candidate extraction file not found: {input_path}. "
            "Run scripts/04_extract_with_llm.py first."
        )

    candidate_df = pd.read_csv(input_path, low_memory=False)
    audited_df = audit_extracted_records(candidate_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audited_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path, audited_df


def load_approved_record_keys(audit_path: str | Path) -> pd.DataFrame:
    audit_path = Path(audit_path)

    if not audit_path.exists():
        raise FileNotFoundError(
            f"Extraction audit file not found: {audit_path}. "
            "Run python scripts/04b_audit_extractions.py before merging."
        )

    audit_df = pd.read_csv(audit_path, low_memory=False)
    required_columns = {
        "source_id",
        "record_id",
        "record_fingerprint",
        "audit_status",
        "audit_method",
        "reviewed_by",
        "reviewed_at",
    }
    missing_columns = required_columns - set(audit_df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Extraction audit is missing required columns: {missing}")

    invalid_statuses = set(audit_df["audit_status"].dropna().astype(str)) - AUDIT_STATUSES

    if invalid_statuses:
        invalid = ", ".join(sorted(invalid_statuses))
        raise ValueError(f"Extraction audit contains invalid statuses: {invalid}")

    invalid_methods = set(audit_df["audit_method"].dropna().astype(str)) - AUDIT_METHODS

    if invalid_methods:
        invalid = ", ".join(sorted(invalid_methods))
        raise ValueError(f"Extraction audit contains invalid methods: {invalid}")

    duplicate_keys = audit_df.duplicated(
        subset=["source_id", "record_id"],
        keep=False,
    )

    if duplicate_keys.any():
        raise ValueError("Extraction audit contains duplicate source_id/record_id keys.")

    approved = audit_df["audit_status"].eq("approved")
    manual_approved = approved & audit_df["audit_method"].eq("human_review")
    missing_manual_metadata = manual_approved & (
        audit_df["reviewed_by"].fillna("").astype(str).str.strip().eq("")
        | audit_df["reviewed_at"].fillna("").astype(str).str.strip().eq("")
    )

    if missing_manual_metadata.any():
        raise ValueError(
            "Human-approved records require reviewed_by and reviewed_at metadata."
        )

    return audit_df.loc[
        approved,
        AUDIT_DECISION_COLUMNS,
    ].copy()
