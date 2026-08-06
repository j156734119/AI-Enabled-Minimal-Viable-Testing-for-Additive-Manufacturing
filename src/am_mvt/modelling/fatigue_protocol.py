from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from am_mvt.modelling.basquin import normalise_runout, r_ratio_bin


E466_MAX_FREQUENCY_HZ = 200.0
ULTRASONIC_MIN_FREQUENCY_HZ = 1000.0
FATIGUE_THRESHOLDS = (10_000_000.0, 20_000_000.0)
E606_STRAIN_FIELDS = (
    "total_strain_amplitude",
    "plastic_strain_amplitude",
    "elastic_strain_amplitude",
    "strain_ratio",
    "strain_rate",
)


def fatigue_protocol(frame: pd.DataFrame) -> pd.Series:
    frequency = pd.to_numeric(
        frame.get("frequency_Hz", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    control = (
        frame.get(
            "load_control",
            frame.get("control_mode", pd.Series("", index=frame.index)),
        )
        .astype("string")
        .str.strip()
        .str.lower()
    )
    has_strain = pd.Series(False, index=frame.index)
    for column in E606_STRAIN_FIELDS[:3]:
        if column in frame:
            has_strain |= pd.to_numeric(frame[column], errors="coerce").notna()

    result = pd.Series("ambiguous_frequency", index=frame.index, dtype="string")
    strain_controlled = control.str.contains("strain", na=False) & has_strain
    force_controlled = control.str.contains("force|load|stress", na=False)
    control_missing = control.fillna("").eq("")
    eligible_stress_control = force_controlled | control_missing
    result.loc[strain_controlled] = "e606_strain_controlled"
    result.loc[
        ~strain_controlled
        & eligible_stress_control
        & frequency.le(E466_MAX_FREQUENCY_HZ)
    ] = "e466_conventional"
    result.loc[
        ~strain_controlled
        & eligible_stress_control
        & frequency.ge(ULTRASONIC_MIN_FREQUENCY_HZ)
    ] = "ultrasonic_vhcf"
    non_e466_control = ~strain_controlled & ~eligible_stress_control
    result.loc[non_e466_control] = "non_e466_control"
    return result


def protocolise_fatigue_data(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in E606_STRAIN_FIELDS:
        if column not in result:
            result[column] = np.nan

    result["fatigue_protocol"] = fatigue_protocol(result)
    raw_control = (
        result.get("load_control", pd.Series("", index=result.index))
        .astype("string")
        .str.strip()
        .str.lower()
    )
    result["control_mode"] = np.select(
        [
            raw_control.str.contains("strain", na=False),
            raw_control.str.contains("displacement", na=False),
            raw_control.str.contains("force|load|stress", na=False),
            result["fatigue_protocol"].isin(
                ["e466_conventional", "ultrasonic_vhcf"]
            ),
        ],
        [
            "strain_controlled",
            "displacement_controlled",
            "force_controlled",
            "inferred_force_or_stress_controlled",
        ],
        default="unknown_control",
    )
    result["frequency_regime"] = result["fatigue_protocol"].map(
        {
            "e466_conventional": "conventional_le_200_hz",
            "ultrasonic_vhcf": "ultrasonic_ge_1000_hz",
            "ambiguous_frequency": "ambiguous_or_missing_frequency",
            "e606_strain_controlled": "strain_controlled",
            "non_e466_control": "non_e466_load_control",
        }
    )
    runout = normalise_runout(
        result.get("runout", pd.Series(pd.NA, index=result.index))
    )
    result["event_observed"] = runout.map({False: True, True: False}).astype("boolean")
    cycles = pd.to_numeric(
        result.get("fatigue_life_cycles", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    result["censor_lower_cycles"] = np.where(runout.eq(True), cycles, np.nan)
    result["runout_limit_cycles"] = np.where(runout.eq(True), cycles, np.nan)
    result["stress_definition"] = "amplitude"
    result["r_ratio_bin"] = r_ratio_bin(
        result.get("r_ratio", pd.Series(np.nan, index=result.index))
    )
    result["log10_stress_amplitude"] = np.log10(
        pd.to_numeric(result.get("stress_amplitude_MPa"), errors="coerce").where(
            lambda values: values.gt(0)
        )
    )
    result["log10_frequency"] = np.log10(
        pd.to_numeric(result.get("frequency_Hz"), errors="coerce").where(
            lambda values: values.gt(0)
        )
    )

    maximum = pd.to_numeric(
        result.get("max_stress_MPa", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    ratio = pd.to_numeric(
        result.get("r_ratio", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    amplitude = pd.to_numeric(
        result.get("stress_amplitude_MPa", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    expected = maximum * (1.0 - ratio) / 2.0
    relative_error = (amplitude - expected).abs() / expected.abs().replace(0, np.nan)
    checkable = maximum.notna() & ratio.notna() & amplitude.notna() & expected.ne(0)
    result["stress_consistency_relative_error"] = relative_error
    result["stress_consistency_status"] = np.select(
        [~checkable, relative_error.gt(0.10)],
        ["not_checkable", "review_required"],
        default="consistent",
    )
    return result


def fatigue_protocol_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "record_id",
        "source_id",
        "dataset_id",
        "fatigue_protocol",
        "control_mode",
        "frequency_regime",
        "frequency_Hz",
        "runout",
        "event_observed",
        "fatigue_life_cycles",
        "censor_lower_cycles",
        "runout_limit_cycles",
        "stress_definition",
        "stress_amplitude_MPa",
        "max_stress_MPa",
        "r_ratio",
        "stress_consistency_relative_error",
        "stress_consistency_status",
    ]
    available = [column for column in columns if column in frame]
    result = frame[available].copy()
    result["aft_training_eligible"] = (
        frame["fatigue_protocol"].eq("e466_conventional")
        & frame["event_observed"].notna()
        & ~frame["stress_consistency_status"].eq("review_required")
    )
    result["audit_reason"] = np.select(
        [
            frame["event_observed"].isna(),
            frame["stress_consistency_status"].eq("review_required"),
            frame["fatigue_protocol"].eq("ultrasonic_vhcf"),
            frame["fatigue_protocol"].eq("ambiguous_frequency"),
            frame["fatigue_protocol"].eq("e606_strain_controlled"),
            frame["fatigue_protocol"].eq("non_e466_control"),
        ],
        [
            "unknown_runout_status",
            "stress_definition_inconsistent",
            "separate_ultrasonic_route",
            "ambiguous_or_missing_frequency",
            "separate_e606_route",
            "non_e466_load_control",
        ],
        default="eligible_e466_conventional",
    )
    return result


def regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["fatigue_protocol", "frequency_regime"], dropna=False)
        .agg(
            records=("fatigue_protocol", "size"),
            dataset_groups=("evaluation_group_id", "nunique"),
            known_event_status=(
                "event_observed",
                lambda values: int(values.notna().sum()),
            ),
            observed_failures=(
                "event_observed",
                lambda values: int(values.eq(True).sum()),
            ),
            right_censored=(
                "event_observed",
                lambda values: int(values.eq(False).sum()),
            ),
            minimum_frequency_Hz=("frequency_Hz", "min"),
            maximum_frequency_Hz=("frequency_Hz", "max"),
        )
        .reset_index()
    )


def aft_survival_probability(
    predicted_location_cycles: np.ndarray,
    threshold_cycles: float,
    distribution: str,
    scale: float,
) -> np.ndarray:
    location = np.maximum(np.asarray(predicted_location_cycles, dtype=float), 1.0)
    z = (math.log(float(threshold_cycles)) - np.log(location)) / float(scale)
    if distribution == "normal":
        cdf = np.vectorize(NormalDist().cdf)(z)
        return 1.0 - cdf
    if distribution == "logistic":
        return 1.0 / (1.0 + np.exp(np.clip(z, -700, 700)))
    if distribution == "extreme":
        return np.exp(-np.exp(np.clip(z, -700, 700)))
    raise ValueError(f"Unsupported AFT distribution: {distribution}")


def aft_life_quantile(
    predicted_location_cycles: np.ndarray,
    probability: float,
    distribution: str,
    scale: float,
) -> np.ndarray:
    if not 0 < probability < 1:
        raise ValueError("probability must be strictly between zero and one")
    if distribution == "normal":
        z = NormalDist().inv_cdf(probability)
    elif distribution == "logistic":
        z = math.log(probability / (1.0 - probability))
    elif distribution == "extreme":
        z = math.log(-math.log(1.0 - probability))
    else:
        raise ValueError(f"Unsupported AFT distribution: {distribution}")
    location = np.maximum(np.asarray(predicted_location_cycles, dtype=float), 1.0)
    return np.exp(np.log(location) + float(scale) * z)


def threshold_labels(frame: pd.DataFrame, threshold_cycles: float) -> pd.Series:
    life = pd.to_numeric(frame["fatigue_life_cycles"], errors="coerce")
    event = frame["event_observed"].astype("boolean")
    labels = pd.Series(np.nan, index=frame.index, dtype=float)
    labels.loc[event.eq(True) & life.lt(threshold_cycles)] = 0.0
    labels.loc[life.ge(threshold_cycles)] = 1.0
    return labels


@dataclass
class IsotonicCalibration:
    x_thresholds: list[float]
    y_thresholds: list[float]
    level: str
    key: str
    sample_count: int
    positive_count: int
    negative_count: int

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(values, dtype=float),
            np.asarray(self.x_thresholds, dtype=float),
            np.asarray(self.y_thresholds, dtype=float),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_thresholds": self.x_thresholds,
            "y_thresholds": self.y_thresholds,
            "level": self.level,
            "key": self.key,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IsotonicCalibration":
        return cls(**payload)


def fatigue_domain_key(row: pd.Series, level: str) -> str:
    if level == "exact":
        columns = ["alloy", "am_process", "fatigue_protocol", "r_ratio_bin"]
    elif level == "family":
        columns = ["alloy_family", "fatigue_protocol", "r_ratio_bin"]
    elif level == "protocol":
        columns = ["fatigue_protocol"]
    else:
        raise ValueError(f"Unknown fatigue domain level: {level}")
    return "|".join(
        "missing" if pd.isna(row.get(column)) else str(row.get(column))
        for column in columns
    )


def select_fatigue_domain_route(
    row: pd.Series,
    support: list[dict[str, Any]],
) -> tuple[str, str]:
    eligible = {
        (str(item["level"]), str(item["key"]))
        for item in support
        if bool(item.get("eligible"))
    }
    for level in ("exact", "family"):
        key = fatigue_domain_key(row, level)
        if (level, key) in eligible:
            return level, key
    return "not_assessable", ""


def calibrate_threshold_probability(
    raw_probability: float,
    row: pd.Series,
    threshold_cycles: float,
    calibrations: dict[str, dict[str, Any]],
) -> tuple[float, str]:
    for level in ("exact", "family", "protocol"):
        key = fatigue_domain_key(row, level)
        lookup = f"{int(threshold_cycles)}::{level}::{key}"
        payload = calibrations.get(lookup)
        if payload:
            calibration = IsotonicCalibration.from_dict(payload)
            value = float(calibration.predict(np.asarray([raw_probability]))[0])
            return value, level
    return float(raw_probability), "uncalibrated"


def fit_isotonic_calibration(
    probabilities: pd.Series,
    labels: pd.Series,
    *,
    level: str,
    key: str,
    minimum_class_count: int = 20,
) -> IsotonicCalibration | None:
    data = pd.DataFrame({"probability": probabilities, "label": labels}).dropna()
    positives = int(data["label"].eq(1).sum())
    negatives = int(data["label"].eq(0).sum())
    if positives < minimum_class_count or negatives < minimum_class_count:
        return None
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(data["probability"], data["label"])
    return IsotonicCalibration(
        x_thresholds=model.X_thresholds_.astype(float).tolist(),
        y_thresholds=model.y_thresholds_.astype(float).tolist(),
        level=level,
        key=key,
        sample_count=len(data),
        positive_count=positives,
        negative_count=negatives,
    )


def e606_assessment(frame: pd.DataFrame) -> str:
    required = E606_STRAIN_FIELDS[:3]
    if not all(column in frame for column in required):
        return "not_assessable_missing_strain_controlled_data"
    complete = (
        frame[list(required)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    if not complete.any():
        return "not_assessable_missing_strain_controlled_data"
    return "strain_controlled_data_available"
