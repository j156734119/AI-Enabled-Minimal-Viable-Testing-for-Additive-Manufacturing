from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BasquinCurve:
    intercept: float
    slope: float
    sample_count: int
    level: str
    key: str


def normalise_runout(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "runout": True,
                "run-out": True,
                "run out": True,
                "run-out samples": True,
                "run out samples": True,
                "survived": True,
                "false": False,
                "0": False,
                "no": False,
                "failure": False,
                "failed": False,
            }
        )
    )


def r_ratio_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = [-np.inf, -0.5, -0.05, 0.05, 0.3, 0.7, np.inf]
    labels = [
        "R_le_-0.5",
        "R_-0.5_to_-0.05",
        "R_near_zero",
        "R_0.05_to_0.3",
        "R_0.3_to_0.7",
        "R_gt_0.7",
    ]
    result = pd.cut(numeric, bins=bins, labels=labels, include_lowest=True)
    return result.astype("string").fillna("R_missing")


def fit_curve(
    stress_amplitude: pd.Series,
    log_life: pd.Series,
    level: str,
    key: str,
    minimum_rows: int,
) -> BasquinCurve | None:
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(stress_amplitude, errors="coerce"),
            "y": pd.to_numeric(log_life, errors="coerce"),
        }
    ).dropna()
    frame = frame.loc[frame["x"] > 0].copy()

    if len(frame) < minimum_rows or frame["x"].nunique() < 3:
        return None

    x = np.log10(frame["x"].to_numpy(dtype=float))
    y = frame["y"].to_numpy(dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(np.square(x - x_mean).sum())

    if denominator <= 0:
        return None

    slope = float(((x - x_mean) * (y - y_mean)).sum() / denominator)

    if slope >= 0:
        return None

    intercept = y_mean - slope * x_mean
    return BasquinCurve(
        intercept=intercept,
        slope=slope,
        sample_count=len(frame),
        level=level,
        key=key,
    )


class HierarchicalBasquin:
    def __init__(
        self,
        family_r_min_rows: int = 30,
        family_min_rows: int = 50,
        global_min_rows: int = 100,
    ) -> None:
        self.family_r_min_rows = family_r_min_rows
        self.family_min_rows = family_min_rows
        self.global_min_rows = global_min_rows
        self.global_curve: BasquinCurve | None = None
        self.family_curves: dict[str, BasquinCurve] = {}
        self.family_r_curves: dict[str, BasquinCurve] = {}

    def fit(self, df: pd.DataFrame) -> "HierarchicalBasquin":
        working = df.copy()
        working["runout_bool"] = normalise_runout(working["runout"])
        working = working.loc[working["runout_bool"].eq(False)].copy()
        working["alloy_family_key"] = (
            working["alloy_family"]
            .astype("string")
            .str.strip()
            .fillna("missing")
        )
        working["r_ratio_bin"] = r_ratio_bin(working["r_ratio"])

        self.global_curve = fit_curve(
            working["stress_amplitude_MPa"],
            working["log10_fatigue_life_cycles"],
            level="global",
            key="global",
            minimum_rows=self.global_min_rows,
        )

        if self.global_curve is None:
            raise ValueError("Could not fit a valid negative-slope global Basquin curve.")

        for family, group in working.groupby("alloy_family_key"):
            curve = fit_curve(
                group["stress_amplitude_MPa"],
                group["log10_fatigue_life_cycles"],
                level="alloy_family",
                key=str(family),
                minimum_rows=self.family_min_rows,
            )
            if curve is not None:
                self.family_curves[str(family)] = curve

        for (family, ratio_bin), group in working.groupby(
            ["alloy_family_key", "r_ratio_bin"],
            dropna=False,
        ):
            key = f"{family}::{ratio_bin}"
            curve = fit_curve(
                group["stress_amplitude_MPa"],
                group["log10_fatigue_life_cycles"],
                level="alloy_family_r_ratio",
                key=key,
                minimum_rows=self.family_r_min_rows,
            )
            if curve is not None:
                self.family_r_curves[key] = curve

        return self

    def curve_for_row(self, row: pd.Series) -> BasquinCurve:
        if self.global_curve is None:
            raise RuntimeError("Basquin model has not been fitted.")

        family_value = row.get("alloy_family")
        family = (
            "missing"
            if family_value is None or pd.isna(family_value)
            else str(family_value).strip()
        )
        ratio = r_ratio_bin(pd.Series([row.get("r_ratio")])).iloc[0]
        combined_key = f"{family}::{ratio}"

        return (
            self.family_r_curves.get(combined_key)
            or self.family_curves.get(family)
            or self.global_curve
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        predictions = []

        for _, row in df.iterrows():
            stress = pd.to_numeric(
                pd.Series([row.get("stress_amplitude_MPa")]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(stress) or stress <= 0:
                predictions.append(np.nan)
                continue

            curve = self.curve_for_row(row)
            predictions.append(
                curve.intercept + curve.slope * math.log10(float(stress))
            )

        return np.asarray(predictions, dtype=float)

    def parameters_frame(self) -> pd.DataFrame:
        curves = [self.global_curve] if self.global_curve is not None else []
        curves += list(self.family_curves.values())
        curves += list(self.family_r_curves.values())
        return pd.DataFrame(asdict(curve) for curve in curves)

    def to_dict(self) -> dict[str, object]:
        return {
            "family_r_min_rows": self.family_r_min_rows,
            "family_min_rows": self.family_min_rows,
            "global_min_rows": self.global_min_rows,
            "global_curve": (
                asdict(self.global_curve) if self.global_curve is not None else None
            ),
            "family_curves": {
                key: asdict(value) for key, value in self.family_curves.items()
            },
            "family_r_curves": {
                key: asdict(value) for key, value in self.family_r_curves.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HierarchicalBasquin":
        model = cls(
            family_r_min_rows=int(payload["family_r_min_rows"]),
            family_min_rows=int(payload["family_min_rows"]),
            global_min_rows=int(payload["global_min_rows"]),
        )
        global_curve = payload.get("global_curve")
        if global_curve:
            model.global_curve = BasquinCurve(**global_curve)
        model.family_curves = {
            key: BasquinCurve(**value)
            for key, value in dict(payload.get("family_curves", {})).items()
        }
        model.family_r_curves = {
            key: BasquinCurve(**value)
            for key, value in dict(payload.get("family_r_curves", {})).items()
        }
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalBasquin":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def basquin_residual_features(
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[str], list[str]]:
    excluded = {
        "stress_amplitude_MPa",
        "max_stress_MPa",
        "yield_strength_MPa",
        "uts_MPa",
        "elongation_percent",
        "youngs_modulus_GPa",
        "hardness_HV",
    }
    numeric = [feature for feature in numeric_features if feature not in excluded]
    return numeric, list(categorical_features)
