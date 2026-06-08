from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if not valid.any():
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
    }


def conformal_radius(y_true, y_pred, coverage: float = 0.90) -> float:
    residuals = np.abs(
        np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    )
    residuals = residuals[np.isfinite(residuals)]

    if not len(residuals):
        return float("nan")

    quantile = min(
        1.0,
        math.ceil((len(residuals) + 1) * coverage) / len(residuals),
    )
    return float(np.quantile(residuals, quantile, method="higher"))


def harrell_c_index(
    lower_bound,
    event_observed,
    predicted_time,
) -> float:
    times = np.asarray(lower_bound, dtype=float)
    events = np.asarray(event_observed, dtype=bool)
    predictions = np.asarray(predicted_time, dtype=float)
    valid = np.isfinite(times) & np.isfinite(predictions)
    times = times[valid]
    events = events[valid]
    predictions = predictions[valid]
    unique_predictions = np.unique(predictions)
    ranks = np.searchsorted(unique_predictions, predictions) + 1
    tree = np.zeros(len(unique_predictions) + 1, dtype=int)

    def update(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def query(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    concordant = 0.0
    comparable = 0
    order = np.argsort(-times, kind="mergesort")
    position = 0
    later_count = 0

    while position < len(order):
        group_end = position + 1
        current_time = times[order[position]]

        while group_end < len(order) and times[order[group_end]] == current_time:
            group_end += 1

        for ordered_index in order[position:group_end]:
            if not events[ordered_index]:
                continue
            rank = int(ranks[ordered_index])
            lower_or_equal = query(rank)
            lower = query(rank - 1)
            equal = lower_or_equal - lower
            greater = later_count - lower_or_equal
            comparable += later_count
            concordant += greater + 0.5 * equal

        for ordered_index in order[position:group_end]:
            update(int(ranks[ordered_index]))
            later_count += 1

        position = group_end

    return concordant / comparable if comparable else float("nan")
