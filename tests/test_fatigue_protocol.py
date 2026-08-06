from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from am_mvt.modelling.basquin import normalise_runout
from am_mvt.modelling.experiment_training import (
    aft_monotone_constraints,
    make_aft_bounds,
)
from am_mvt.modelling.experiment_data import build_evaluation_groups
from am_mvt.modelling.fatigue_protocol import (
    aft_life_quantile,
    aft_survival_probability,
    e606_assessment,
    protocolise_fatigue_data,
    threshold_labels,
)
from am_mvt.optimisation.client_target_matrix import E466_STRESS_LEVELS_MPA


def test_runout_variants_are_normalised_without_guessing_unknown_values():
    values = pd.Series(
        [
            "true",
            "1",
            "yes",
            "runout",
            "run-out",
            "run-out samples",
            "false",
            "failure",
            "unknown",
        ]
    )

    result = normalise_runout(values)

    assert result.iloc[:6].eq(True).all()
    assert result.iloc[6:8].eq(False).all()
    assert pd.isna(result.iloc[8])


def test_protocol_routing_separates_conventional_ultrasonic_and_ambiguous():
    frame = pd.DataFrame(
        {
            "frequency_Hz": [50.0, 200.0, 500.0, 20_000.0, np.nan],
            "runout": [False] * 5,
            "fatigue_life_cycles": [1_000_000.0] * 5,
            "stress_amplitude_MPa": [110.0] * 5,
        }
    )

    result = protocolise_fatigue_data(frame)

    assert result["fatigue_protocol"].tolist() == [
        "e466_conventional",
        "e466_conventional",
        "ambiguous_frequency",
        "ultrasonic_vhcf",
        "ambiguous_frequency",
    ]


def test_fatigue_grouping_prefers_dataset_id_over_shared_doi():
    frame = pd.DataFrame(
        {
            "doi": ["10.1/shared", "10.1/shared"],
            "dataset_id": ["sn_curve_a", "sn_curve_b"],
            "source_id": ["source", "source"],
        }
    )

    groups = build_evaluation_groups(frame, prefer_dataset=True)

    assert groups.tolist() == ["dataset:sn_curve_a", "dataset:sn_curve_b"]


def test_stress_definition_inconsistency_is_flagged_for_review():
    frame = pd.DataFrame(
        {
            "frequency_Hz": [50.0],
            "stress_amplitude_MPa": [110.0],
            "max_stress_MPa": [110.0],
            "r_ratio": [0.0],
            "fatigue_life_cycles": [1_000_000.0],
            "runout": [False],
        }
    )

    result = protocolise_fatigue_data(frame)

    assert result.loc[0, "stress_consistency_status"] == "review_required"


def test_unknown_runout_is_rejected_by_aft_bounds():
    frame = pd.DataFrame(
        {
            "fatigue_life_cycles": [1_000.0, 2_000.0],
            "runout": [False, "unknown"],
        }
    )

    with pytest.raises(ValueError, match="explicit failure or runout"):
        make_aft_bounds(frame)


def test_threshold_labels_respect_right_censoring():
    frame = protocolise_fatigue_data(
        pd.DataFrame(
            {
                "frequency_Hz": [50.0] * 4,
                "stress_amplitude_MPa": [110.0] * 4,
                "fatigue_life_cycles": [5e6, 15e6, 5e6, 15e6],
                "runout": [False, False, True, True],
            }
        )
    )

    labels = threshold_labels(frame, 10_000_000.0)

    assert labels.iloc[0] == 0.0
    assert labels.iloc[1] == 1.0
    assert pd.isna(labels.iloc[2])
    assert labels.iloc[3] == 1.0


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
def test_aft_probabilities_and_quantiles_are_ordered(distribution: str):
    locations = np.asarray([20_000_000.0, 10_000_000.0, 5_000_000.0])
    p10m = aft_survival_probability(locations, 10_000_000.0, distribution, 1.0)
    p20m = aft_survival_probability(locations, 20_000_000.0, distribution, 1.0)
    q10 = aft_life_quantile(locations, 0.10, distribution, 1.0)
    q90 = aft_life_quantile(locations, 0.90, distribution, 1.0)

    assert np.all(p20m <= p10m)
    assert np.all(np.diff(p10m) <= 0)
    assert np.all(q10 <= q90)


def test_stress_feature_has_negative_monotonic_constraint():
    constraints = aft_monotone_constraints(
        5,
        ["r_ratio", "log10_stress_amplitude"],
    )

    assert constraints == "(0,-1,0,0,0)"


def test_e606_without_strain_controlled_fields_is_not_assessable():
    frame = pd.DataFrame({"stress_amplitude_MPa": [110.0]})

    assert e606_assessment(frame) == ("not_assessable_missing_strain_controlled_data")


def test_e466_matrix_levels_include_110_mpa():
    assert E466_STRESS_LEVELS_MPA == [80.0, 95.0, 110.0, 125.0, 140.0]
