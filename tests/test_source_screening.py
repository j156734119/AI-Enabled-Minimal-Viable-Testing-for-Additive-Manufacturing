import pandas as pd

from am_mvt.ingestion.llm_source_screening import (
    MEETING_ONE_JOURNAL_SCOPE,
    is_crossref_title_relevant,
    normalise_score,
    select_balanced_candidates,
)


def test_normalise_score_uses_consistent_zero_to_ten_scale():
    assert normalise_score(0.9) == 9.0
    assert normalise_score(9.0) == 9.0
    assert normalise_score(15.0) == 10.0


def test_balanced_selection_reserves_each_journal():
    rows = []

    for scope in MEETING_ONE_JOURNAL_SCOPE:
        for index in range(3):
            rows.append(
                {
                    "journal": scope.journal,
                    "selection_score": 10 - index,
                }
            )

    selected = select_balanced_candidates(
        pd.DataFrame(rows),
        target_count=16,
        min_per_journal=2,
    )

    assert len(selected) == 16
    assert selected.groupby("journal").size().min() == 2


def test_crossref_title_filter_rejects_reviews_and_non_am_welding():
    assert is_crossref_title_relevant(
        "Fatigue of Ti-6Al-4V produced by laser powder bed fusion"
    )
    assert not is_crossref_title_relevant(
        "Porosity and tensile strength after gas metal arc welding"
    )
    assert not is_crossref_title_relevant(
        "Review of metal wire arc additive manufacturing"
    )
