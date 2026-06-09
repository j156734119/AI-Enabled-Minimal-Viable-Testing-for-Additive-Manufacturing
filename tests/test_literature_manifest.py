import pandas as pd

from am_mvt.ingestion.literature_manifest import (
    MANIFEST_COLUMNS,
    article_number_from_filename,
    mark_duplicate_sources,
)


def test_article_number_from_repository_filename() -> None:
    assert article_number_from_filename(
        "011_addma_2019_316l_fatigue_orientation_surface_roughness.pdf"
    ) == "011"
    assert article_number_from_filename("paper_without_number.pdf") == ""


def test_duplicate_sources_use_lowest_article_number_as_canonical() -> None:
    rows = []
    for number, source_id, journal in [
        ("039", "duplicate_source", "Wrong Journal"),
        ("004", "canonical_source", "Progress in Additive Manufacturing"),
    ]:
        row = {column: "" for column in MANIFEST_COLUMNS}
        row.update(
            {
                "article_number": number,
                "source_id": source_id,
                "title": "Same paper",
                "journal": journal,
                "year": 2022,
                "doi": "https://doi.org/10.1000/example",
                "local_pdf_filename": f"{number}.pdf",
                "content_sha256": "a" * 64,
                "canonical_source_id": source_id,
                "processing_status": "canonical",
                "ready_for_parsing": True,
            }
        )
        rows.append(row)

    result = mark_duplicate_sources(pd.DataFrame(rows))
    canonical = result.loc[result["article_number"].eq("004")].iloc[0]
    duplicate = result.loc[result["article_number"].eq("039")].iloc[0]

    assert canonical["processing_status"] == "canonical"
    assert duplicate["processing_status"] == "duplicate"
    assert duplicate["duplicate_of"] == "canonical_source"
    assert not bool(duplicate["ready_for_parsing"])
    assert "journal" in duplicate["notes"]


def test_conflicting_doi_metadata_is_not_merged() -> None:
    rows = []
    for number, source_id, title, digest in [
        ("050", "paper_a", "Effect of building orientations", "a" * 64),
        ("052", "paper_b", "Extrusion printing conditions", "b" * 64),
    ]:
        row = {column: "" for column in MANIFEST_COLUMNS}
        row.update(
            {
                "article_number": number,
                "source_id": source_id,
                "title": title,
                "year": 2022 if number == "050" else 2024,
                "doi": "10.1000/conflict",
                "local_pdf_filename": f"{number}.pdf",
                "content_sha256": digest,
                "canonical_source_id": source_id,
                "processing_status": "canonical",
                "ready_for_parsing": True,
            }
        )
        rows.append(row)

    result = mark_duplicate_sources(pd.DataFrame(rows))
    conflict = result.loc[result["article_number"].eq("052")].iloc[0]

    assert conflict["processing_status"] == "metadata_conflict"
    assert conflict["duplicate_of"] == ""
    assert not bool(conflict["ready_for_parsing"])
