from am_mvt.ingestion.literature_manifest import article_number_from_filename


def test_article_number_from_repository_filename() -> None:
    assert article_number_from_filename(
        "011_addma_2019_316l_fatigue_orientation_surface_roughness.pdf"
    ) == "011"
    assert article_number_from_filename("paper_without_number.pdf") == ""
