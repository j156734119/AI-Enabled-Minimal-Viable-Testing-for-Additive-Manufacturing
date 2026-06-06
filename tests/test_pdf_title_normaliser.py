from am_mvt.ingestion.pdf_title_normaliser import (
    compact_title_slug,
    infer_journal_from_filename,
    infer_journal_from_text,
    journal_code,
    normalise_doi,
    normalise_title,
    sanitise_windows_filename,
    title_similarity,
)


def test_numbered_title_prefix_is_removed() -> None:
    assert normalise_title("02——A Study of LPBF Fatigue.pdf") == (
        "a study of lpbf fatigue"
    )


def test_title_similarity_handles_number_and_double_extension() -> None:
    score = title_similarity(
        "Effect of Build Orientation on Ti-6Al-4V",
        "002_Effect of Build Orientation on Ti-6Al-4V.pdf.pdf",
    )

    assert score == 1.0


def test_doi_and_windows_filename_are_normalised() -> None:
    assert normalise_doi("https://doi.org/10.1016/j.addma.2022.102661.") == (
        "10.1016/j.addma.2022.102661"
    )
    assert sanitise_windows_filename('A title: with / invalid * characters') == (
        "A title with invalid characters"
    )


def test_compact_repository_filename_components() -> None:
    assert journal_code("Additive Manufacturing") == "addma"
    assert infer_journal_from_text(
        "Additive Manufacturing 25 (2019) 6-15 journal homepage"
    ) == "additive manufacturing"
    assert compact_title_slug(
        "Fatigue behavior of additive manufactured 316L stainless steel parts: "
        "Effects of layer orientation and surface roughness"
    ) == "316l_fatigue_orientation_surface_roughness"


def test_journal_is_recovered_from_repository_filename() -> None:
    assert (
        infer_journal_from_filename("021_rpj_2022_example.pdf")
        == "Rapid Prototyping Journal"
    )
    assert infer_journal_from_filename("022_metals_2023_example.pdf") == "Metals"


def test_generic_process_phrase_is_not_misread_as_journal() -> None:
    assert infer_journal_from_text(
        "This study concerns additive manufacturing of a metal alloy."
    ) == ""
