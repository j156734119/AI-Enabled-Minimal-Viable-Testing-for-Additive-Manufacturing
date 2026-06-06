from pathlib import Path

from am_mvt.extraction.batch_extract import infer_source_pdf_from_chunk_name
from am_mvt.extraction.postprocess import (
    normalise_pdf_filename,
    source_id_from_pdf_filename,
)


def test_pdf_filename_does_not_gain_double_extension():
    assert (
        infer_source_pdf_from_chunk_name(
            Path("002_paper.pdf_chunk_0001.txt")
        )
        == "002_paper.pdf"
    )
    assert normalise_pdf_filename("002_paper.pdf.pdf") == "002_paper.pdf"


def test_source_id_is_derived_from_pdf_filename():
    assert source_id_from_pdf_filename("02--Paper Title.pdf") == "02_paper_title"
