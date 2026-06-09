from __future__ import annotations

import json

import pandas as pd

from scripts.import_manual_extraction_csv import import_manual_csv


def test_manual_import_maps_units_and_provenance(tmp_path):
    input_path = tmp_path / "manual.csv"
    output_dir = tmp_path / "outputs"
    pd.DataFrame(
        [
            {
                "source_title": "Paper",
                "doi": "10.1108/RPJ-02-2022-0041",
                "publication_year": 2023,
                "journal": "Rapid Prototyping Journal",
                "page_or_section": "p.1",
                "table_or_figure": "Table 1",
                "evidence_text": "UTS=900 MPa",
                "alloy": "Ti-6Al-4V",
                "alloy_family": "titanium alloy",
                "am_process": "electron beam melting",
                "hatch_spacing_mm": 0.1,
                "layer_thickness_mm": 0.05,
                "ultimate_tensile_strength_MPa": 900,
                "confidence": "high",
                "needs_human_check": False,
            }
        ]
    ).to_csv(input_path, index=False)

    outputs = import_manual_csv(input_path, output_dir)
    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    record = payload["records"][0]

    assert record["hatch_spacing_um"] == 100
    assert record["layer_thickness_um"] == 50
    assert record["uts_MPa"] == 900
    assert record["confidence"] == 0.9
    assert payload["_metadata"]["input_mode"] == "manual_pdf_vision_csv"
