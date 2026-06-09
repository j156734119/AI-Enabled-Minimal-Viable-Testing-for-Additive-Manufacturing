from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from am_mvt.config import get_path


PDF_BY_DOI = {
    "10.1108/RPJ-02-2022-0041": (
        "049_rpj_2023_ti_6al_4v_determination_elastoplastic_alloy_"
        "electron_beam_melting.pdf"
    ),
    "10.1108/RPJ-11-2021-0325": (
        "050_rpj_2022_alsi10mg_building_orientations_heat_treatments_"
        "alloy_fabricated_selective_laser_melting.pdf"
    ),
    "10.1108/RPJ-05-2016-0071": (
        "051_rpj_2017_inconel_718_role_process_parameters_during_"
        "direct_metal_deposition.pdf"
    ),
    "10.1108/RPJ-06-2023-0204": (
        "052_rpj_2024_extrusion_based_3d_printing_technology_"
        "investigate_impact_changing_print_conditions.pdf"
    ),
}

COLUMN_MAP = {
    "publication_year": "source_year",
    "energy_density_J_mm3": "ved_J_mm3",
    "stress_ratio_R": "r_ratio",
    "ultimate_tensile_strength_MPa": "uts_MPa",
    "elastic_modulus_GPa": "youngs_modulus_GPa",
    "hardness": "hardness_HV",
}

REQUIRED_EVIDENCE = [
    "source_title",
    "doi",
    "page_or_section",
    "evidence_text",
    "confidence",
    "needs_human_check",
]


def clean_value(value: Any) -> Any:
    if pd.isna(value) or str(value).strip() == "":
        return None
    if isinstance(value, str):
        return value.strip()
    return value


def normalise_confidence(value: Any) -> float:
    labels = {"high": 0.9, "medium": 0.7, "low": 0.5}
    text = str(value).strip().lower()
    if text in labels:
        return labels[text]
    confidence = float(value)
    if not 0 <= confidence <= 1:
        raise ValueError(f"Confidence must be in [0, 1]: {value}")
    return confidence


def convert_row(row: pd.Series) -> dict[str, Any]:
    record = {
        COLUMN_MAP.get(column, column): clean_value(value)
        for column, value in row.items()
    }
    record["confidence"] = normalise_confidence(row["confidence"])
    if record.get("hatch_spacing_mm") is not None:
        record["hatch_spacing_um"] = float(record.pop("hatch_spacing_mm")) * 1000
    else:
        record.pop("hatch_spacing_mm", None)
    if record.get("layer_thickness_mm") is not None:
        record["layer_thickness_um"] = (
            float(record.pop("layer_thickness_mm")) * 1000
        )
    else:
        record.pop("layer_thickness_mm", None)
    record["runout"] = None
    return record


def import_manual_csv(
    input_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    input_path = Path(input_path)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_path("data", "interim", "llm_outputs")
    )
    frame = pd.read_csv(input_path, low_memory=False)
    missing_columns = set(REQUIRED_EVIDENCE) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "Manual extraction CSV is missing: "
            + ", ".join(sorted(missing_columns))
        )
    for column in REQUIRED_EVIDENCE:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Manual extraction evidence field is empty: {column}")

    unknown_dois = sorted(set(frame["doi"].astype(str)) - set(PDF_BY_DOI))
    if unknown_dois:
        raise ValueError("No source PDF mapping for: " + ", ".join(unknown_dois))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for doi, group in frame.groupby("doi", sort=False):
        source_file = PDF_BY_DOI[str(doi)]
        chunk_id = f"{Path(source_file).stem}_chunk_0000"
        payload = {
            "records": [convert_row(row) for _, row in group.iterrows()],
            "_metadata": {
                "source_file": source_file,
                "chunk_id": chunk_id,
                "model": "chatgpt_manual_vision",
                "input_mode": "manual_pdf_vision_csv",
                "input_csv": input_path.name,
            },
        }
        output_path = output_dir / f"{chunk_id}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs.append(output_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import evidence-grounded manual PDF vision extraction CSV."
    )
    parser.add_argument("input_csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = import_manual_csv(args.input_csv)
    print(f"Imported manual extraction files: {len(outputs)}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
