from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from am_mvt.config import get_path


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    source_name: str
    doi: str
    title: str
    year: int | None
    source_type: str
    url: str
    licence: str
    notes: str


CORE_DATASET_SOURCES: list[DatasetSource] = [
    DatasetSource(
        source_id="fatigue_am_alloys_figshare_2023",
        source_name="Fatigue Database of Additively Manufactured Alloys",
        doi="10.6084/m9.figshare.22337629",
        title="Fatigue Database of Additively Manufactured Alloys",
        year=2023,
        source_type="public_dataset",
        url="https://doi.org/10.6084/m9.figshare.22337629",
        licence="Check original dataset licence before redistribution",
        notes=(
            "Literature-derived public dataset for fatigue and mechanical "
            "properties of additively manufactured alloys."
        ),
    ),
    DatasetSource(
        source_id="materials_design_statistical_assessment_2025",
        source_name="Critical statistical assessment of data in metal additive manufacturing",
        doi="10.1016/j.matdes.2025.114301",
        title="Critical statistical assessment of data in metal additive manufacturing",
        year=2025,
        source_type="public_dataset_or_literature_dataset",
        url="https://doi.org/10.1016/j.matdes.2025.114301",
        licence="Check original article and dataset licence before redistribution",
        notes=(
            "Literature-derived metal additive manufacturing dataset associated "
            "with a statistical assessment study."
        ),
    ),
]


def get_core_dataset_sources() -> list[dict[str, object]]:
    return [asdict(source) for source in CORE_DATASET_SOURCES]


def write_core_sources_csv(output_path: str | Path | None = None) -> Path:
    if output_path is None:
        output_path = get_path("data", "raw", "metadata", "candidate_sources.csv")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(get_core_dataset_sources())
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path