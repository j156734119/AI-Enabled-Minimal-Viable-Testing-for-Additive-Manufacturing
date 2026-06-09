from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from am_mvt.config import get_path
from am_mvt.skill_loader import build_skill_system_prompt
from am_mvt.utils.openai import extract_output_text, get_openai_client
from am_mvt.utils.text import normalise_doi


DEFAULT_SCREENING_MODEL = os.getenv("OPENAI_SCREENING_MODEL", "gpt-4o-mini")


@dataclass(frozen=True)
class JournalScope:
    journal: str
    priority_tier: str
    publisher_domain: str


MEETING_ONE_JOURNAL_SCOPE: list[JournalScope] = [
    JournalScope("Additive Manufacturing", "Tier 1", "www.sciencedirect.com"),
    JournalScope(
        "Journal of Materials Processing Technology",
        "Tier 1",
        "www.sciencedirect.com",
    ),
    JournalScope("Journal of Manufacturing Processes", "Tier 1", "www.sciencedirect.com"),
    JournalScope("Rapid Prototyping Journal", "Tier 1", "www.emerald.com"),
    JournalScope("Metals", "Tier 1", "www.mdpi.com"),
    JournalScope("Virtual and Physical Prototyping", "Tier 2", "www.tandfonline.com"),
    JournalScope("Progress in Additive Manufacturing", "Tier 2", "link.springer.com"),
    JournalScope("Advanced Engineering Materials", "Tier 3", "onlinelibrary.wiley.com"),
]


JOURNAL_ALIASES: dict[str, str] = {
    "additive manufacturing": "Additive Manufacturing",
    "addit manuf": "Additive Manufacturing",
    "journal of materials processing technology": "Journal of Materials Processing Technology",
    "j mater process technol": "Journal of Materials Processing Technology",
    "journal of manufacturing processes": "Journal of Manufacturing Processes",
    "j manuf process": "Journal of Manufacturing Processes",
    "rapid prototyping journal": "Rapid Prototyping Journal",
    "rapid prototyping j": "Rapid Prototyping Journal",
    "metals": "Metals",
    "metals basel": "Metals",
    "metals (basel)": "Metals",
    "virtual and physical prototyping": "Virtual and Physical Prototyping",
    "virtual & physical prototyping": "Virtual and Physical Prototyping",
    "progress in additive manufacturing": "Progress in Additive Manufacturing",
    "advanced engineering materials": "Advanced Engineering Materials",
    "adv eng mater": "Advanced Engineering Materials",
}


SEARCH_FOCUS_AREAS: list[str] = [
    (
        "static tensile data: yield strength, ultimate tensile strength, "
        "elongation to failure, elastic modulus, and processing "
        "parameters for metal AM"
    ),
    (
        "fatigue data: S-N curves, stress amplitude, fatigue life cycles, "
        "runout, surface condition, defects, porosity, and build orientation"
    ),
    (
        "process-structure-property data: LPBF, DED, WAAM, binder jetting, "
        "heat treatment, post-processing, porosity, and "
        "mechanical property tables"
    ),
]


JOURNAL_SEARCH_HINTS = {
    "Metals": (
        "Use the Metals journal article path site:mdpi.com/2075-4701 and the "
        "Metals Additive Manufacturing section. Do not return papers from "
        "other MDPI journals such as Materials, Polymers, or Applied Sciences."
    ),
}


AccessType = Literal[
    "open_access",
    "public_supplementary",
    "manual_download_required",
    "university_subscription",
    "uncertain",
]


class SourceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    journal: str
    year: int | None
    doi: str | None
    url: str | None
    pdf_url: str | None
    access_type: AccessType
    priority_tier: str
    relevance_score: float
    data_richness_score: float
    impact_score: float
    selection_score: float
    relevance_reason: str
    expected_extractable_data: str
    manual_action: str
    source_evidence_url: str | None
    notes: str


class SourceScreeningResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[SourceCandidate]


SOURCE_SCREENING_SCHEMA = SourceScreeningResponse.model_json_schema()


SCREENING_SYSTEM_PROMPT = """
You are a cautious source-screening assistant for an MSc dissertation on
AI-enabled minimal viable testing for metal additive manufacturing.

Screen candidate papers only. Do not claim that a PDF has been downloaded.
Do not use or request university credentials, publisher credentials, cookies,
VPN sessions, passwords, or institutional access tokens.

Prefer papers likely to contain extractable original mechanical testing data:
process parameters, material/alloy, build orientation, surface condition,
heat treatment, porosity/defect metrics, tensile properties, elastic modulus,
fatigue stress-life data, or S-N fatigue life.

Exclude generic mechanical-property prediction competitions and keep the focus
on agent-assisted, evidence-grounded reduced but representative mechanical
testing for additive manufacturing.

Use conservative language. If access is uncertain, mark it as
manual_download_required or uncertain.
"""


def get_client() -> OpenAI:
    return get_openai_client()


def make_source_id(title: str, year: int | None) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", title.lower())[:8]
    base = "_".join(words) or "candidate_source"

    if year:
        return f"{base}_{year}"

    return base


def normalise_journal_name(journal: str) -> str | None:
    cleaned = re.sub(r"[^a-zA-Z0-9&() ]+", " ", journal).lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in JOURNAL_ALIASES:
        return JOURNAL_ALIASES[cleaned]

    without_parentheses = cleaned.replace("(", "").replace(")", "")

    if without_parentheses in JOURNAL_ALIASES:
        return JOURNAL_ALIASES[without_parentheses]

    return None


def build_search_prompt(
    journal_scope: JournalScope,
    per_journal_limit: int,
    year_from: int,
    year_to: int,
    focus_area: str,
) -> str:
    search_hint = JOURNAL_SEARCH_HINTS.get(
        journal_scope.journal,
        f"Prefer results from site:{journal_scope.publisher_domain}.",
    )
    return f"""
Search the web for candidate original research papers from this journal only:

Journal: {journal_scope.journal}
Priority tier: {journal_scope.priority_tier}
Preferred publisher domain: {journal_scope.publisher_domain}
Year range: {year_from}-{year_to}
Search focus: {focus_area}
Journal-specific search instruction: {search_hint}

Return up to {per_journal_limit} candidate papers that are highly relevant to:
- metal additive manufacturing
- tensile properties, fatigue life, S-N fatigue data, elastic modulus,
  elongation to failure, yield strength, or UTS
- process parameters, porosity, defects, surface condition, heat treatment,
  post-processing, build orientation, or fatigue loading conditions

Rank papers higher when they are likely to contain extractable numerical data,
tables, supplementary datasets, or clear experimental condition-property pairs.
Prioritise papers that can support at least one of these current modelling
targets: UTS, S-N fatigue life, elongation/yield response, or elastic/Young's
modulus. Do not prioritise hardness, residual stress, or failure-mode-only
papers because those variables are future extensions rather than current
modelling targets.

Only include papers from the specified journal. Do not include review-only
papers unless they provide reusable public datasets. Do not include sources
outside the specified journal scope.

Use a consistent 0-10 scale for every score. A score of 0 means unsuitable and
10 means exceptionally relevant/data-rich/high-impact. Do not use a 0-1 scale.

Return JSON matching the schema.
"""


def screen_one_journal(
    client: OpenAI,
    journal_scope: JournalScope,
    per_journal_limit: int,
    year_from: int,
    year_to: int,
    model: str,
    focus_area: str,
    retry_count: int = 2,
) -> list[dict[str, Any]]:
    prompt = build_search_prompt(
        journal_scope=journal_scope,
        per_journal_limit=per_journal_limit,
        year_from=year_from,
        year_to=year_to,
        focus_area=focus_area,
    )
    system_prompt = build_skill_system_prompt(
        SCREENING_SYSTEM_PROMPT,
        "source-screening",
    )

    last_error: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tools=[{"type": "web_search"}],
                tool_choice="auto",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "am_source_screening_candidates",
                        "strict": True,
                        "schema": SOURCE_SCREENING_SCHEMA,
                    }
                },
                temperature=0,
            )

            parsed = SourceScreeningResponse.model_validate_json(
                extract_output_text(response)
            )
            candidates = [
                candidate.model_dump(mode="json")
                for candidate in parsed.candidates
            ]
            for candidate in candidates:
                candidate["_requested_journal"] = journal_scope.journal
            return candidates

        except Exception as exc:
            last_error = exc

            if attempt < retry_count:
                time.sleep(2 * attempt)

    return [
        {
            "title": f"SCREENING_FAILED: {journal_scope.journal}",
            "journal": journal_scope.journal,
            "year": None,
            "doi": None,
            "url": None,
            "pdf_url": None,
            "access_type": "uncertain",
            "priority_tier": journal_scope.priority_tier,
            "relevance_score": 0.0,
            "data_richness_score": 0.0,
            "impact_score": 0.0,
            "selection_score": 0.0,
            "relevance_reason": "Screening failed and should be rerun manually.",
            "expected_extractable_data": "",
            "manual_action": "Rerun source screening or search this journal manually.",
            "source_evidence_url": None,
            "notes": str(last_error),
        }
    ]


def normalise_candidates(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    approved_journals = {
        item.journal.lower(): item.priority_tier for item in MEETING_ONE_JOURNAL_SCOPE
    }
    rows: list[dict[str, Any]] = []

    for row in candidates:
        title = str(row.get("title") or "").strip()
        journal = str(row.get("journal") or "").strip()
        canonical_journal = normalise_journal_name(journal)

        if not title or title.startswith("SCREENING_FAILED"):
            continue

        if canonical_journal is None:
            continue

        requested_journal = str(row.get("_requested_journal") or "").strip()

        if requested_journal and canonical_journal != requested_journal:
            continue

        year = row.get("year")
        source_id = make_source_id(title=title, year=year if isinstance(year, int) else None)

        row = {
            "source_id": source_id,
            "title": title,
            "journal": canonical_journal,
            "year": year,
            "doi": row.get("doi"),
            "url": row.get("url"),
            "pdf_url": row.get("pdf_url"),
            "access_type": row.get("access_type", "uncertain"),
            "priority_tier": approved_journals[canonical_journal.lower()],
            "relevance_score": normalise_score(row.get("relevance_score", 0.0)),
            "data_richness_score": normalise_score(
                row.get("data_richness_score", 0.0)
            ),
            "impact_score": normalise_score(row.get("impact_score", 0.0)),
            "selection_score": normalise_score(row.get("selection_score", 0.0)),
            "relevance_reason": row.get("relevance_reason", ""),
            "expected_extractable_data": row.get("expected_extractable_data", ""),
            "source_evidence_url": row.get("source_evidence_url"),
            "local_pdf_filename": "",
            "download_status": "not_downloaded",
            "manual_action": row.get("manual_action", ""),
            "notes": row.get("notes", ""),
            "journal_scope_verified": True,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["_normalised_doi"] = df["doi"].map(normalise_doi).replace("", pd.NA)
    df["_normalised_title"] = (
        df["title"]
        .astype("string")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )
    df["_dedupe_key"] = df["_normalised_doi"].fillna(
        "title:" + df["_normalised_title"]
    )
    df = df.drop_duplicates(subset=["_dedupe_key"], keep="first")
    df = df.drop(
        columns=["_normalised_doi", "_normalised_title", "_dedupe_key"]
    )

    if "selection_score" in df.columns:
        df = df.sort_values(
            by=["selection_score", "data_richness_score", "impact_score"],
            ascending=False,
        )

    return df.reset_index(drop=True)


def normalise_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if 0.0 <= score <= 1.0:
        score *= 10.0

    return max(0.0, min(10.0, score))


def select_balanced_candidates(
    df: pd.DataFrame,
    target_count: int,
    min_per_journal: int,
) -> pd.DataFrame:
    if df.empty:
        return df

    selected_indices: list[int] = []

    for scope in MEETING_ONE_JOURNAL_SCOPE:
        journal_rows = df.loc[df["journal"].eq(scope.journal)]
        selected_indices.extend(journal_rows.head(min_per_journal).index.tolist())

    selected_indices = list(dict.fromkeys(selected_indices))
    remaining_slots = max(0, target_count - len(selected_indices))
    remaining = df.loc[~df.index.isin(selected_indices)].head(remaining_slots)
    selected = pd.concat(
        [df.loc[selected_indices], remaining],
        ignore_index=True,
        sort=False,
    )

    return selected.head(target_count).reset_index(drop=True)


def archive_existing_source_screening_outputs(
    output_paths: dict[str, str],
    timestamp: str | None = None,
) -> Path | None:
    existing_paths = [
        Path(path)
        for path in output_paths.values()
        if Path(path).is_file()
    ]
    if not existing_paths:
        return None

    archive_timestamp = timestamp or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    archive_dir = get_path("archive", "source_search_runs", archive_timestamp)
    if archive_dir.exists():
        raise FileExistsError(f"Source-search archive already exists: {archive_dir}")

    for source_path in existing_paths:
        relative_path = source_path.relative_to(get_path())
        destination = archive_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    return archive_dir


def write_source_screening_outputs(
    df: pd.DataFrame,
    timestamp: str | None = None,
) -> dict[str, str]:
    output_paths = {
        "interim": str(get_path("data", "interim", "candidate_sources_llm.csv")),
        "table": str(
            get_path("outputs", "tables", "source_screening_candidates_top50.csv")
        ),
        "journal_scope": str(
            get_path("outputs", "tables", "source_screening_journal_scope.csv")
        ),
    }
    scope_df = pd.DataFrame([asdict(item) for item in MEETING_ONE_JOURNAL_SCOPE])
    output_frames = {
        "interim": df,
        "table": df,
        "journal_scope": scope_df,
    }
    temporary_paths: dict[str, Path] = {}

    try:
        for key, path_text in output_paths.items():
            path = Path(path_text)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_name(f".{path.name}.tmp")
            output_frames[key].to_csv(
                temporary_path,
                index=False,
                encoding="utf-8-sig",
            )
            temporary_paths[key] = temporary_path

        archive_existing_source_screening_outputs(
            output_paths=output_paths,
            timestamp=timestamp,
        )

        for key, path_text in output_paths.items():
            os.replace(temporary_paths[key], path_text)
    finally:
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()

    return output_paths


def run_openai_agent_source_screening(
    target_count: int = 50,
    per_journal_limit: int = 8,
    min_per_journal: int = 1,
    year_from: int = 2015,
    year_to: int = 2026,
    model: str = DEFAULT_SCREENING_MODEL,
    search_rounds: int = 3,
) -> tuple[pd.DataFrame, dict[str, str]]:
    client = get_client()
    all_candidates: list[dict[str, Any]] = []
    focus_areas = SEARCH_FOCUS_AREAS[: max(1, search_rounds)]

    for focus_index, focus_area in enumerate(focus_areas, start=1):
        print(f"Search round {focus_index}/{len(focus_areas)}")

        for journal_scope in MEETING_ONE_JOURNAL_SCOPE:
            print(
                f"Screening {journal_scope.journal} "
                f"({journal_scope.priority_tier})..."
            )
            all_candidates.extend(
                screen_one_journal(
                    client=client,
                    journal_scope=journal_scope,
                    per_journal_limit=per_journal_limit,
                    year_from=year_from,
                    year_to=year_to,
                    model=model,
                    focus_area=focus_area,
                )
            )

    df = normalise_candidates(all_candidates)

    if not df.empty:
        journal_counts = df["journal"].value_counts()
        broad_focus = (
            "broad metal additive manufacturing experimental mechanical "
            "property data, including tensile, fatigue, modulus, porosity, "
            "surface condition, orientation, and post-processing"
        )

        for journal_scope in MEETING_ONE_JOURNAL_SCOPE:
            current_count = int(journal_counts.get(journal_scope.journal, 0))

            if current_count >= min_per_journal:
                continue

            print(
                f"Replenishing {journal_scope.journal}: "
                f"{current_count}/{min_per_journal} candidates..."
            )
            all_candidates.extend(
                screen_one_journal(
                    client=client,
                    journal_scope=journal_scope,
                    per_journal_limit=max(per_journal_limit, min_per_journal * 2),
                    year_from=year_from,
                    year_to=year_to,
                    model=model,
                    focus_area=broad_focus,
                )
            )

        df = normalise_candidates(all_candidates)
        df = select_balanced_candidates(
            df=df,
            target_count=target_count,
            min_per_journal=min_per_journal,
        )

    if df.empty:
        raise RuntimeError(
            "OpenAI agent source screening returned no valid candidates. "
            "Existing Step 01 outputs were left unchanged."
        )

    covered_journals = set(df["journal"].astype(str))
    missing_journals = [
        scope.journal
        for scope in MEETING_ONE_JOURNAL_SCOPE
        if scope.journal not in covered_journals
    ]
    if missing_journals:
        print(
            "Warning: no valid candidate was returned for: "
            + ", ".join(missing_journals)
        )

    output_paths = write_source_screening_outputs(df)
    return df, output_paths
