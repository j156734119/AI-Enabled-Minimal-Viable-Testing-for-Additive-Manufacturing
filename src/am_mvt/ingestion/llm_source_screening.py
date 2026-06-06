from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from am_mvt.config import get_path


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
        "elongation to failure, elastic modulus, hardness, and processing "
        "parameters for metal AM"
    ),
    (
        "fatigue data: S-N curves, stress amplitude, fatigue life cycles, "
        "runout, surface condition, defects, porosity, and build orientation"
    ),
    (
        "process-structure-property data: LPBF, DED, WAAM, binder jetting, "
        "heat treatment, post-processing, porosity, residual stress, and "
        "mechanical property tables"
    ),
]


SOURCE_SCREENING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "journal": {"type": "string"},
                    "year": {"type": ["integer", "null"]},
                    "doi": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                    "pdf_url": {"type": ["string", "null"]},
                    "access_type": {
                        "type": "string",
                        "enum": [
                            "open_access",
                            "public_supplementary",
                            "manual_download_required",
                            "university_subscription",
                            "uncertain",
                        ],
                    },
                    "priority_tier": {"type": "string"},
                    "relevance_score": {"type": "number"},
                    "data_richness_score": {"type": "number"},
                    "impact_score": {"type": "number"},
                    "selection_score": {"type": "number"},
                    "relevance_reason": {"type": "string"},
                    "expected_extractable_data": {"type": "string"},
                    "manual_action": {"type": "string"},
                    "source_evidence_url": {"type": ["string", "null"]},
                    "notes": {"type": "string"},
                },
                "required": [
                    "title",
                    "journal",
                    "year",
                    "doi",
                    "url",
                    "pdf_url",
                    "access_type",
                    "priority_tier",
                    "relevance_score",
                    "data_richness_score",
                    "impact_score",
                    "selection_score",
                    "relevance_reason",
                    "expected_extractable_data",
                    "manual_action",
                    "source_evidence_url",
                    "notes",
                ],
            },
        }
    },
    "required": ["candidates"],
}


SCREENING_SYSTEM_PROMPT = """
You are a cautious source-screening assistant for an MSc dissertation on
AI-enabled minimal viable testing for metal additive manufacturing.

Screen candidate papers only. Do not claim that a PDF has been downloaded.
Do not use or request university credentials, publisher credentials, cookies,
VPN sessions, passwords, or institutional access tokens.

Prefer papers likely to contain extractable original mechanical testing data:
process parameters, material/alloy, build orientation, surface condition,
heat treatment, porosity/defect metrics, tensile properties, hardness,
elastic modulus, fatigue stress-life data, or S-N fatigue life.

Exclude generic mechanical-property prediction competitions and keep the focus
on agent-assisted, evidence-grounded reduced but representative mechanical
testing for additive manufacturing.

Use conservative language. If access is uncertain, mark it as
manual_download_required or uncertain.
"""


def load_project_env() -> None:
    env_path = get_path(".env")

    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_client() -> OpenAI:
    load_project_env()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to the project .env file before "
            "running web source screening."
        )

    return OpenAI()


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)

    if output_text:
        return output_text

    parts: list[str] = []

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)

            if text:
                parts.append(text)

    return "\n".join(parts)


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
    return f"""
Search the web for candidate original research papers from this journal only:

Journal: {journal_scope.journal}
Priority tier: {journal_scope.priority_tier}
Preferred publisher domain: {journal_scope.publisher_domain}
Year range: {year_from}-{year_to}
Search focus: {focus_area}

Return up to {per_journal_limit} candidate papers that are highly relevant to:
- metal additive manufacturing
- tensile properties, fatigue life, S-N fatigue data, hardness, elastic modulus,
  elongation to failure, yield strength, or UTS
- process parameters, porosity, defects, surface condition, heat treatment,
  post-processing, build orientation, or fatigue loading conditions

Rank papers higher when they are likely to contain extractable numerical data,
tables, supplementary datasets, or clear experimental condition-property pairs.
Prioritise papers that can support at least one of these current modelling
targets: UTS, S-N fatigue life, elongation/yield response, or elastic/Young's
modulus. Hardness and failure-mode labels are useful secondary fields.

Only include papers from the specified journal. Do not include review-only
papers unless they provide reusable public datasets. Do not include sources
outside the specified journal scope.

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

    last_error: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SCREENING_SYSTEM_PROMPT},
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

            parsed = json.loads(extract_output_text(response))
            candidates = parsed.get("candidates", [])

            if isinstance(candidates, list):
                return candidates

            return []

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
            rows.append(row)
            continue

        if canonical_journal is None:
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
            "relevance_score": row.get("relevance_score", 0.0),
            "data_richness_score": row.get("data_richness_score", 0.0),
            "impact_score": row.get("impact_score", 0.0),
            "selection_score": row.get("selection_score", 0.0),
            "relevance_reason": row.get("relevance_reason", ""),
            "expected_extractable_data": row.get("expected_extractable_data", ""),
            "source_evidence_url": row.get("source_evidence_url"),
            "local_pdf_filename": "",
            "download_status": "not_downloaded",
            "manual_action": row.get("manual_action", ""),
            "notes": row.get("notes", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if "source_id" in df.columns:
        df = df.drop_duplicates(subset=["source_id"], keep="first")

    if "selection_score" in df.columns:
        df = df.sort_values(
            by=["selection_score", "data_richness_score", "impact_score"],
            ascending=False,
        )

    return df.reset_index(drop=True)


def save_csv_with_permission_fallback(df: pd.DataFrame, path: str) -> str:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        root, extension = os.path.splitext(path)
        fallback_path = f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{extension}"
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        return fallback_path


def run_llm_web_source_screening(
    target_count: int = 50,
    per_journal_limit: int = 8,
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

        current_df = normalise_candidates(all_candidates)

        if len(current_df) >= target_count:
            break

    df = normalise_candidates(all_candidates)

    if not df.empty:
        df = df.head(target_count)

    output_paths = {
        "interim": str(get_path("data", "interim", "candidate_sources_llm.csv")),
        "table": str(get_path("outputs", "tables", "source_screening_candidates_top50.csv")),
        "journal_scope": str(get_path("outputs", "tables", "source_screening_journal_scope.csv")),
    }

    for path in output_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    output_paths["interim"] = save_csv_with_permission_fallback(
        df=df,
        path=output_paths["interim"],
    )
    output_paths["table"] = save_csv_with_permission_fallback(
        df=df,
        path=output_paths["table"],
    )

    scope_df = pd.DataFrame([asdict(item) for item in MEETING_ONE_JOURNAL_SCOPE])
    output_paths["journal_scope"] = save_csv_with_permission_fallback(
        df=scope_df,
        path=output_paths["journal_scope"],
    )

    return df, output_paths
