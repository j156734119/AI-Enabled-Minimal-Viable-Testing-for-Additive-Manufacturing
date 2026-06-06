from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


def load_source_screening_skill() -> str:
    skill_path = get_path("skills", "source_screening", "SKILL.md")

    if not skill_path.exists():
        return ""

    return skill_path.read_text(encoding="utf-8", errors="ignore")


def crossref_candidate_score(title: str) -> float:
    keywords = {
        "additive",
        "manufacturing",
        "fatigue",
        "tensile",
        "strength",
        "elongation",
        "modulus",
        "porosity",
        "defect",
        "laser",
        "powder",
        "metal",
        "alloy",
    }
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    return min(10.0, 2.0 + len(title_tokens & keywords) * 0.9)


def is_crossref_title_relevant(title: str) -> bool:
    lowered = title.lower()
    am_terms = [
        "additive manufact",
        "3d print",
        "powder bed fusion",
        "selective laser",
        "direct metal deposition",
        "directed energy deposition",
        "wire arc",
        "waam",
        "binder jet",
    ]
    metal_terms = [
        "metal",
        "alloy",
        "steel",
        "aluminium",
        "aluminum",
        "titanium",
        "inconel",
        "ti-6al-4v",
        "alsi",
        "in718",
        "316l",
        "17-4",
    ]
    data_terms = [
        "fatigue",
        "strength",
        "tensile",
        "mechanical",
        "modulus",
        "elongation",
        "porosity",
        "density",
        "hardness",
        "defect",
        "microstructure",
        "process parameter",
        "parameter optim",
    ]
    excluded_terms = ["review", "survey", "state of the art", "overview"]

    return (
        any(term in lowered for term in am_terms)
        and any(term in lowered for term in metal_terms)
        and any(term in lowered for term in data_terms)
        and not any(term in lowered for term in excluded_terms)
    )


def search_crossref_journal(
    journal_scope: JournalScope,
    per_journal_limit: int,
    year_from: int,
    year_to: int,
    timeout_seconds: int = 20,
) -> list[dict[str, Any]]:
    query = (
        "metal additive manufacturing fatigue tensile strength elongation "
        "elastic modulus porosity process parameters"
    )
    params = urlencode(
        {
            "query.container-title": journal_scope.journal,
            "query.bibliographic": query,
            "filter": (
                f"from-pub-date:{year_from}-01-01,"
                f"until-pub-date:{year_to}-12-31,type:journal-article"
            ),
            "rows": max(per_journal_limit * 3, 20),
            "select": (
                "DOI,title,container-title,published,URL,is-referenced-by-count"
            ),
        }
    )
    request = Request(
        f"https://api.crossref.org/works?{params}",
        headers={
            "User-Agent": (
                "am-mvt-source-screening/1.0 "
                "(MSc research metadata screening)"
            )
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"  Crossref lookup failed for {journal_scope.journal}: {exc}")
        return []

    items = payload.get("message", {}).get("items", [])
    candidates: list[dict[str, Any]] = []

    for item in items:
        container_titles = item.get("container-title") or []
        canonical_journal = normalise_journal_name(
            container_titles[0] if container_titles else ""
        )

        if canonical_journal != journal_scope.journal:
            continue

        titles = item.get("title") or []
        title = str(titles[0] if titles else "").strip()

        if not title or not is_crossref_title_relevant(title):
            continue

        date_parts = item.get("published", {}).get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        citations = int(item.get("is-referenced-by-count") or 0)
        relevance_score = crossref_candidate_score(title)
        impact_score = min(10.0, 2.0 + math.log10(citations + 1) * 2.5)
        selection_score = (
            relevance_score * 0.55
            + 5.0 * 0.25
            + impact_score * 0.20
        )

        candidates.append(
            {
                "title": title,
                "journal": journal_scope.journal,
                "year": year,
                "doi": item.get("DOI"),
                "url": item.get("URL"),
                "pdf_url": None,
                "access_type": "uncertain",
                "priority_tier": journal_scope.priority_tier,
                "relevance_score": relevance_score,
                "data_richness_score": 5.0,
                "impact_score": impact_score,
                "selection_score": selection_score,
                "relevance_reason": (
                    "Crossref journal-constrained metadata match; abstract/full "
                    "text relevance still requires screening."
                ),
                "expected_extractable_data": "",
                "manual_action": "Review abstract and obtain the PDF lawfully.",
                "source_evidence_url": item.get("URL"),
                "notes": "Metadata discovered and journal-checked through Crossref.",
                "_requested_journal": journal_scope.journal,
            }
        )

        if len(candidates) >= per_journal_limit:
            break

    return candidates


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
    skill_text = load_source_screening_skill()
    system_prompt = SCREENING_SYSTEM_PROMPT

    if skill_text:
        system_prompt += (
            "\n\nApply the following repository Source Screening Skill as "
            "the bounded task specification:\n" + skill_text
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

            parsed = json.loads(extract_output_text(response))
            candidates = parsed.get("candidates", [])

            if isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        candidate["_requested_journal"] = journal_scope.journal
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

    df["_normalised_doi"] = (
        df["doi"]
        .astype("string")
        .str.lower()
        .str.replace("https://doi.org/", "", regex=False)
        .str.strip()
        .replace("", pd.NA)
    )
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
    min_per_journal: int = 4,
    year_from: int = 2015,
    year_to: int = 2026,
    model: str = DEFAULT_SCREENING_MODEL,
    search_rounds: int = 3,
    use_crossref: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    client = get_client()
    all_candidates: list[dict[str, Any]] = []
    focus_areas = SEARCH_FOCUS_AREAS[: max(1, search_rounds)]

    if use_crossref:
        print("Collecting journal-constrained Crossref metadata candidates...")

        for journal_scope in MEETING_ONE_JOURNAL_SCOPE:
            all_candidates.extend(
                search_crossref_journal(
                    journal_scope=journal_scope,
                    per_journal_limit=per_journal_limit,
                    year_from=year_from,
                    year_to=year_to,
                )
            )

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
