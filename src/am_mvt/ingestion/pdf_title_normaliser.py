from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from am_mvt.config import get_path
from am_mvt.utils.text import normalise_doi


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
NUMBERED_PDF_PATTERN = re.compile(r"^(\d{1,4})(?:__|_)")

JOURNAL_CODES = {
    "additive manufacturing": "addma",
    "journal of materials processing technology": "jmpt",
    "journal of manufacturing processes": "jmapro",
    "rapid prototyping journal": "rpj",
    "metals": "metals",
    "virtual and physical prototyping": "vpp",
    "progress in additive manufacturing": "progaddmanuf",
    "advanced engineering materials": "adem",
}
JOURNAL_NAMES_BY_CODE = {
    "addma": "Additive Manufacturing",
    "jmpt": "Journal of Materials Processing Technology",
    "jmapro": "Journal of Manufacturing Processes",
    "rpj": "Rapid Prototyping Journal",
    "metals": "Metals",
    "vpp": "Virtual and Physical Prototyping",
    "progaddmanuf": "Progress in Additive Manufacturing",
    "adem": "Advanced Engineering Materials",
}

TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "the",
    "to",
    "under",
    "using",
    "via",
    "with",
    "effect",
    "effects",
    "study",
    "investigation",
    "analysis",
    "characterization",
    "characterisation",
    "behavior",
    "behaviour",
    "properties",
    "property",
    "parts",
    "part",
    "layer",
    "steel",
    "stainless",
    "manufactured",
    "manufacturing",
    "additive",
    "additively",
}


@dataclass(frozen=True)
class PdfEvidence:
    metadata_title: str
    extracted_title: str
    doi: str
    first_pages_text: str


def normalise_title(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"(?:\.pdf)+$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*(?:[-_—–－]+\s*)+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(left: object, right: object) -> float:
    left_text = normalise_title(left)
    right_text = normalise_title(right)

    if not left_text or not right_text:
        return 0.0

    if left_text == right_text:
        return 1.0

    sequence = fuzz.ratio(left_text, right_text) / 100.0
    token_order = fuzz.token_sort_ratio(left_text, right_text) / 100.0
    return max(sequence, token_order)


def is_plausible_title(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    lowered = cleaned.lower()

    if len(cleaned) < 15 or len(cleaned) > 350:
        return False

    if DOI_PATTERN.search(cleaned):
        return False

    rejected_fragments = [
        "www.",
        "http",
        "received ",
        "accepted ",
        "available online",
        "copyright",
        "contents lists available",
        "journal homepage",
    ]

    return not any(fragment in lowered for fragment in rejected_fragments)


def extract_title_from_first_page(page) -> str:
    page_dict = page.get_text("dict")
    lines: list[tuple[float, float, str]] = []

    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(
                str(span.get("text", "")).strip()
                for span in spans
                if str(span.get("text", "")).strip()
            )

            if not text:
                continue

            max_size = max(float(span.get("size", 0.0)) for span in spans)
            y_position = float(line.get("bbox", [0, 0, 0, 0])[1])
            lines.append((y_position, max_size, text))

    plausible = [line for line in lines if is_plausible_title(line[2])]

    if not plausible:
        return ""

    plausible.sort(key=lambda item: item[0])
    largest_size = max(item[1] for item in plausible)
    title_lines = [
        item for item in plausible
        if item[1] >= largest_size * 0.82 and item[0] <= page.rect.height * 0.55
    ]

    if not title_lines:
        return max(plausible, key=lambda item: item[1])[2]

    title_lines.sort(key=lambda item: item[0])
    combined = " ".join(item[2] for item in title_lines[:4])

    return re.sub(r"\s+", " ", combined).strip()


def extract_pdf_evidence(pdf_path: Path, max_text_pages: int = 2) -> PdfEvidence:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required. Install it with: python -m pip install pymupdf"
        ) from exc

    with fitz.open(pdf_path) as document:
        metadata_title = str((document.metadata or {}).get("title") or "").strip()
        extracted_title = ""
        page_texts: list[str] = []

        for page_index in range(min(len(document), max_text_pages)):
            page = document.load_page(page_index)
            page_texts.append(page.get_text("text"))

            if page_index == 0:
                extracted_title = extract_title_from_first_page(page)

    first_pages_text = "\n".join(page_texts)
    doi_match = DOI_PATTERN.search(first_pages_text)
    doi = normalise_doi(doi_match.group(0)) if doi_match else ""

    if not is_plausible_title(metadata_title):
        metadata_title = ""

    return PdfEvidence(
        metadata_title=metadata_title,
        extracted_title=extracted_title,
        doi=doi,
        first_pages_text=first_pages_text,
    )


def read_candidate_sources() -> pd.DataFrame:
    paths = [
        get_path("data", "interim", "candidate_sources_llm.csv"),
        get_path("outputs", "tables", "source_screening_candidates_top50.csv"),
        get_path("data", "raw", "metadata", "candidate_sources.csv"),
    ]

    for path in paths:
        if path.exists():
            return pd.read_csv(path, low_memory=False)

    return pd.DataFrame()


def match_candidate(
    evidence: PdfEvidence,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    if candidates.empty:
        return {
            "canonical_title": evidence.metadata_title or evidence.extracted_title,
            "matched_source_id": "",
            "matched_candidate_doi": "",
            "matched_journal": "",
            "matched_year": "",
            "match_method": "pdf_title_only",
            "match_score": 0.0,
            "needs_human_check": True,
        }

    working = candidates.copy()

    for col in ["source_id", "title", "source_title", "doi"]:
        if col not in working.columns:
            working[col] = pd.NA

    working["_candidate_title"] = working["title"].combine_first(
        working["source_title"]
    )
    working["_normalised_doi"] = working["doi"].map(normalise_doi)

    if evidence.doi:
        doi_matches = working.loc[working["_normalised_doi"].eq(evidence.doi)]

        if not doi_matches.empty:
            row = doi_matches.iloc[0]
            return {
                "canonical_title": str(row["_candidate_title"]),
                "matched_source_id": row.get("source_id", ""),
                "matched_candidate_doi": row.get("doi", ""),
                "matched_journal": row.get("journal", ""),
                "matched_year": row.get("year", row.get("source_year", "")),
                "match_method": "doi_exact",
                "match_score": 1.0,
                "needs_human_check": False,
            }

    observed_titles = [
        title
        for title in [evidence.metadata_title, evidence.extracted_title]
        if title
    ]
    best_score = 0.0
    best_row: pd.Series | None = None

    for _, row in working.iterrows():
        candidate_title = row["_candidate_title"]
        score = max(
            (title_similarity(observed, candidate_title) for observed in observed_titles),
            default=0.0,
        )

        normalised_candidate = normalise_title(candidate_title)
        normalised_text = normalise_title(evidence.first_pages_text[:12000])

        if normalised_candidate and normalised_candidate in normalised_text:
            score = max(score, 0.99)

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is not None and best_score >= 0.72:
        return {
            "canonical_title": str(best_row["_candidate_title"]),
            "matched_source_id": best_row.get("source_id", ""),
            "matched_candidate_doi": best_row.get("doi", ""),
            "matched_journal": best_row.get("journal", ""),
            "matched_year": best_row.get("year", best_row.get("source_year", "")),
            "match_method": "candidate_title_similarity",
            "match_score": round(best_score, 4),
            "needs_human_check": best_score < 0.86,
        }

    fallback_title = evidence.metadata_title or evidence.extracted_title

    return {
        "canonical_title": fallback_title,
        "matched_source_id": "",
        "matched_candidate_doi": "",
        "matched_journal": "",
        "matched_year": "",
        "match_method": "pdf_title_only",
        "match_score": round(best_score, 4),
        "needs_human_check": True,
    }


def sanitise_windows_filename(value: object, max_length: int = 170) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")

    if not text:
        text = "untitled_paper"

    return text[:max_length].rstrip(" .")


def journal_code(value: object) -> str:
    normalised = normalise_title(value)

    if normalised in JOURNAL_CODES:
        return JOURNAL_CODES[normalised]

    compact = "".join(word[0] for word in normalised.split() if word)
    return compact[:12] or "unknownjournal"


def infer_journal_from_text(text: str) -> str:
    normalised_text = normalise_title(text[:20000])

    for journal_name in JOURNAL_CODES:
        if journal_name == "additive manufacturing":
            continue

        if normalise_title(journal_name) in normalised_text:
            return journal_name

    lowered_text = text[:20000].lower()

    if (
        "j.addma" in lowered_text
        or "elsevier.com/locate/addma" in lowered_text
        or "journal of additive manufacturing" in lowered_text
        or re.search(r"\badditive manufacturing\s+\d+\s*\(", lowered_text)
    ):
        return "additive manufacturing"

    return ""


def infer_journal_from_filename(filename: str) -> str:
    match = re.match(r"^\d{1,4}_([a-z0-9]+)_", Path(filename).name.lower())

    if not match:
        return ""

    return JOURNAL_NAMES_BY_CODE.get(match.group(1), "")


def normalise_year(value: object, text: str = "") -> str:
    if value is not None and not pd.isna(value):
        match = re.search(r"\b(19|20)\d{2}\b", str(value))

        if match:
            return match.group(0)

    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    return years[0] if years else "unknownyear"


def compact_title_slug(title: object, max_tokens: int = 10) -> str:
    tokens = normalise_title(title).split()
    selected: list[str] = []

    for token in tokens:
        if token in TITLE_STOPWORDS or len(token) < 2:
            continue

        if token not in selected:
            selected.append(token)

        if len(selected) >= max_tokens:
            break

    material_pattern = re.compile(
        r"^(?:\d{2,4}l?|alsi\d+mg|inconel|ti\d*|6al|4v|ta\d+|h\d+|"
        r"maraging|nickel|aluminium|aluminum|titanium)$"
    )
    material_tokens = [token for token in selected if material_pattern.match(token)]
    topic_tokens = [token for token in selected if token not in material_tokens]

    return "_".join(material_tokens + topic_tokens) or "untitled_paper"


def next_pdf_number(output_dir: Path) -> int:
    numbers = []

    for path in output_dir.glob("*.pdf"):
        match = NUMBERED_PDF_PATTERN.match(path.name)

        if match:
            numbers.append(int(match.group(1)))

    return max(numbers, default=0) + 1


def unique_target_path(
    output_dir: Path,
    number: int,
    title: str,
    journal: object,
    year: object,
    evidence_text: str,
) -> Path:
    code = journal_code(journal)
    year_text = normalise_year(year, evidence_text)
    slug = compact_title_slug(title)
    target = output_dir / f"{number:03d}_{code}_{year_text}_{slug}.pdf"
    suffix = 2

    while target.exists():
        target = output_dir / (
            f"{number:03d}_{code}_{year_text}_{slug}_{suffix}.pdf"
        )
        suffix += 1

    return target


def prepare_pdf_normalisation(
    inbox_dir: Path,
    output_dir: Path,
    apply_changes: bool = False,
) -> pd.DataFrame:
    inbox_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_candidate_sources()
    rows: list[dict[str, object]] = []
    number = next_pdf_number(output_dir)

    for source_path in sorted(inbox_dir.glob("*.pdf")):
        try:
            evidence = extract_pdf_evidence(source_path)
            match = match_candidate(evidence, candidates)
            canonical_title = str(match["canonical_title"] or "").strip()

            if not canonical_title:
                canonical_title = source_path.stem
                match["needs_human_check"] = True
                match["match_method"] = "original_filename_fallback"

            target_path = unique_target_path(
                output_dir=output_dir,
                number=number,
                title=canonical_title,
                journal=(
                    match.get("matched_journal", "")
                    or infer_journal_from_text(evidence.first_pages_text)
                ),
                year=match.get("matched_year", ""),
                evidence_text=evidence.first_pages_text,
            )
            action = "planned_move"

            if apply_changes:
                shutil.move(str(source_path), str(target_path))
                action = "moved"

            rows.append(
                {
                    "original_filename": source_path.name,
                    "metadata_title": evidence.metadata_title,
                    "extracted_first_page_title": evidence.extracted_title,
                    "doi_found": evidence.doi,
                    **match,
                    "normalised_filename": target_path.name,
                    "normalised_path": target_path.relative_to(get_path()).as_posix(),
                    "action": action,
                    "error": "",
                }
            )
            number += 1

        except Exception as exc:
            rows.append(
                {
                    "original_filename": source_path.name,
                    "metadata_title": "",
                    "extracted_first_page_title": "",
                    "doi_found": "",
                    "canonical_title": "",
                    "matched_source_id": "",
                    "matched_candidate_doi": "",
                    "matched_journal": "",
                    "matched_year": "",
                    "match_method": "error",
                    "match_score": 0.0,
                    "needs_human_check": True,
                    "normalised_filename": "",
                    "normalised_path": "",
                    "action": "not_moved",
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)
