from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from am_mvt.config import get_path
from am_mvt.ingestion.pdf_title_normaliser import (
    extract_pdf_evidence,
    infer_journal_from_filename,
    infer_journal_from_text,
    match_candidate,
    normalise_doi,
    normalise_year,
    read_candidate_sources,
)


MANIFEST_COLUMNS = [
    "article_number",
    "source_id",
    "title",
    "journal",
    "year",
    "doi",
    "source_url",
    "pdf_url",
    "access_type",
    "priority_tier",
    "local_pdf_filename",
    "file_size_bytes",
    "title_identification_method",
    "candidate_match_score",
    "title_verified",
    "needs_human_check",
    "ready_for_parsing",
    "parsed_text_present",
    "notes",
]


def article_number_from_filename(filename: str) -> str:
    match = re.match(r"^(\d{1,4})(?:__|_)", filename)
    return match.group(1).zfill(3) if match else ""


def candidate_row_for_match(
    candidates: pd.DataFrame,
    source_id: object,
    doi: object,
) -> dict[str, Any]:
    if candidates.empty:
        return {}

    working = candidates.copy()

    for col in [
        "source_id",
        "title",
        "source_title",
        "journal",
        "year",
        "source_year",
        "doi",
        "url",
        "source_url",
        "pdf_url",
        "access_type",
        "priority_tier",
    ]:
        if col not in working.columns:
            working[col] = pd.NA

    source_id_text = str(source_id or "").strip()

    if source_id_text:
        matches = working.loc[
            working["source_id"].astype("string").fillna("").eq(source_id_text)
        ]

        if not matches.empty:
            return matches.iloc[0].to_dict()

    normalised_doi = normalise_doi(doi)

    if normalised_doi:
        matches = working.loc[working["doi"].map(normalise_doi).eq(normalised_doi)]

        if not matches.empty:
            return matches.iloc[0].to_dict()

    return {}


def first_non_empty(*values: object) -> object:
    for value in values:
        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        if str(value).strip():
            return value

    return ""


def build_manifest_row(
    pdf_path: Path,
    candidates: pd.DataFrame,
    parsed_text_dir: Path,
) -> dict[str, Any]:
    try:
        evidence = extract_pdf_evidence(pdf_path)
        match = match_candidate(evidence, candidates)
        candidate = candidate_row_for_match(
            candidates=candidates,
            source_id=match.get("matched_source_id", ""),
            doi=first_non_empty(
                match.get("matched_candidate_doi", ""),
                evidence.doi,
            ),
        )

        title = first_non_empty(
            candidate.get("title"),
            candidate.get("source_title"),
            match.get("canonical_title"),
            evidence.metadata_title,
            evidence.extracted_title,
            pdf_path.stem,
        )
        journal = first_non_empty(
            candidate.get("journal"),
            match.get("matched_journal"),
            infer_journal_from_filename(pdf_path.name),
            infer_journal_from_text(evidence.first_pages_text),
        )
        year = normalise_year(
            first_non_empty(
                candidate.get("year"),
                candidate.get("source_year"),
                match.get("matched_year"),
            ),
            evidence.first_pages_text,
        )
        doi = normalise_doi(
            first_non_empty(
                candidate.get("doi"),
                match.get("matched_candidate_doi"),
                evidence.doi,
            )
        )
        source_url = first_non_empty(
            candidate.get("url"),
            candidate.get("source_url"),
            f"https://doi.org/{doi}" if doi else "",
        )
        title_verified = (
            match.get("match_method") == "doi_exact"
            or float(match.get("match_score", 0.0) or 0.0) >= 0.86
        )
        parsed_text_path = parsed_text_dir / f"{pdf_path.stem}.txt"

        return {
            "article_number": article_number_from_filename(pdf_path.name),
            "source_id": first_non_empty(
                candidate.get("source_id"),
                match.get("matched_source_id"),
            ),
            "title": title,
            "journal": journal,
            "year": year,
            "doi": doi,
            "source_url": source_url,
            "pdf_url": first_non_empty(candidate.get("pdf_url")),
            "access_type": first_non_empty(candidate.get("access_type"), "unknown"),
            "priority_tier": first_non_empty(candidate.get("priority_tier")),
            "local_pdf_filename": pdf_path.name,
            "file_size_bytes": pdf_path.stat().st_size,
            "title_identification_method": match.get("match_method", ""),
            "candidate_match_score": match.get("match_score", 0.0),
            "title_verified": title_verified,
            "needs_human_check": bool(match.get("needs_human_check", True)),
            "ready_for_parsing": True,
            "parsed_text_present": parsed_text_path.exists(),
            "notes": (
                "Manifest records metadata only; the PDF itself is not included."
            ),
        }

    except Exception as exc:
        return {
            "article_number": article_number_from_filename(pdf_path.name),
            "source_id": "",
            "title": pdf_path.stem,
            "journal": "",
            "year": "",
            "doi": "",
            "source_url": "",
            "pdf_url": "",
            "access_type": "unknown",
            "priority_tier": "",
            "local_pdf_filename": pdf_path.name,
            "file_size_bytes": pdf_path.stat().st_size,
            "title_identification_method": "error",
            "candidate_match_score": 0.0,
            "title_verified": False,
            "needs_human_check": True,
            "ready_for_parsing": False,
            "parsed_text_present": False,
            "notes": f"Metadata extraction failed: {exc}",
        }


def build_literature_manifest(
    pdf_dir: str | Path | None = None,
    parsed_text_dir: str | Path | None = None,
) -> pd.DataFrame:
    if pdf_dir is None:
        pdf_dir = get_path("data", "raw", "pdfs")
    else:
        pdf_dir = Path(pdf_dir)

    if parsed_text_dir is None:
        parsed_text_dir = get_path("data", "interim", "parsed_text")
    else:
        parsed_text_dir = Path(parsed_text_dir)

    pdf_dir.mkdir(parents=True, exist_ok=True)
    parsed_text_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_candidate_sources()
    rows = [
        build_manifest_row(
            pdf_path=pdf_path,
            candidates=candidates,
            parsed_text_dir=parsed_text_dir,
        )
        for pdf_path in sorted(pdf_dir.glob("*.pdf"))
    ]

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    if not manifest.empty:
        manifest = manifest.sort_values(
            by=["article_number", "local_pdf_filename"],
            na_position="last",
        ).reset_index(drop=True)

    return manifest


def save_literature_manifest(
    output_path: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    if output_path is None:
        output_path = get_path("docs", "literature_manifest.csv")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_literature_manifest()
    manifest.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path, manifest
