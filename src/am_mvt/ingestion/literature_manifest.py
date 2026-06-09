from __future__ import annotations

import re
from difflib import SequenceMatcher
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
from am_mvt.utils.artifacts import sha256_file


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
    "content_sha256",
    "canonical_source_id",
    "duplicate_of",
    "processing_status",
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


def source_id_from_filename(filename: str) -> str:
    stem = Path(filename).name
    while stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    return re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()


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
        source_id = first_non_empty(
            candidate.get("source_id"),
            match.get("matched_source_id"),
            source_id_from_filename(pdf_path.name),
        )

        return {
            "article_number": article_number_from_filename(pdf_path.name),
            "source_id": source_id,
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
            "content_sha256": sha256_file(pdf_path),
            "canonical_source_id": source_id,
            "duplicate_of": "",
            "processing_status": "canonical",
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
            "content_sha256": sha256_file(pdf_path),
            "canonical_source_id": source_id_from_filename(pdf_path.name),
            "duplicate_of": "",
            "processing_status": "error",
            "title_identification_method": "error",
            "candidate_match_score": 0.0,
            "title_verified": False,
            "needs_human_check": True,
            "ready_for_parsing": False,
            "parsed_text_present": False,
            "notes": f"Metadata extraction failed: {exc}",
        }


def mark_duplicate_sources(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest

    result = manifest.copy()
    result["_normalised_doi"] = result["doi"].map(normalise_doi)
    result["_duplicate_key"] = result.apply(
        lambda row: (
            f"doi:{row['_normalised_doi']}"
            if row["_normalised_doi"]
            else f"sha256:{row['content_sha256']}"
        ),
        axis=1,
    )

    for _, group in result.groupby("_duplicate_key", sort=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values(
            by=["article_number", "local_pdf_filename"],
            na_position="last",
        )
        canonical_index = ordered.index[0]
        canonical = result.loc[canonical_index]
        canonical_id = str(canonical["source_id"])
        metadata_columns = ["title", "journal", "year", "doi"]

        for duplicate_index in ordered.index[1:]:
            duplicate = result.loc[duplicate_index]
            same_content = (
                str(duplicate["content_sha256"])
                == str(canonical["content_sha256"])
            )
            title_similarity = SequenceMatcher(
                None,
                re.sub(r"\W+", " ", str(canonical["title"]).casefold()).strip(),
                re.sub(r"\W+", " ", str(duplicate["title"]).casefold()).strip(),
            ).ratio()
            conflicts = [
                column
                for column in metadata_columns
                if str(result.at[duplicate_index, column]).strip()
                and str(canonical[column]).strip()
                and str(result.at[duplicate_index, column]).strip().casefold()
                != str(canonical[column]).strip().casefold()
            ]
            if not same_content and title_similarity < 0.80:
                result.at[duplicate_index, "processing_status"] = (
                    "metadata_conflict"
                )
                result.at[duplicate_index, "ready_for_parsing"] = False
                existing = str(result.at[duplicate_index, "notes"]).strip()
                result.at[duplicate_index, "notes"] = (
                    f"{existing} DOI matched another record but title/content "
                    "did not; excluded pending human review."
                ).strip()
                continue
            result.at[duplicate_index, "canonical_source_id"] = canonical_id
            result.at[duplicate_index, "duplicate_of"] = canonical_id
            result.at[duplicate_index, "processing_status"] = "duplicate"
            result.at[duplicate_index, "ready_for_parsing"] = False
            if conflicts:
                existing = str(result.at[duplicate_index, "notes"]).strip()
                result.at[duplicate_index, "notes"] = (
                    f"{existing} Metadata conflict with canonical source: "
                    + ", ".join(conflicts)
                ).strip()

        result.at[canonical_index, "canonical_source_id"] = canonical_id

    return result.drop(columns=["_normalised_doi", "_duplicate_key"])


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

    manifest = mark_duplicate_sources(
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    )

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
