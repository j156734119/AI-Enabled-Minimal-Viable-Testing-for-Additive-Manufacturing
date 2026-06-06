"""
Step 02: Prepare lawful file folders and provenance audit tables.

This script intentionally does not automate publisher downloads. The user should
manually obtain PDFs or supplementary files through lawful routes, then place
them in data/raw/pdfs or data/raw/supplementary.
"""

from pathlib import Path
from difflib import SequenceMatcher
import re
import unicodedata

import pandas as pd

from am_mvt.config import get_path


def read_candidate_sources() -> pd.DataFrame:
    candidates = [
        get_path("data", "interim", "candidate_sources_llm.csv"),
        get_path("outputs", "tables", "source_screening_candidates_top50.csv"),
        get_path("data", "raw", "metadata", "candidate_sources.csv"),
    ]

    for path in candidates:
        if path.exists():
            return pd.read_csv(path, low_memory=False)

    return pd.DataFrame()


def make_safe_filename_stem(value: object) -> str:
    text = str(value or "").strip().lower()
    keep = []

    for char in text:
        if char.isalnum():
            keep.append(char)
        elif char in {" ", "-", "_"}:
            keep.append("_")

    stem = "".join(keep)

    while "__" in stem:
        stem = stem.replace("__", "_")

    return stem.strip("_")[:120] or "manual_pdf"


def normalise_title_for_matching(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"(?:\.pdf)+$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*(?:[-_—–－]+\s*)+", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_match_score(candidate_title: object, pdf_filename: str) -> float:
    title = normalise_title_for_matching(candidate_title)
    filename = normalise_title_for_matching(pdf_filename)

    if not title or not filename:
        return 0.0

    if title == filename:
        return 1.0

    if title in filename or filename in title:
        shorter = min(len(title), len(filename))
        longer = max(len(title), len(filename))
        return 0.95 * (shorter / longer) + 0.05

    title_tokens = set(title.split())
    filename_tokens = set(filename.split())
    token_overlap = len(title_tokens & filename_tokens) / max(
        1,
        min(len(title_tokens), len(filename_tokens)),
    )
    sequence_score = SequenceMatcher(None, title, filename).ratio()

    return max(sequence_score, token_overlap * 0.95)


def match_candidate_to_pdf(
    row: pd.Series,
    pdf_paths: list[Path],
    minimum_score: float = 0.78,
) -> dict[str, object]:
    explicit_filename = row.get("local_pdf_filename")

    if pd.notna(explicit_filename) and str(explicit_filename).strip():
        explicit_name = str(explicit_filename).strip()

        for pdf_path in pdf_paths:
            if pdf_path.name.casefold() == explicit_name.casefold():
                return {
                    "matched_local_pdf_filename": pdf_path.name,
                    "pdf_match_method": "explicit_filename",
                    "pdf_match_score": 1.0,
                    "download_status": "manual_file_present",
                }

    expected_filename = infer_expected_pdf_filename(row)

    for pdf_path in pdf_paths:
        if pdf_path.name.casefold() == expected_filename.casefold():
            return {
                "matched_local_pdf_filename": pdf_path.name,
                "pdf_match_method": "expected_filename",
                "pdf_match_score": 1.0,
                "download_status": "manual_file_present",
            }

    title = row.get("title")

    if pd.isna(title) or not str(title).strip():
        title = row.get("source_title")

    scored_paths = [
        (title_match_score(title, pdf_path.name), pdf_path)
        for pdf_path in pdf_paths
    ]

    if scored_paths:
        best_score, best_path = max(scored_paths, key=lambda item: item[0])

        if best_score >= minimum_score:
            method = "normalised_title_exact" if best_score == 1.0 else "title_similarity"
            return {
                "matched_local_pdf_filename": best_path.name,
                "pdf_match_method": method,
                "pdf_match_score": round(best_score, 4),
                "download_status": "manual_file_present",
            }

    return {
        "matched_local_pdf_filename": "",
        "pdf_match_method": "no_match",
        "pdf_match_score": 0.0,
        "download_status": "not_downloaded",
    }


def infer_expected_pdf_filename(row: pd.Series) -> str:
    source_id = row.get("source_id")

    if pd.notna(source_id) and str(source_id).strip():
        return f"{make_safe_filename_stem(source_id)}.pdf"

    title = row.get("title") or row.get("source_title")
    year = row.get("year") or row.get("source_year")
    year_text = str(int(year)) if pd.notna(year) else ""
    stem = make_safe_filename_stem(f"{year_text}_{title}")

    return f"{stem}.pdf"


def build_pdf_inventory(pdf_dir: Path) -> pd.DataFrame:
    rows = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        rows.append(
            {
                "local_pdf_filename": pdf_path.name,
                "local_pdf_path": pdf_path.relative_to(get_path()).as_posix(),
                "file_size_bytes": pdf_path.stat().st_size,
                "download_status": "manual_file_present",
                "notes": "User-provided local PDF; verify lawful access before extraction.",
            }
        )

    return pd.DataFrame(rows)


def build_source_provenance_table(candidate_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if candidate_df.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                "title",
                "journal",
                "year",
                "doi",
                "url",
                "pdf_url",
                "access_type",
                "priority_tier",
                "expected_local_pdf_filename",
                "matched_local_pdf_filename",
                "pdf_match_method",
                "pdf_match_score",
                "download_status",
                "manual_action",
                "needs_human_check",
                "notes",
            ]
        )

    result = candidate_df.copy()

    for col in [
        "source_id",
        "title",
        "journal",
        "year",
        "doi",
        "url",
        "pdf_url",
        "access_type",
        "priority_tier",
        "manual_action",
        "notes",
    ]:
        if col not in result.columns:
            result[col] = pd.NA

    result["expected_local_pdf_filename"] = result.apply(
        infer_expected_pdf_filename,
        axis=1,
    )

    match_df = result.apply(
        lambda row: pd.Series(match_candidate_to_pdf(row, pdf_paths)),
        axis=1,
    )

    for col in match_df.columns:
        result[col] = match_df[col]

    result["needs_human_check"] = True

    default_action = (
        "Manually obtain through open-access, publisher public page, "
        "author manuscript, or lawful university route. Do not automate login."
    )
    result["manual_action"] = result["manual_action"].fillna(default_action)
    result.loc[result["manual_action"].astype("string").str.strip().eq(""), "manual_action"] = (
        default_action
    )

    columns = [
        "source_id",
        "title",
        "journal",
        "year",
        "doi",
        "url",
        "pdf_url",
        "access_type",
        "priority_tier",
        "expected_local_pdf_filename",
        "matched_local_pdf_filename",
        "pdf_match_method",
        "pdf_match_score",
        "download_status",
        "manual_action",
        "needs_human_check",
        "notes",
    ]

    return result[columns]


def main() -> None:
    pdf_dir = get_path("data", "raw", "pdfs")
    pdf_inbox_dir = pdf_dir / "inbox"
    supplementary_dir = get_path("data", "raw", "supplementary")
    dataset_dir = get_path("data", "raw", "open_datasets")
    table_dir = get_path("outputs", "tables")

    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_inbox_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = read_candidate_sources()
    provenance_df = build_source_provenance_table(candidate_df, pdf_dir)
    inventory_df = build_pdf_inventory(pdf_dir)

    provenance_path = table_dir / "source_provenance_audit.csv"
    inventory_path = table_dir / "local_pdf_inventory.csv"

    provenance_df.to_csv(provenance_path, index=False, encoding="utf-8-sig")
    inventory_df.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    print("Step 02 complete: folders and provenance audit tables prepared.")
    print(f"PDF folder: {pdf_dir}")
    print(f"New PDF inbox: {pdf_inbox_dir}")
    print(f"Supplementary folder: {supplementary_dir}")
    print(f"Open dataset folder: {dataset_dir}")
    print(f"Source provenance audit: {provenance_path}")
    print(f"Local PDF inventory: {inventory_path}")
    print("No publisher login, cookie, VPN, or subscription download was automated.")


if __name__ == "__main__":
    main()
