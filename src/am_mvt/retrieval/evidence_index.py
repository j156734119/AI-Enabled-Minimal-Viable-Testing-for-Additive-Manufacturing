from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from am_mvt.config import get_path
from am_mvt.parsing.document_pipeline import load_active_chunk_manifest


DEFAULT_EVIDENCE_QUERY = (
    "additive manufacturing mechanical testing tensile strength yield strength "
    "elongation elastic modulus fatigue life stress amplitude R ratio frequency "
    "porosity defect surface condition heat treatment build orientation"
)


@dataclass(frozen=True)
class EvidenceSearchResult:
    chunk_id: str
    chunk_path: str
    source_file: str
    source_id: str
    chunk_sha256: str
    score: float
    evidence_snippet: str


def default_index_path() -> Path:
    return get_path("data", "interim", "evidence_index.joblib")


def _resolve_chunk_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return get_path(*path.parts)


def _normalise_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()

    required = {
        "chunk_id",
        "chunk_path",
        "chunk_sha256",
        "source_id",
        "source_file",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            "Active chunk manifest is missing required columns: "
            + ", ".join(sorted(missing))
        )

    result = manifest.copy()
    result["chunk_id"] = result["chunk_id"].astype(str)
    result["chunk_path"] = result["chunk_path"].astype(str)
    result["chunk_sha256"] = result["chunk_sha256"].astype(str)
    result["source_id"] = result["source_id"].astype(str)
    result["source_file"] = result["source_file"].astype(str)
    return result


def load_active_evidence_chunks(
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    manifest = _normalise_manifest(load_active_chunk_manifest(manifest_path))
    rows: list[dict[str, Any]] = []

    for row in manifest.itertuples(index=False):
        chunk_path = _resolve_chunk_path(row.chunk_path)
        if not chunk_path.exists() or not chunk_path.is_file():
            continue
        text = chunk_path.read_text(encoding="utf-8", errors="ignore")
        rows.append(
            {
                "chunk_id": row.chunk_id,
                "chunk_path": str(row.chunk_path),
                "chunk_sha256": row.chunk_sha256,
                "source_id": row.source_id,
                "source_file": row.source_file,
                "text": text,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "chunk_id",
            "chunk_path",
            "chunk_sha256",
            "source_id",
            "source_file",
            "text",
        ],
    )


def build_evidence_index(
    *,
    manifest_path: str | Path | None = None,
    index_path: str | Path | None = None,
) -> Path:
    chunks = load_active_evidence_chunks(manifest_path)
    if chunks.empty:
        raise ValueError("No active evidence chunks were found.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(chunks["text"].fillna(""))

    output_path = Path(index_path) if index_path is not None else default_index_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "matrix": matrix,
            "chunks": chunks.drop(columns=["text"]),
            "texts": chunks["text"].tolist(),
        },
        output_path,
    )
    return output_path


def _snippet(text: str, max_chars: int = 300) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def query_evidence_index(
    query: str,
    *,
    index_path: str | Path | None = None,
    top_k: int = 5,
) -> list[EvidenceSearchResult]:
    if top_k <= 0:
        return []

    path = Path(index_path) if index_path is not None else default_index_path()
    payload = joblib.load(path)
    vectorizer = payload["vectorizer"]
    matrix = payload["matrix"]
    chunks = payload["chunks"].reset_index(drop=True)
    texts = payload["texts"]

    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    order = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), chunks.loc[index, "chunk_id"]),
    )[:top_k]

    results = []
    for index in order:
        row = chunks.loc[index]
        results.append(
            EvidenceSearchResult(
                chunk_id=str(row["chunk_id"]),
                chunk_path=str(row["chunk_path"]),
                source_file=str(row["source_file"]),
                source_id=str(row["source_id"]),
                chunk_sha256=str(row["chunk_sha256"]),
                score=float(scores[index]),
                evidence_snippet=_snippet(str(texts[index])),
            )
        )
    return results


def rank_chunk_paths(
    chunk_paths: list[Path],
    *,
    query: str = DEFAULT_EVIDENCE_QUERY,
    top_k_per_source: int | None = None,
) -> list[Path]:
    if not chunk_paths:
        return []

    texts = [
        path.read_text(encoding="utf-8", errors="ignore")
        if path.exists()
        else ""
        for path in chunk_paths
    ]
    if not any(text.strip() for text in texts):
        return chunk_paths

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()

    rows = pd.DataFrame(
        {
            "path": chunk_paths,
            "score": scores,
            "original_order": range(len(chunk_paths)),
            "source_stem": [
                path.stem.split("_chunk_", maxsplit=1)[0]
                for path in chunk_paths
            ],
        }
    )
    rows = rows.sort_values(
        by=["score", "source_stem", "original_order"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    if top_k_per_source is not None and top_k_per_source > 0:
        rows = rows.groupby("source_stem", group_keys=False).head(top_k_per_source)
        rows = rows.sort_values(
            by=["score", "source_stem", "original_order"],
            ascending=[False, True, True],
            kind="mergesort",
        )

    return [Path(path) for path in rows["path"].tolist()]
