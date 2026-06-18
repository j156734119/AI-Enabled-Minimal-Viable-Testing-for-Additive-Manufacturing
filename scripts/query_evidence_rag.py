from __future__ import annotations

import argparse
import csv
import sys

from am_mvt.retrieval.evidence_index import query_evidence_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local-only PDF chunk evidence index."
    )
    parser.add_argument("query", type=str, help="Natural-language evidence query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Optional path to the evidence index joblib file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = query_evidence_index(
        args.query,
        index_path=args.index_path,
        top_k=args.top_k,
    )
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "rank",
            "score",
            "source_file",
            "source_id",
            "chunk_id",
            "chunk_sha256",
            "evidence_snippet",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for rank, result in enumerate(results, start=1):
        writer.writerow(
            {
                "rank": rank,
                "score": f"{result.score:.6f}",
                "source_file": result.source_file,
                "source_id": result.source_id,
                "chunk_id": result.chunk_id,
                "chunk_sha256": result.chunk_sha256,
                "evidence_snippet": result.evidence_snippet,
            }
        )


if __name__ == "__main__":
    main()
