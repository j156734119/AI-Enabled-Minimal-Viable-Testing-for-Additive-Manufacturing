from __future__ import annotations

import argparse

from am_mvt.agent.react_ledger import ReactLedger, record_human_download_boundary
from am_mvt.retrieval.evidence_index import build_evidence_index, load_active_evidence_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local-only TF-IDF evidence index from active PDF chunks."
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Optional output path for the joblib evidence index.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional ReAct-style ledger run id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_active_evidence_chunks()
    if chunks.empty:
        raise SystemExit(
            "No active chunks found. Run python scripts/03_parse_documents.py first."
        )

    ledger = ReactLedger(run_id=args.run_id)
    record_human_download_boundary(
        ledger,
        candidate_refs=[],
        observed_pdf_refs=chunks["chunk_id"].astype(str).tolist(),
    )
    ledger.record(
        plan_summary=(
            "Build local retrieval evidence over user-supplied PDF chunks for "
            "retrieval-augmented extraction."
        ),
        action_type="build_local_evidence_index",
        input_refs=chunks["chunk_id"].astype(str).tolist(),
        observation_summary=f"Indexed {len(chunks)} active chunks.",
        decision="index_ready_for_local_rag_queries",
        evidence_refs=chunks["chunk_sha256"].astype(str).tolist(),
    )

    index_path = build_evidence_index(index_path=args.index_path)
    print(f"Evidence index written: {index_path}")
    print(f"ReAct-style ledger: {ledger.csv_path}")


if __name__ == "__main__":
    main()

