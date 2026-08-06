from __future__ import annotations

import argparse
import csv
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from am_mvt.config import get_path
from am_mvt.utils.artifacts import ensure_within_project, sha256_file


LEGACY_DATA_FILES = [
    "data/processed/sources.csv",
    "data/processed/build_conditions.csv",
    "data/processed/mechanical_tests.csv",
    "data/processed/extraction_audit.csv",
    "data/processed/modelling_dataset_validation_report.csv",
]
DUPLICATE_SOURCE_PREFIXES = [
    "039_addma_2022_316l_energy_density_porosity_structure_tensile_produced_laser_powder_bed",
    "042_addma_2025_alsi10mg_fatigue_response_laser_powder_bed_fusion_influence_build_orientation",
    "045_addma_2025_alsi10mg_impact_different_pore_types_tensile_fatigue_produced_laser_powder",
]
CANONICAL_INTERIM_FILES = [
    "data/interim/llm_extracted_records.csv",
    "data/interim/llm_extraction_audit.csv",
    "data/interim/llm_extraction_audit_review.csv",
]
LEGACY_DOCUMENT_CODE_FILES = [
    "scripts/build_meeting_two_docs.py",
    "scripts/build_meeting_two_simple.py",
    "scripts/build_project_progress_update.py",
    "scripts/build_three_line_progress_update.py",
]
LOCAL_WORKING_ARTIFACTS = [
    ".codex-doc-review-meeting-three",
    ".codex_doc_work",
    ".codex_tmp",
    "tmp",
]
DOCUMENT_EXTENSIONS = {".doc", ".docx", ".pdf"}


@dataclass(frozen=True)
class MoveItem:
    source: Path
    destination: Path


@dataclass(frozen=True)
class ManifestRow:
    original_path: str
    archived_path: str
    size_bytes: int
    modified_at_utc: str
    sha256: str


def default_archive_name() -> str:
    return "cleanup_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or archive explicitly approved legacy artifacts."
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--archive-name", default=default_archive_name())
    parser.add_argument(
        "--keep-experiment",
        action="append",
        dest="keep_experiments",
        help=(
            "Experiment run to retain in place. Repeat the option to keep "
            "multiple runs. Defaults to balanced_v2 when omitted."
        ),
    )
    parser.add_argument("--include-current-processed", action="store_true")
    parser.add_argument("--include-duplicate-derivatives", action="store_true")
    parser.add_argument("--include-documents", action="store_true")
    parser.add_argument("--include-document-code", action="store_true")
    parser.add_argument("--include-local-working-artifacts", action="store_true")
    return parser.parse_args(argv)


def _destination(archive_root: Path, source: Path) -> Path:
    return archive_root / source.relative_to(get_path())


def build_move_plan(
    *,
    archive_name: str = "cleanup_test",
    keep_experiment: str | None = "balanced_v2",
    keep_experiments: Iterable[str] | None = None,
    include_current_processed: bool = False,
    include_duplicate_derivatives: bool = False,
    include_documents: bool = False,
    include_document_code: bool = False,
    include_local_working_artifacts: bool = False,
) -> list[MoveItem]:
    archive_root = get_path("archive", archive_name)
    sources: list[Path] = []
    processed_dir = get_path("data", "processed")
    retained_experiments = set(keep_experiments or [])
    if keep_experiment:
        retained_experiments.add(keep_experiment)

    sources.extend(
        path
        for path in (
            get_path(*relative.split("/")) for relative in LEGACY_DATA_FILES
        )
        if path.is_file()
    )
    sources.extend(
        sorted(processed_dir.glob("master_modelling_dataset_backup_*.csv"))
        if processed_dir.exists()
        else []
    )
    if include_current_processed and processed_dir.exists():
        sources.extend(sorted(processed_dir.glob("*.csv")))
        sources.extend(
            path
            for path in (
                get_path(*relative.split("/"))
                for relative in CANONICAL_INTERIM_FILES
            )
            if path.is_file()
        )

    models_dir = get_path("outputs", "models")
    if models_dir.exists():
        sources.extend(
            path
            for path in models_dir.iterdir()
            if path.is_file() and path.name != ".gitkeep"
        )

    tables_dir = get_path("outputs", "tables")
    if tables_dir.exists():
        sources.extend(sorted(tables_dir.glob("project_*.csv")))

    experiments_dir = get_path("outputs", "experiments")
    if experiments_dir.exists():
        sources.extend(
            path
            for path in experiments_dir.iterdir()
            if path.is_dir() and path.name not in retained_experiments
        )

    if include_duplicate_derivatives:
        for directory in [
            get_path("data", "interim", "parsed_text"),
            get_path("data", "interim", "text_chunks"),
            get_path("data", "interim", "llm_outputs"),
        ]:
            if not directory.exists():
                continue
            for prefix in DUPLICATE_SOURCE_PREFIXES:
                sources.extend(sorted(directory.glob(f"{prefix}*")))

    if include_documents:
        outputs_dir = get_path("outputs")
        if outputs_dir.exists():
            sources.extend(
                path
                for path in outputs_dir.iterdir()
                if path.is_file() and path.suffix.lower() in DOCUMENT_EXTENSIONS
            )
            sources.extend(sorted(outputs_dir.glob("qa_meeting_*")))
            sources.extend(sorted(outputs_dir.glob("qa_progress_update_*")))
            meeting_style = outputs_dir / "meeting_one_style.json"
            if meeting_style.exists():
                sources.append(meeting_style)

        dissertation_documents = get_path(
            "data",
            "interim",
            "dissertation_outline_docx",
        )
        if dissertation_documents.exists():
            sources.append(dissertation_documents)

    if include_document_code:
        sources.extend(
            path
            for path in (
                get_path(*relative.split("/"))
                for relative in LEGACY_DOCUMENT_CODE_FILES
            )
            if path.exists()
        )

    if include_local_working_artifacts:
        sources.extend(
            path
            for path in (
                get_path(*relative.split("/"))
                for relative in LOCAL_WORKING_ARTIFACTS
            )
            if path.exists()
        )

    unique_sources = sorted({path.resolve() for path in sources if path.exists()})
    return [
        MoveItem(source=source, destination=_destination(archive_root, source))
        for source in unique_sources
    ]


def build_manifest_rows(moves: list[MoveItem]) -> list[ManifestRow]:
    project_root = get_path().resolve()
    rows: list[ManifestRow] = []
    for move in moves:
        files = sorted(move.source.rglob("*")) if move.source.is_dir() else [move.source]
        for source_file in files:
            if not source_file.is_file():
                continue
            suffix = (
                source_file.relative_to(move.source)
                if move.source.is_dir()
                else Path()
            )
            destination_file = move.destination / suffix
            stat = source_file.stat()
            rows.append(
                ManifestRow(
                    original_path=source_file.relative_to(project_root).as_posix(),
                    archived_path=destination_file.relative_to(project_root).as_posix(),
                    size_bytes=stat.st_size,
                    modified_at_utc=datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                    sha256=sha256_file(source_file),
                )
            )
    return rows


def validate_move_plan(moves: list[MoveItem]) -> None:
    destinations: set[Path] = set()
    for move in moves:
        source = ensure_within_project(move.source, project_root=get_path())
        destination = ensure_within_project(
            move.destination,
            project_root=get_path(),
        )
        if not source.exists():
            raise FileNotFoundError(f"Artifact disappeared: {source}")
        if destination.exists():
            raise FileExistsError(f"Archive destination exists: {destination}")
        if destination in destinations:
            raise ValueError(f"Duplicate archive destination: {destination}")
        destinations.add(destination)


def write_manifest(rows: list[ManifestRow], archive_name: str) -> Path:
    manifest_path = get_path("archive", archive_name, "archive_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return manifest_path


def apply_move_plan(
    moves: list[MoveItem],
    rows: list[ManifestRow],
    archive_name: str = "cleanup_test",
) -> Path:
    if not rows:
        raise RuntimeError("No files matched the approved archive whitelist.")
    for move in moves:
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.source), str(move.destination))
    return write_manifest(rows, archive_name)


def main() -> None:
    args = parse_args()
    moves = build_move_plan(
        archive_name=args.archive_name,
        keep_experiment=None,
        keep_experiments=args.keep_experiments or ["balanced_v2"],
        include_current_processed=args.include_current_processed,
        include_duplicate_derivatives=args.include_duplicate_derivatives,
        include_documents=args.include_documents,
        include_document_code=args.include_document_code,
        include_local_working_artifacts=args.include_local_working_artifacts,
    )
    validate_move_plan(moves)
    rows = build_manifest_rows(moves)
    print(f"{'APPLY' if args.apply else 'DRY RUN'}: {len(moves)} items, {len(rows)} files")
    for move in moves:
        print(
            f"{move.source.relative_to(get_path()).as_posix()} -> "
            f"{move.destination.relative_to(get_path()).as_posix()}"
        )
    if not args.apply:
        print("No files were moved.")
        return
    manifest = apply_move_plan(moves, rows, args.archive_name)
    print(f"Archive complete: {manifest}")


if __name__ == "__main__":
    main()
