from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from am_mvt.config import get_path


ARCHIVE_NAME = "legacy_20260608"
CURRENT_EXPERIMENT = "cpu_fast_v1"
LEGACY_DATA_FILES = [
    "data/processed/sources.csv",
    "data/processed/build_conditions.csv",
    "data/processed/mechanical_tests.csv",
    "data/processed/extraction_audit.csv",
    "data/processed/modelling_dataset_validation_report.csv",
]


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or archive explicitly approved legacy datasets, models, "
            "and experiment results. The default is a non-mutating dry run."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move the approved legacy artifacts and write archive_manifest.csv.",
    )
    return parser.parse_args(argv)


def ensure_within_project(path: Path) -> Path:
    project_root = get_path().resolve()
    resolved = path.resolve()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"Path is outside the project workspace: {resolved}")
    return resolved


def build_move_plan() -> list[MoveItem]:
    archive_root = get_path("archive", ARCHIVE_NAME)
    moves: list[MoveItem] = []

    data_sources = [
        *sorted(get_path("data", "processed").glob("master_modelling_dataset_backup_*.csv")),
        *(get_path(*relative.split("/")) for relative in LEGACY_DATA_FILES),
    ]
    for source in data_sources:
        if source.is_file():
            moves.append(
                MoveItem(
                    source=source,
                    destination=archive_root / "data" / source.name,
                )
            )

    models_dir = get_path("outputs", "models")
    for source in sorted(models_dir.iterdir()):
        if source.name != ".gitkeep" and source.is_file():
            moves.append(
                MoveItem(
                    source=source,
                    destination=archive_root / "models" / source.name,
                )
            )

    tables_dir = get_path("outputs", "tables")
    for source in sorted(tables_dir.glob("project_*.csv")):
        moves.append(
            MoveItem(
                source=source,
                destination=archive_root / "results" / "tables" / source.name,
            )
        )

    experiments_dir = get_path("outputs", "experiments")
    for source in sorted(experiments_dir.iterdir()):
        if source.is_dir() and source.name != CURRENT_EXPERIMENT:
            moves.append(
                MoveItem(
                    source=source,
                    destination=archive_root / "results" / "experiments" / source.name,
                )
            )

    return moves


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_rows(moves: list[MoveItem]) -> list[ManifestRow]:
    project_root = get_path().resolve()
    rows: list[ManifestRow] = []

    for move in moves:
        source_files = (
            sorted(move.source.rglob("*"))
            if move.source.is_dir()
            else [move.source]
        )
        for source_file in source_files:
            if not source_file.is_file():
                continue
            relative_inside_source = (
                source_file.relative_to(move.source)
                if move.source.is_dir()
                else Path()
            )
            destination_file = move.destination / relative_inside_source
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
    seen_destinations: set[Path] = set()

    for move in moves:
        source = ensure_within_project(move.source)
        destination = ensure_within_project(move.destination)
        if not source.exists():
            raise FileNotFoundError(f"Legacy artifact disappeared: {source}")
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")
        if destination in seen_destinations:
            raise ValueError(f"Duplicate archive destination: {destination}")
        seen_destinations.add(destination)


def write_manifest(rows: list[ManifestRow]) -> Path:
    manifest_path = get_path("archive", ARCHIVE_NAME, "archive_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return manifest_path


def apply_move_plan(moves: list[MoveItem], rows: list[ManifestRow]) -> Path:
    for move in moves:
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.source), str(move.destination))
    return write_manifest(rows)


def main() -> None:
    args = parse_args()
    moves = build_move_plan()
    validate_move_plan(moves)
    rows = build_manifest_rows(moves)

    action = "APPLY" if args.apply else "DRY RUN"
    print(f"{action}: {len(moves)} move items, {len(rows)} files")
    for move in moves:
        print(
            f"{move.source.relative_to(get_path()).as_posix()} -> "
            f"{move.destination.relative_to(get_path()).as_posix()}"
        )

    if not args.apply:
        print("\nNo files were moved. Re-run with --apply after reviewing the list.")
        return

    if not rows:
        raise RuntimeError("No legacy files matched the approved archive whitelist.")

    manifest_path = apply_move_plan(moves, rows)
    print(f"\nArchive complete: {manifest_path.parent}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
