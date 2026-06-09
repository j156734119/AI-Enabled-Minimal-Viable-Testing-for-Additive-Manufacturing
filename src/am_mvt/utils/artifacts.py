from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from am_mvt.config import get_path


@dataclass(frozen=True)
class ArchivedFile:
    original_path: str
    archived_path: str
    size_bytes: int
    modified_at_utc: str
    sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_archive_name(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def ensure_within_project(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else get_path().resolve()
    )
    resolved = Path(path).resolve()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"Path is outside the project workspace: {resolved}")
    return resolved


def archive_files(
    paths: list[Path],
    archive_root: Path,
    *,
    manifest_name: str = "archive_manifest.csv",
    project_root: str | Path | None = None,
) -> Path | None:
    project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else get_path().resolve()
    )
    archive_root = ensure_within_project(
        archive_root,
        project_root=project_root,
    )
    files = sorted(
        {
            ensure_within_project(path, project_root=project_root)
            for path in paths
            if Path(path).is_file()
        }
    )
    if not files:
        return None

    rows: list[ArchivedFile] = []
    destinations: list[tuple[Path, Path]] = []
    for source in files:
        relative = source.relative_to(project_root)
        destination = archive_root / relative
        if destination.exists():
            raise FileExistsError(f"Archive destination exists: {destination}")
        stat = source.stat()
        rows.append(
            ArchivedFile(
                original_path=relative.as_posix(),
                archived_path=destination.relative_to(project_root).as_posix(),
                size_bytes=stat.st_size,
                modified_at_utc=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                sha256=sha256_file(source),
            )
        )
        destinations.append((source, destination))

    for source, destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

    manifest_path = archive_root / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return manifest_path
