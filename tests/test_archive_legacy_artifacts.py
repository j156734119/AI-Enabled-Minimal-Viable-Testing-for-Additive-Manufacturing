from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_archive_module():
    script_path = PROJECT_ROOT / "scripts" / "archive_legacy_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "archive_legacy_artifacts",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_archive_plan_moves_only_legacy_whitelist(tmp_path, monkeypatch):
    module = load_archive_module()
    monkeypatch.setattr(module, "get_path", lambda *parts: tmp_path.joinpath(*parts))

    processed = tmp_path / "data" / "processed"
    models = tmp_path / "outputs" / "models"
    tables = tmp_path / "outputs" / "tables"
    experiments = tmp_path / "outputs" / "experiments"
    for directory in [processed, models, tables, experiments]:
        directory.mkdir(parents=True)

    legacy_data = processed / "sources.csv"
    legacy_backup = processed / "master_modelling_dataset_backup_20260606.csv"
    current_data = processed / "master_modelling_dataset.csv"
    legacy_model = models / "old_model.joblib"
    current_run = experiments / "balanced_v2"
    legacy_run = experiments / "smoke_old"
    legacy_table = tables / "project_regression_model_metrics.csv"
    current_table = tables / "source_provenance_audit.csv"

    for path in [
        legacy_data,
        legacy_backup,
        current_data,
        legacy_model,
        legacy_table,
        current_table,
    ]:
        path.write_text(path.name, encoding="utf-8")
    current_run.mkdir()
    (current_run / "model_registry.json").write_text("current", encoding="utf-8")
    legacy_run.mkdir()
    (legacy_run / "smoke.txt").write_text("legacy", encoding="utf-8")

    moves = module.build_move_plan(
        archive_name="cleanup_test",
        keep_experiment="balanced_v2",
    )
    module.validate_move_plan(moves)
    rows = module.build_manifest_rows(moves)
    planned_sources = {move.source for move in moves}

    assert legacy_data in planned_sources
    assert legacy_backup in planned_sources
    assert legacy_model in planned_sources
    assert legacy_table in planned_sources
    assert legacy_run in planned_sources
    assert current_data not in planned_sources
    assert current_table not in planned_sources
    assert current_run not in planned_sources
    assert all(len(row.sha256) == 64 for row in rows)

    manifest_path = module.apply_move_plan(
        moves,
        rows,
        archive_name="cleanup_test",
    )

    assert manifest_path.is_file()
    assert current_data.is_file()
    assert current_table.is_file()
    assert (current_run / "model_registry.json").is_file()
    assert not legacy_data.exists()
    assert not legacy_model.exists()
    assert not legacy_run.exists()


def test_archive_plan_can_include_duplicate_derivatives(tmp_path, monkeypatch):
    module = load_archive_module()
    monkeypatch.setattr(module, "get_path", lambda *parts: tmp_path.joinpath(*parts))

    duplicate = (
        tmp_path
        / "data"
        / "interim"
        / "text_chunks"
        / f"{module.DUPLICATE_SOURCE_PREFIXES[0]}_chunk_0000.txt"
    )
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text("duplicate", encoding="utf-8")

    moves = module.build_move_plan(
        archive_name="cleanup_test",
        include_duplicate_derivatives=True,
    )

    assert duplicate.resolve() in {move.source for move in moves}
    assert all("cleanup_test" in str(move.destination) for move in moves)
