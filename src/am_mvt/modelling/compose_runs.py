from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CORE_TABLES = [
    "experiment_metrics.csv",
    "experiment_summary.csv",
    "oof_predictions.csv",
    "physical_checks.csv",
]

MODEL_SCOPED_TABLES = [
    "feature_importance.csv",
    "grouped_error_analysis.csv",
    "variable_coverage.csv",
    "combination_coverage.csv",
    "sensitivity_analysis.csv",
    "shap_importance.csv",
    "shap_values_sample.csv",
    "feature_importance_comparison.csv",
    "b2_combination_diagnostics.csv",
    "relationship_evidence.csv",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _registry_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(entry.get("model_key", "")),
        str(entry.get("target", "")),
        str(entry.get("mode", "")),
        str(entry.get("route", "")),
    )


def _copy_registry_artifacts(
    source: Path,
    destination: Path,
    entries: list[dict[str, Any]],
) -> None:
    for entry in entries:
        artifact_references = [
            entry.get("artifact"),
            entry.get("preprocessor_artifact"),
        ]
        for domain_model in entry.get("domain_models", []):
            artifact_references.extend(
                [
                    domain_model.get("artifact"),
                    domain_model.get("preprocessor_artifact"),
                ]
            )
        for relative in artifact_references:
            if not relative:
                continue
            source_path = source / str(relative)
            destination_path = destination / str(relative)
            if not source_path.exists():
                raise FileNotFoundError(f"Missing registry artifact: {source_path}")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def _merge_model_scoped_table(
    base_path: Path,
    supplement_path: Path,
    *,
    replace_conflicts: bool,
) -> pd.DataFrame:
    base_frame = _read_csv(base_path)
    supplement_frame = _read_csv(supplement_path)
    if base_frame.empty:
        return supplement_frame
    if supplement_frame.empty:
        return base_frame
    key_columns = [
        column
        for column in ["model_key", "target", "mode", "route"]
        if column in base_frame.columns and column in supplement_frame.columns
    ]
    if replace_conflicts and key_columns:
        supplement_keys = set(
            supplement_frame[key_columns].astype(str).apply(tuple, axis=1)
        )
        base_keys = base_frame[key_columns].astype(str).apply(tuple, axis=1)
        base_frame = base_frame.loc[~base_keys.isin(supplement_keys)]
    return pd.concat(
        [base_frame, supplement_frame],
        ignore_index=True,
        sort=False,
    )


def compose_experiment_runs(
    base_run: str | Path,
    supplement_run: str | Path,
    output_run: str | Path,
    *,
    replace_conflicts: bool = False,
) -> Path:
    base = Path(base_run)
    supplement = Path(supplement_run)
    output = Path(output_run)
    if output.exists():
        raise FileExistsError(f"Output run already exists: {output}")
    if not base.exists() or not supplement.exists():
        raise FileNotFoundError("Both base and supplement runs must exist.")

    shutil.copytree(base, output)
    base_registry = json.loads(
        (base / "model_registry.json").read_text(encoding="utf-8")
    )
    supplement_registry = json.loads(
        (supplement / "model_registry.json").read_text(encoding="utf-8")
    )
    combined: dict[tuple[str, str, str, str], dict[str, Any]] = {
        _registry_key(entry): entry for entry in base_registry
    }
    for entry in supplement_registry:
        key = _registry_key(entry)
        if key in combined and not replace_conflicts:
            raise ValueError(f"Duplicate model route while composing runs: {key}")
        combined[key] = entry
    _copy_registry_artifacts(supplement, output, supplement_registry)
    (output / "model_registry.json").write_text(
        json.dumps(list(combined.values()), indent=2),
        encoding="utf-8",
    )

    for filename in CORE_TABLES:
        base_frame = _read_csv(base / "tables" / filename)
        if replace_conflicts and not base_frame.empty:
            conflict_keys = {_registry_key(entry) for entry in supplement_registry}
            key_columns = ["model_key", "target", "mode", "route"]
            if all(column in base_frame for column in key_columns):
                row_keys = base_frame[key_columns].astype(str).apply(tuple, axis=1)
                base_frame = base_frame.loc[~row_keys.isin(conflict_keys)]
        parts = [
            frame
            for frame in [
                base_frame,
                _read_csv(supplement / "tables" / filename),
            ]
            if not frame.empty
        ]
        merged = (
            pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        )
        if filename == "physical_checks.csv" and not merged.empty:
            merged = merged.drop_duplicates()
        merged.to_csv(
            output / "tables" / filename,
            index=False,
            encoding="utf-8-sig",
        )
    for filename in MODEL_SCOPED_TABLES:
        merged = _merge_model_scoped_table(
            base / "tables" / filename,
            supplement / "tables" / filename,
            replace_conflicts=replace_conflicts,
        )
        if not merged.empty:
            merged.to_csv(
                output / "tables" / filename,
                index=False,
                encoding="utf-8-sig",
            )
    for source_table in (supplement / "tables").glob("*"):
        if (
            source_table.name in CORE_TABLES
            or source_table.name in MODEL_SCOPED_TABLES
            or not source_table.is_file()
        ):
            continue
        shutil.copy2(source_table, output / "tables" / source_table.name)

    base_config = json.loads((base / "run_config.json").read_text(encoding="utf-8"))
    supplement_config = json.loads(
        (supplement / "run_config.json").read_text(encoding="utf-8")
    )
    task_configs = {
        (item["model_key"], item["target"], item["mode"]): item
        for item in base_config.get("task_configs", [])
    }
    task_configs.update(
        {
            (item["model_key"], item["target"], item["mode"]): item
            for item in supplement_config.get("task_configs", [])
        }
    )
    base_config.update(
        {
            "run_name": output.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_targets": sorted(
                {entry["target"] for entry in combined.values()}
            ),
            "task_configs": list(task_configs.values()),
            "composed_from": [str(base.resolve()), str(supplement.resolve())],
            "replace_conflicts": replace_conflicts,
        }
    )
    (output / "run_config.json").write_text(
        json.dumps(base_config, indent=2),
        encoding="utf-8",
    )
    return output
