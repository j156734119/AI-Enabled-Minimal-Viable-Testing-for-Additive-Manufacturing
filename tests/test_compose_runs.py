from __future__ import annotations

import json

import pandas as pd
import pytest

from am_mvt.modelling.compose_runs import compose_experiment_runs


def _write_run(path, candidate: str) -> None:
    (path / "models").mkdir(parents=True)
    (path / "tables").mkdir()
    artifact = path / "models" / "fatigue.json"
    artifact.write_text(candidate, encoding="utf-8")
    registry = [
        {
            "model_key": "model2_sn_fatigue",
            "target": "log10_fatigue_life_cycles",
            "mode": "process_only",
            "route": "xgboost_aft",
            "candidate": candidate,
            "artifact": "models/fatigue.json",
        }
    ]
    (path / "model_registry.json").write_text(
        json.dumps(registry),
        encoding="utf-8",
    )
    (path / "run_config.json").write_text(
        json.dumps(
            {
                "task_configs": [
                    {
                        "model_key": "model2_sn_fatigue",
                        "target": "log10_fatigue_life_cycles",
                        "mode": "process_only",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    row = {
        "model_key": "model2_sn_fatigue",
        "target": "log10_fatigue_life_cycles",
        "mode": "process_only",
        "route": "xgboost_aft",
        "candidate": candidate,
    }
    for filename in [
        "experiment_metrics.csv",
        "experiment_summary.csv",
        "oof_predictions.csv",
        "physical_checks.csv",
    ]:
        pd.DataFrame([row]).to_csv(path / "tables" / filename, index=False)
    pd.DataFrame(
        [
            {
                **row,
                "feature": "surface_condition",
                "importance_fraction": 0.1 if candidate == "legacy" else 0.2,
            }
        ]
    ).to_csv(path / "tables" / "feature_importance.csv", index=False)


def test_compose_can_explicitly_replace_conflicting_routes(tmp_path):
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    output = tmp_path / "output"
    _write_run(base, "legacy")
    _write_run(supplement, "protocol_aware")

    compose_experiment_runs(
        base,
        supplement,
        output,
        replace_conflicts=True,
    )

    registry = json.loads((output / "model_registry.json").read_text())
    summary = pd.read_csv(output / "tables" / "experiment_summary.csv")
    assert registry[0]["candidate"] == "protocol_aware"
    assert summary["candidate"].tolist() == ["protocol_aware"]
    importance = pd.read_csv(output / "tables" / "feature_importance.csv")
    assert importance["importance_fraction"].tolist() == [0.2]
    assert (output / "models" / "fatigue.json").read_text() == "protocol_aware"


def test_compose_still_rejects_conflicts_by_default(tmp_path):
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    _write_run(base, "legacy")
    _write_run(supplement, "protocol_aware")

    with pytest.raises(ValueError, match="Duplicate model route"):
        compose_experiment_runs(base, supplement, tmp_path / "output")
