from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_commit_message_hook(tmp_path: Path, message: str) -> subprocess.CompletedProcess:
    message_path = tmp_path / "COMMIT_EDITMSG"
    message_path.write_text(message, encoding="utf-8")
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=.githooks",
            "hook",
            "run",
            "commit-msg",
            "--",
            str(message_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "message",
    [
        "feat: add source screening",
        "fix: prevent leakage",
        "docs: document the workflow",
        "test: cover the commit hook",
        "refactor: simplify model selection",
        "perf: reduce preprocessing time",
        "build: update package metadata",
        "ci: run commit message checks",
        "chore: maintain repository settings",
        "fix(training): prevent leakage",
        "docs!: replace the public workflow",
        "refactor(model-comparison)!: change the registry schema",
        "test(api_v2): cover empty results\n\nAdditional context.",
    ],
)
def test_commit_message_hook_accepts_valid_messages(tmp_path, message):
    result = run_commit_message_hook(tmp_path, message)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "message",
    [
        "",
        "feat:",
        "feat: ",
        "feature: add source screening",
        "Fix: use grouped validation",
        "feat(scope) add missing colon",
        "feat(): empty scope",
        "feat(scope):    ",
        "plain commit message",
    ],
)
def test_commit_message_hook_rejects_invalid_messages(tmp_path, message):
    result = run_commit_message_hook(tmp_path, message)

    assert result.returncode != 0
    assert "invalid commit message" in result.stderr
    assert "feat(extraction): retain page-level evidence metadata" in result.stderr
