from __future__ import annotations

import pytest

from scripts.validate_commit_message import is_valid_commit_message


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
def test_commit_message_hook_accepts_valid_messages(message):
    assert is_valid_commit_message(message)


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
def test_commit_message_hook_rejects_invalid_messages(message):
    assert not is_valid_commit_message(message)
