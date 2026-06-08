from pathlib import Path

import pytest

from am_mvt.skill_loader import (
    build_skill_system_prompt,
    canonical_skill_name,
    load_project_skill,
    parse_skill_text,
)


def test_parse_skill_text_removes_frontmatter():
    skill = parse_skill_text(
        (
            "---\n"
            "name: example-skill\n"
            "description: Use for an example task.\n"
            "---\n"
            "# Goal\n"
            "Perform the task.\n"
        ),
        Path("SKILL.md"),
    )

    assert skill.name == "example-skill"
    assert skill.body.startswith("# Goal")
    assert "description:" not in skill.body


def test_loader_requires_hyphen_case_request_name():
    with pytest.raises(FileNotFoundError):
        load_project_skill("source_screening")


def test_build_skill_system_prompt_injects_body_only():
    prompt = build_skill_system_prompt("Base prompt.", "source-screening")
    assert "Base prompt." in prompt
    assert "# Goal" in prompt
    assert "name: source-screening" not in prompt


def test_invalid_frontmatter_is_rejected():
    with pytest.raises(ValueError, match="Unexpected"):
        parse_skill_text(
            (
                "---\n"
                "name: example-skill\n"
                "description: Example.\n"
                "metadata: invalid\n"
                "---\n"
                "Body\n"
            ),
            Path("SKILL.md"),
        )


def test_canonical_skill_name_preserves_hyphen_case():
    assert canonical_skill_name("Evidence-Grounded-Extraction") == (
        "evidence-grounded-extraction"
    )
