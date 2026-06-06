from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from am_mvt.config import get_path


@dataclass(frozen=True)
class ProjectSkill:
    name: str
    description: str
    body: str
    path: Path


def canonical_skill_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def skill_path_candidates(name: str) -> list[Path]:
    canonical_name = canonical_skill_name(name)
    legacy_name = canonical_name.replace("-", "_")
    candidates = [
        get_path("skills", canonical_name, "SKILL.md"),
        get_path("skills", legacy_name, "SKILL.md"),
    ]
    return list(dict.fromkeys(candidates))


def parse_skill_text(text: str, path: Path) -> ProjectSkill:
    if not text.startswith("---"):
        raise ValueError(f"Skill is missing YAML frontmatter: {path}")

    parts = text.split("---", 2)

    if len(parts) != 3:
        raise ValueError(f"Skill frontmatter is not closed correctly: {path}")

    metadata = yaml.safe_load(parts[1]) or {}

    if not isinstance(metadata, dict):
        raise ValueError(f"Skill frontmatter must be a YAML mapping: {path}")

    allowed_fields = {"name", "description"}
    unexpected_fields = set(metadata) - allowed_fields

    if unexpected_fields:
        fields = ", ".join(sorted(unexpected_fields))
        raise ValueError(f"Unexpected skill frontmatter fields in {path}: {fields}")

    name = str(metadata.get("name") or "").strip()
    description = str(metadata.get("description") or "").strip()
    body = parts[2].strip()

    if not name or not description:
        raise ValueError(f"Skill frontmatter requires name and description: {path}")

    if canonical_skill_name(name) != name:
        raise ValueError(f"Skill name must use lowercase hyphen-case: {path}")

    if not body:
        raise ValueError(f"Skill body is empty: {path}")

    return ProjectSkill(
        name=name,
        description=description,
        body=body,
        path=path,
    )


def load_project_skill(name: str) -> ProjectSkill:
    candidates = skill_path_candidates(name)

    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            skill = parse_skill_text(text, path)
            requested_name = canonical_skill_name(name)

            if skill.name != requested_name:
                raise ValueError(
                    f"Skill name {skill.name!r} does not match requested "
                    f"name {requested_name!r}: {path}"
                )

            return skill

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Required project skill not found. Searched: {searched}")


def build_skill_system_prompt(base_prompt: str, skill_name: str) -> str:
    skill = load_project_skill(skill_name)
    return (
        f"{base_prompt.strip()}\n\n"
        f"Apply the following repository skill as the bounded task "
        f"specification.\n\n{skill.body}"
    )
