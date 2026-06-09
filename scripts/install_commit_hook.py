from __future__ import annotations

import subprocess
from pathlib import Path


HOOK = """#!/bin/sh
python scripts/validate_commit_message.py "$1"
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    hook_dir = root / ".githooks"
    hook_dir.mkdir(exist_ok=True)
    hook_path = hook_dir / "commit-msg"
    hook_path.write_text(HOOK, encoding="ascii", newline="\n")
    subprocess.run(
        ["git", "config", "core.hooksPath", ".githooks"],
        cwd=root,
        check=True,
    )
    print(f"Installed local commit hook: {hook_path}")


if __name__ == "__main__":
    main()
