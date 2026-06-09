from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PATTERN = re.compile(
    r"^(feat|fix|docs|test|refactor|perf|build|ci|chore)"
    r"(\([A-Za-z0-9][A-Za-z0-9._/-]*\))?!?: \S.*$"
)

ERROR_MESSAGE = """ERROR: invalid commit message.

The first line must follow Conventional Commits:
  <type>: <description>
  <type>(<scope>): <description>
  <type>!: <description>
  <type>(<scope>)!: <description>

Allowed types:
  feat|fix|docs|test|refactor|perf|build|ci|chore

Valid examples:
  feat(extraction): retain page-level evidence metadata
  fix(training): prevent leakage during preprocessing
  docs: clarify lawful PDF acquisition workflow
  refactor!: replace the legacy modelling interface
"""


def is_valid_commit_message(message: str) -> bool:
    first_line = message.splitlines()[0].rstrip("\r") if message else ""
    return PATTERN.fullmatch(first_line) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("message_file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    message = args.message_file.read_text(encoding="utf-8")
    if is_valid_commit_message(message):
        return 0
    print(ERROR_MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
