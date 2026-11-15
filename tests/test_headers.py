"""Checks that every code and configuration file opens with the required header.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Make the header rule (programmer, date, version, intent and decision in every file) a test,
    so a new file without one fails CI instead of waiting for a reviewer to notice.

Decision:
    The check reads only the first 40 lines, where a header belongs, and skips original/,
    because the CS 370 files are kept exactly as they were submitted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("Programmer: Michael Soto", "Date:", "Version:", "Intent:", "Decision:")
SKIP = {"original", ".git", ".venv", "runs", "__pycache__"}
FILES = sorted(
    path
    for pattern in ("*.py", "*.toml", "*.yml", "requirements*.txt")
    for path in ROOT.rglob(pattern)
    if not SKIP.intersection(path.relative_to(ROOT).parts)
)


def test_the_check_found_the_project_files():
    names = {path.name for path in FILES}
    assert {"maze.py", "agent.py", "cli.py", "default.toml", "ci.yml", "requirements.txt"} <= names


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_file_has_the_header(path):
    head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:40])
    missing = [field for field in FIELDS if field not in head]
    assert not missing, f"{path.name} is missing {missing}"
