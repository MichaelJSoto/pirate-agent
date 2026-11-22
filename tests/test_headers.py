"""Checks the file headers and the docstring on every function, method, class and test.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Make two project rules tests, so a file or a function that breaks one fails CI instead of
    waiting for a reviewer to notice: every file opens with the header (programmer, date,
    version, intent and decision), and every definition in the Python code has a docstring.

Decision:
    The header check reads only the first 40 lines, where a header belongs, and skips
    original/, because the CS 370 files are kept exactly as they were submitted. Ruff's D1
    rules already require docstrings on public definitions, but pydocstyle skips private
    (_name) and nested functions by design, so the docstring check walks the syntax tree with
    the standard library's ast module instead of adding a tool. It is checked on a sample of
    its own first, so it cannot pass by finding nothing.
"""

from __future__ import annotations

import ast
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
PYTHON_FILES = [path for path in FILES if path.suffix == ".py"]
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def missing_docstrings(source: str) -> tuple[int, list[str]]:
    """Counts the definitions in `source` and lists those with no docstring, in line order.

    ast.walk visits every node once, so nested functions, methods and private helpers are all
    found, at O(n) in the size of the file. A lambda cannot hold a docstring, so it is not a
    definition here.
    """
    found = sorted(
        (node for node in ast.walk(ast.parse(source)) if isinstance(node, DEFINITIONS)),
        key=lambda node: node.lineno,
    )
    missing = [f"line {node.lineno} {node.name}" for node in found if ast.get_docstring(node) is None]
    return len(found), missing


def test_the_check_found_the_project_files():
    """The file search reaches the package, the configs, CI and the requirements.

    Without this, the header test could pass by checking nothing.
    """
    names = {path.name for path in FILES}
    assert {"maze.py", "agent.py", "cli.py", "default.toml", "ci.yml", "requirements.txt"} <= names


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_file_has_the_header(path):
    """Each code and configuration file opens with the programmer, date, version, intent and decision.

    The CS 370 header had a name and a date but no version, intent or decision (finding S6).
    """
    head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:40])
    missing = [field for field in FIELDS if field not in head]
    assert not missing, f"{path.name} is missing {missing}"


def test_docstring_check_finds_what_ruff_skips():
    """The checker reports a private helper, a nested function and a bare class, not a documented one."""
    sample = (
        "def _private():\n"
        "    return 1\n"
        "\n"
        "def documented():\n"
        '    """Says why it exists."""\n'
        "    def nested():\n"
        "        return 2\n"
        "    return nested()\n"
        "\n"
        "class Bare:\n"
        "    handler = lambda self: 3\n"
    )
    assert missing_docstrings(sample) == (4, ["line 1 _private", "line 6 nested", "line 10 Bare"])


def test_every_definition_has_a_docstring():
    """Every function, method, class and test in src/, scripts/ and tests/ has a docstring.

    CONTRIBUTING.md sets the rule. Ruff fails CI on a public definition without one; this test
    covers the private and nested ones Ruff skips, so a new helper without a docstring fails CI
    too.
    """
    total, gaps = 0, []
    for path in PYTHON_FILES:
        count, missing = missing_docstrings(path.read_text(encoding="utf-8"))
        total += count
        gaps += [f"{path.relative_to(ROOT).as_posix()} {gap}" for gap in missing]
    assert not gaps, f"{len(gaps)} definition(s) have no docstring:\n" + "\n".join(gaps)
    assert total > 250  # guards against a search that silently finds nothing
