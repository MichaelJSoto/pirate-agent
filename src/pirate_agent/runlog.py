"""Run logs: the JSON record of every training run, and the run folders that hold them.

Programmer: Michael Soto
Date: November 15, 2026
Version: 2.1.1
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Make every result reproducible and checkable. A run folder holds the saved model, a
    per-game history, and a run log with the settings, seed, maze, package versions and
    results, plus the SHA-256 that ties the model file to that run.

Decision:
    In CS 370 the trained model was never saved and every number lived only in notebook
    output (code review finding S6). A run log is now written atomically, to a temporary file
    that is then renamed, so a crash never leaves half a log beside a model. Reading one is
    treated like reading any other file from disk: its size is capped, it must parse as
    JSON, and every field is checked before use. The model's file name must be exactly
    model.keras, so an edited log cannot point the loader at a file outside its run folder.
    Paths are stored relative to the run folder, so a log never records a user name or a
    home directory.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import tempfile
from collections.abc import Iterable
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__
from .maze import Maze, parse_maze

SCHEMA_VERSION = 1
RUN_LOG_NAME = "run.json"
MODEL_NAME = "model.keras"
HISTORY_NAME = "history.csv"
MAX_LOG_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


class RunLogError(ValueError):
    """A run folder or run log that cannot be used."""


def new_run_id(seed: int, now: datetime | None = None) -> str:
    """A sortable, readable run name such as 20261110-153000-seed1671."""
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-seed{seed}"


def make_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Creates a new, empty run folder. An existing folder is never reused or overwritten."""
    path = Path(runs_dir) / run_id
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RunLogError(f"run folder {run_id} already exists") from None
    return path


def package_versions() -> dict[str, str]:
    """The versions behind every number a run reports."""
    versions = {"python": platform.python_version(), "pirate_agent": __version__}
    for name in ("numpy", "tensorflow", "keras"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def maze_record(maze: Maze, source: str) -> dict[str, Any]:
    """The maze as stored in a run log: its layout text, so evaluation needs no other file."""
    return {
        "source": source,
        "rows": maze.shape[0],
        "cols": maze.shape[1],
        "target": list(maze.target),
        "layout": maze.to_text().splitlines(),
    }


def write_run_log(path: Path, record: dict[str, Any]) -> None:
    """Writes the log atomically: a temporary file in the same folder, then a rename."""
    path = Path(path)
    text = json.dumps(record, indent=2) + "\n"
    # Same folder as the final file, so os.replace() is a rename on one disk and cannot half-finish.
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".run-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        # Clean up the temporary file on any failure, even Ctrl+C, then let the error continue.
        Path(temporary).unlink(missing_ok=True)
        raise


def _require(record: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str = "run log") -> Any:
    """Returns record[key] if it exists and has the expected type, else raises RunLogError."""
    if key not in record:
        raise RunLogError(f"{where} is missing '{key}'")
    value = record[key]
    # bool is a subclass of int in Python, so True would pass as a number without the second test.
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise RunLogError(f"{where} field '{key}' has the wrong type")
    return value


def read_run_log(path: Path) -> dict[str, Any]:
    """Reads and checks a run log. Raises RunLogError for anything missing, malformed or unsafe."""
    path = Path(path)
    if not path.is_file():
        raise RunLogError(f"no run log at {path.parent.name}/{path.name}")
    # A run log is untrusted input like any other file: cap its size before reading it.
    if path.stat().st_size > MAX_LOG_BYTES:
        raise RunLogError(f"{path.name} is larger than {MAX_LOG_BYTES:,} bytes")
    try:
        record = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RunLogError(f"{path.name} is not valid JSON") from None
    if not isinstance(record, dict):
        raise RunLogError(f"{path.name} must hold a JSON object")
    if _require(record, "schema_version", int) != SCHEMA_VERSION:
        raise RunLogError(
            f"{path.name} uses schema {record['schema_version']}; this program reads {SCHEMA_VERSION}"
        )
    _require(record, "run_id", str)
    _require(record, "training", dict)
    maze = _require(record, "maze", dict)
    layout = _require(maze, "layout", list, "maze record")
    if not all(isinstance(line, str) for line in layout):
        raise RunLogError("maze record field 'layout' must be a list of text rows")
    # Only the fixed name is accepted, so an edited log cannot point the loader at another file.
    model = _require(record, "model", dict)
    if _require(model, "file", str, "model record") != MODEL_NAME:
        raise RunLogError(f"the model file must be named {MODEL_NAME}, in the run folder itself")
    if not _SHA256.fullmatch(_require(model, "sha256", str, "model record")):
        raise RunLogError("model record field 'sha256' is not a SHA-256 hex digest")
    return record


def maze_from_log(record: dict[str, Any]) -> Maze:
    """Rebuilds the run's maze from its stored layout, validating it again."""
    return parse_maze("\n".join(record["maze"]["layout"]), source="the run log's maze")


def write_history(path: Path, history: Iterable[Any]) -> None:
    """One CSV row per training game: epoch, loss, moves, won, win_rate, epsilon."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["epoch", "loss", "moves", "won", "win_rate", "epsilon"])
        for row in history:
            writer.writerow(
                [
                    row.epoch,
                    f"{row.loss:.6f}",
                    row.moves,
                    int(row.won),
                    f"{row.win_rate:.4f}",
                    f"{row.epsilon:.5f}",
                ]
            )
