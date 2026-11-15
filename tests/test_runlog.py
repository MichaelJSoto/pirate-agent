"""Tests for run folders, run logs and the training history file.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Show that a run log survives a round trip and that a damaged or edited log is refused
    before anything in it is trusted, including an attempt to point the loader elsewhere.

Decision:
    Every refusal is tested on a log that is valid except for the one field under test.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime

import pytest

from pirate_agent.runlog import (
    MAX_LOG_BYTES,
    RunLogError,
    make_run_dir,
    maze_from_log,
    maze_record,
    new_run_id,
    package_versions,
    read_run_log,
    write_history,
    write_run_log,
)
from pirate_agent.trainer import EpochRecord

GOOD_HASH = "ab" * 32


def record(maze):
    return {
        "schema_version": 1,
        "run_id": "20261110-120000-seed1",
        "training": {"seed": 1},
        "maze": maze_record(maze, "test"),
        "model": {"file": "model.keras", "sha256": GOOD_HASH},
    }


def test_round_trip(tmp_path, cs370_maze):
    path = tmp_path / "run.json"
    write_run_log(path, record(cs370_maze))
    loaded = read_run_log(path)
    assert loaded["model"]["sha256"] == GOOD_HASH
    rebuilt = maze_from_log(loaded)
    assert (rebuilt.grid == cs370_maze.grid).all()
    assert list(tmp_path.iterdir()) == [path]  # the temporary file was renamed, not left behind


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.pop("model"), "missing 'model'"),
        (lambda r: r.update(schema_version=2), "uses schema 2"),
        (lambda r: r["model"].update(file="../../elsewhere.keras"), "must be named model.keras"),
        (lambda r: r["model"].update(sha256="not-a-hash"), "not a SHA-256"),
        (lambda r: r["maze"].update(layout="#.T"), "wrong type"),
        (lambda r: r["maze"].update(layout=["#.", "x."]), "unexpected character 'x'"),
    ],
)
def test_damaged_logs_are_refused(tmp_path, cs370_maze, change, message):
    bad = copy.deepcopy(record(cs370_maze))
    change(bad)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        maze_from_log(read_run_log(path))


def test_not_json_and_too_big(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RunLogError, match="not valid JSON"):
        read_run_log(path)
    path.write_text(" " * (MAX_LOG_BYTES + 1), encoding="utf-8")
    with pytest.raises(RunLogError, match="larger than"):
        read_run_log(path)
    with pytest.raises(RunLogError, match="no run log"):
        read_run_log(tmp_path / "missing" / "run.json")


def test_run_folders_are_never_reused(tmp_path):
    run_id = new_run_id(1671, datetime(2026, 11, 10, 15, 30, 0))
    assert run_id == "20261110-153000-seed1671"
    make_run_dir(tmp_path, run_id)
    with pytest.raises(RunLogError, match="already exists"):
        make_run_dir(tmp_path, run_id)


def test_history_csv(tmp_path):
    path = tmp_path / "history.csv"
    write_history(path, [EpochRecord(0, 0.5, 12, True, 0.25, 1.0)])
    assert path.read_text(encoding="utf-8").splitlines() == [
        "epoch,loss,moves,won,win_rate,epsilon",
        "0,0.500000,12,1,0.2500,1.00000",
    ]


def test_versions_name_every_package():
    assert set(package_versions()) == {"python", "pirate_agent", "numpy", "tensorflow", "keras"}
