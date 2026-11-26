"""Tests for the command-line interface, from argument errors to a full train and evaluate cycle.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Check each command's exit code and message the way a user or a script would see them,
    including the refusal of a model file that was changed after training.

Decision:
    The end-to-end test trains on a tiny generated maze in a temporary folder, so it runs in
    seconds and never touches the repository's runs/ folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pirate_agent.cli import EXIT_INPUT, EXIT_MODEL, EXIT_OK, main

ROOT = Path(__file__).resolve().parents[1]


def test_generate_writes_a_valid_maze(tmp_path, capsys):
    """generate writes a commented .maze file, reports how much is open, and overwrites only with --force."""
    out = tmp_path / "new.maze"
    assert main(["generate", "--rows", "5", "--cols", "6", "--seed", "3", "--out", str(out)]) == EXIT_OK
    assert out.read_text(encoding="ascii").startswith("; Generated 5 x 6 maze")
    assert "cells are open" in capsys.readouterr().out
    # A second run refuses to overwrite the file unless told to.
    assert main(["generate", "--out", str(out)]) == EXIT_INPUT
    assert main(["generate", "--out", str(out), "--force"]) == EXIT_OK


def test_generate_refuses_a_bad_suffix(tmp_path, caplog):
    """generate refuses an output name that does not end in .maze, with the input-error exit code."""
    assert main(["generate", "--out", str(tmp_path / "maze.txt")]) == EXIT_INPUT
    assert "must end in '.maze'" in caplog.text


def test_train_reports_a_bad_maze_before_loading_tensorflow(tmp_path, caplog):
    """A configuration whose maze has a walled-in treasure exits with EXIT_INPUT and the maze's own message.

    train validates the maze before it imports TensorFlow, so a typo in a maze file is reported
    without the wait for TensorFlow to load.
    """
    (tmp_path / "bad.maze").write_text("..\n.#\n", encoding="ascii")  # the treasure is a wall
    (tmp_path / "bad.toml").write_text('[maze]\nfile = "bad.maze"\n', encoding="utf-8")
    assert main(["train", "--config", str(tmp_path / "bad.toml")]) == EXIT_INPUT
    assert "treasure cell (1, 1) is a wall" in caplog.text


def test_missing_config_and_run(tmp_path, caplog):
    """A missing configuration file or run folder is an input error with a message, not a stack trace."""
    assert main(["train", "--config", str(tmp_path / "none.toml")]) == EXIT_INPUT
    assert main(["evaluate", "--run", str(tmp_path / "no-run")]) == EXIT_INPUT
    assert "is not a run folder" in caplog.text


def test_usage_errors_exit_with_code_2():
    """A missing required argument exits with argparse's usage code 2, apart from the project's codes."""
    with pytest.raises(SystemExit) as error:
        main(["train"])
    assert error.value.code == 2


@pytest.mark.tf
def test_train_evaluate_play_and_tamper(tmp_path, capsys, caplog):
    """The whole cycle on a tiny maze, as a user would run it.

    train saves a solved run whose log holds no home folder path, evaluate verifies the model's
    hash, play reaches the treasure, and a model changed after training is refused with
    EXIT_MODEL. The CS 370 results lived only in notebook output (finding S6); these run
    folders are what replaced it.
    """
    config = tmp_path / "tiny.toml"
    config.write_text(
        "[maze.generate]\nrows = 3\ncols = 3\nwall_density = 0.0\nseed = 1\n"
        "[training]\nseed = 2\nmax_epochs = 400\npatience = 200\nlog_every = 1000\n"
        '[output]\nruns_dir = "runs"\n',
        encoding="utf-8",
    )
    assert main(["-q", "train", "--config", str(config)]) == EXIT_OK
    assert "Stopped: solved" in capsys.readouterr().out
    run_dir = next((tmp_path / "runs").iterdir())
    log = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert log["result"]["stop_reason"] == "solved"
    assert log["audit"]["wins"] == log["audit"]["cells"] == 8
    assert str(Path.home()) not in (run_dir / "run.json").read_text(encoding="utf-8")

    assert main(["-q", "evaluate", "--run", str(run_dir)]) == EXIT_OK
    assert "model hash verified" in capsys.readouterr().out

    assert main(["-q", "play", "--run", str(run_dir), "--start", "0,0"]) == EXIT_OK
    assert "Reached the treasure from (0, 0)" in capsys.readouterr().out
    assert main(["-q", "play", "--run", str(run_dir), "--start", "zero"]) == EXIT_INPUT

    model = run_dir / "model.keras"
    model.write_bytes(model.read_bytes() + b"\0")  # change the file after training
    assert main(["-q", "evaluate", "--run", str(run_dir)]) == EXIT_MODEL
    assert "does not match its run log" in caplog.text


def test_shipped_maze_and_configs_exist():
    """The CS 370 maze and the default configuration the README points users to are in the repository."""
    assert (ROOT / "mazes" / "cs370.maze").is_file()
    assert (ROOT / "configs" / "default.toml").is_file()
