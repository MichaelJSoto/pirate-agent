"""Tests for the TOML configuration loader.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Show that a configuration file is validated as strictly as a maze file: misspelled keys,
    wrong types and out-of-range values are all refused with a message that names the setting.

Decision:
    The shipped configs/ files are loaded in a test too, so a typo in them fails CI instead of
    failing the first person who runs the command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pirate_agent.config import MAX_CONFIG_BYTES, ConfigError, TrainingConfig, load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]
BASE = """
[maze]
file = "../mazes/cs370.maze"
"""


def parse(text: str):
    return parse_config(text, ROOT / "configs")


def test_defaults_match_cs370():
    config = parse(BASE)
    assert config.training == TrainingConfig()
    assert config.training.memory_for(64) == 512  # 8 * maze.size in the graded CS 370 run
    assert config.training.window_for(64) == 32  # maze.size // 2
    assert config.maze_file == (ROOT / "mazes" / "cs370.maze").resolve()
    assert config.runs_dir == (ROOT / "configs" / "runs").resolve()


def test_generate_section():
    config = parse("[maze.generate]\nrows = 6\ncols = 7\nwall_density = 0.2\nseed = 3\n")
    assert config.maze_file is None
    assert (config.generate.rows, config.generate.cols, config.generate.seed) == (6, 7, 3)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("[training]\npatiense = 10\n", "unknown setting\\(s\\) 'patiense' in \\[training\\]"),
        ("[training]\npatience = true\n", "patience must be a number, not true or false"),
        ("[training]\npatience = 2.5\n", "patience must be a whole number"),
        ("[training]\nepsilon_decay = 1.5\n", "epsilon_decay must be from 0 to 1"),
        ("[training]\nlearning_rate = nan\n", "learning_rate must be a finite number"),
        ("[training]\nepsilon_start = 0.01\nepsilon_min = 0.5\n", "epsilon_min cannot be larger"),
        ("[surprise]\nx = 1\n", "unknown setting\\(s\\) 'surprise' in \\[top level\\]"),
        ("[output]\nruns_dir = 5\n", "runs_dir must be a path"),
    ],
)
def test_bad_settings_are_named(extra, message):
    with pytest.raises(ConfigError, match=message):
        parse(BASE + extra)


def test_maze_needs_exactly_one_source():
    with pytest.raises(ConfigError, match="exactly one"):
        parse("[maze]\n")
    with pytest.raises(ConfigError, match="exactly one"):
        parse(BASE + "[maze.generate]\nrows = 4\n")


def test_invalid_toml():
    with pytest.raises(ConfigError, match="not valid TOML"):
        parse("[maze\n")


def test_file_checks(tmp_path):
    with pytest.raises(ConfigError, match="must end in '.toml'"):
        load_config(tmp_path / "config.ini")
    with pytest.raises(ConfigError, match="no such configuration file"):
        load_config(tmp_path / "missing.toml")
    big = tmp_path / "big.toml"
    big.write_text("#" * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ConfigError, match="at most 65,536"):
        load_config(big)


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.toml")), ids=lambda p: p.name)
def test_shipped_configs_load(path):
    config = load_config(path)
    assert (config.maze_file is None) != (config.generate is None)
