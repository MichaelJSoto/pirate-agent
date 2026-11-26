"""Tests for the TOML configuration loader.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
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
    """Parses `text` as if it were a file in configs/, so relative paths resolve as the shipped files' do."""
    return parse_config(text, ROOT / "configs")


def test_defaults_match_cs370():
    """With no [training] table, every setting keeps its CS 370 value.

    That includes the replay memory (8 moves per cell, 512) and the win window (half the
    cells, 32). The magic numbers became named settings (finding S2) without changing what a
    default run does.
    """
    config = parse(BASE)
    assert config.training == TrainingConfig()
    assert config.training.memory_for(64) == 512  # 8 * maze.size in the graded CS 370 run
    assert config.training.window_for(64) == 32  # maze.size // 2
    assert config.maze_file == (ROOT / "mazes" / "cs370.maze").resolve()
    assert config.runs_dir == (ROOT / "configs" / "runs").resolve()


def test_generate_section():
    """A [maze.generate] table stands in for a maze file, and each of its settings is carried through."""
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
    """Each bad setting is refused with a message that names it.

    The cases: a misspelled key, which would otherwise do nothing, as the CS 370 `patience`
    did nothing (finding S7); a boolean; a fraction for a whole number; values out of range or
    not finite; an epsilon floor above its start; an unknown section; and a path that is a number.
    """
    with pytest.raises(ConfigError, match=message):
        parse(BASE + extra)


def test_maze_needs_exactly_one_source():
    """A configuration must name a maze file or a generator, not neither and not both."""
    with pytest.raises(ConfigError, match="exactly one"):
        parse("[maze]\n")
    with pytest.raises(ConfigError, match="exactly one"):
        parse(BASE + "[maze.generate]\nrows = 4\n")


def test_invalid_toml():
    """Text that is not TOML is refused as a ConfigError with the parser's reason, not a TOMLDecodeError."""
    with pytest.raises(ConfigError, match="not valid TOML"):
        parse("[maze\n")


def test_file_checks(tmp_path):
    """load_config() refuses a wrong suffix, a missing file, and a file over 65,536 bytes."""
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
    """Every file in configs/ loads and names exactly one maze source, so a typo there fails CI first."""
    config = load_config(path)
    assert (config.maze_file is None) != (config.generate is None)
