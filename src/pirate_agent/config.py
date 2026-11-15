"""Training settings and the TOML configuration file that supplies them.

Programmer: Michael Soto
Date: November 15, 2026
Version: 2.1.1
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Gather every setting the CS 370 notebook kept in globals and magic numbers (epsilon, its
    decay and floor, the 0.9 and 0.999 thresholds, the never-used `patience`) into one typed,
    documented object, and read it from a file another developer can edit and review.

Decision:
    A configuration file is untrusted input, so it is validated as strictly as a maze file:
    unknown sections and keys are errors (a misspelled `patience` would otherwise be silently
    ignored, which is exactly the CS 370 bug), every value is type-checked (True is not
    accepted as the number 1), and every number must fall inside a stated range. The standard
    library's tomllib parses the file, so no extra dependency is needed.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CONFIG_BYTES = 64 * 1024


class ConfigError(ValueError):
    """A configuration file or setting that cannot be used."""


@dataclass(frozen=True)
class TrainingConfig:
    """Every training setting, with the CS 370 values as defaults unless a comment says otherwise."""

    seed: int = 1671  # the seed used throughout CS 370
    max_epochs: int = 15_000  # qtrain()'s default budget; each epoch is one game
    memory_size: int | None = None  # None: 8 moves per maze cell, as in the graded CS 370 run
    batch_size: int = 32
    target_sync_every: int = 50  # gradient steps between target network refreshes
    learning_rate: float = 0.001  # Adam's default, which CS 370 used implicitly
    discount: float = 0.95  # GameExperience's default
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    settle_win_rate: float = 0.9  # above this win rate, epsilon drops straight to epsilon_settled
    epsilon_settled: float = 0.05
    target_win_rate: float = 1.0  # CS 370 tested win_rate >= 0.999, which means 100%
    win_window: int | None = None  # None: half the maze cells, as in CS 370
    patience: int = 300  # new in 2.0: games without a better win rate before a stalled run stops
    target_shortest_share: float = 1.0  # new in 2.1: share of cells that must take a shortest route
    polish_patience: int = 200  # new in 2.1: polishing games without a new shortest route before stopping
    log_every: int = 25  # games between progress lines in the log

    def memory_for(self, cells: int) -> int:
        """Replay memory size for a maze of `cells` cells: the configured value, or 8 moves per cell."""
        return self.memory_size if self.memory_size is not None else 8 * cells

    def window_for(self, cells: int) -> int:
        """Win-rate window for a maze of `cells` cells: the configured value, or half the cells (CS 370)."""
        return self.win_window if self.win_window is not None else max(1, cells // 2)

    def to_dict(self) -> dict[str, Any]:
        """Every setting as a plain dictionary, recorded in each run log."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GenerateSettings:
    """Settings for a randomly generated maze."""

    rows: int = 8
    cols: int = 8
    wall_density: float = 0.25
    seed: int = 0


@dataclass(frozen=True)
class AppConfig:
    """A whole configuration file: where the maze comes from, how to train, where runs go."""

    training: TrainingConfig
    runs_dir: Path
    maze_file: Path | None = None
    generate: GenerateSettings | None = None
    source: str = "defaults"


# Each rule is (type, lowest allowed value, highest allowed value).
_INT = "int"
_FLOAT = "float"
_TRAINING_RULES: dict[str, tuple[str, float, float]] = {
    "seed": (_INT, 0, 2**32 - 1),
    "max_epochs": (_INT, 1, 1_000_000),
    "memory_size": (_INT, 1, 1_000_000),
    "batch_size": (_INT, 1, 4_096),
    "target_sync_every": (_INT, 1, 1_000_000),
    "learning_rate": (_FLOAT, 1e-9, 1.0),
    "discount": (_FLOAT, 0.0, 1.0),
    "epsilon_start": (_FLOAT, 0.0, 1.0),
    "epsilon_min": (_FLOAT, 0.0, 1.0),
    "epsilon_decay": (_FLOAT, 0.0, 1.0),
    "settle_win_rate": (_FLOAT, 0.0, 1.0),
    "epsilon_settled": (_FLOAT, 0.0, 1.0),
    "target_win_rate": (_FLOAT, 0.0, 1.0),
    "win_window": (_INT, 1, 100_000),
    "patience": (_INT, 1, 1_000_000),
    "target_shortest_share": (_FLOAT, 0.0, 1.0),
    "polish_patience": (_INT, 1, 1_000_000),
    "log_every": (_INT, 1, 1_000_000),
}
_GENERATE_RULES: dict[str, tuple[str, float, float]] = {
    "rows": (_INT, 2, 32),
    "cols": (_INT, 2, 32),
    "wall_density": (_FLOAT, 0.0, 0.6),
    "seed": (_INT, 0, 2**32 - 1),
}
_SECTIONS = {"maze": {"file", "generate"}, "training": set(_TRAINING_RULES), "output": {"runs_dir"}}


def _check_keys(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown setting(s) {', '.join(repr(k) for k in unknown)} in [{where}]; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def _check_value(name: str, value: Any, rule: tuple[str, float, float], where: str) -> Any:
    kind, low, high = rule
    if isinstance(value, bool):
        raise ConfigError(f"[{where}] {name} must be a number, not true or false")
    if kind == _INT and not isinstance(value, int):
        raise ConfigError(f"[{where}] {name} must be a whole number; got {value!r}")
    if kind == _FLOAT:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfigError(f"[{where}] {name} must be a finite number; got {value!r}")
        value = float(value)
    if not low <= value <= high:
        raise ConfigError(f"[{where}] {name} must be from {low:g} to {high:g}; got {value!r}")
    return value


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    table = data.get(name, {})
    if not isinstance(table, dict):
        raise ConfigError(f"[{name}] must be a table of settings")
    return table


def parse_config(text: str, base_dir: Path, source: str = "configuration") -> AppConfig:
    """Parses and validates TOML text. Relative paths are taken from `base_dir`."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{source} is not valid TOML: {error}") from None
    _check_keys(data, set(_SECTIONS), "top level")

    training_table = _section(data, "training")
    _check_keys(training_table, set(_TRAINING_RULES), "training")
    values = {
        name: _check_value(name, value, _TRAINING_RULES[name], "training")
        for name, value in training_table.items()
    }
    training = TrainingConfig(**values)
    if training.epsilon_min > training.epsilon_start:
        raise ConfigError("[training] epsilon_min cannot be larger than epsilon_start")

    maze_table = _section(data, "maze")
    _check_keys(maze_table, _SECTIONS["maze"], "maze")
    if ("file" in maze_table) == ("generate" in maze_table):
        raise ConfigError('[maze] needs exactly one of: file = "..." or a [maze.generate] table')
    maze_file = None
    generate = None
    if "file" in maze_table:
        if not isinstance(maze_table["file"], str) or not maze_table["file"].strip():
            raise ConfigError("[maze] file must be a path in quotes")
        maze_file = (base_dir / maze_table["file"]).resolve()
    else:
        gen_table = maze_table["generate"]
        if not isinstance(gen_table, dict):
            raise ConfigError("[maze.generate] must be a table of settings")
        _check_keys(gen_table, set(_GENERATE_RULES), "maze.generate")
        generate = GenerateSettings(
            **{
                name: _check_value(name, value, _GENERATE_RULES[name], "maze.generate")
                for name, value in gen_table.items()
            }
        )

    output_table = _section(data, "output")
    _check_keys(output_table, _SECTIONS["output"], "output")
    runs_dir_text = output_table.get("runs_dir", "runs")
    if not isinstance(runs_dir_text, str) or not runs_dir_text.strip():
        raise ConfigError("[output] runs_dir must be a path in quotes")
    return AppConfig(
        training=training,
        runs_dir=(base_dir / runs_dir_text).resolve(),
        maze_file=maze_file,
        generate=generate,
        source=source,
    )


def load_config(path: str | Path) -> AppConfig:
    """Reads and validates a .toml configuration file. The size is checked before reading."""
    path = Path(path)
    if path.suffix != ".toml":
        raise ConfigError(f"{path.name}: configuration files must end in '.toml'")
    if not path.is_file():
        raise ConfigError(f"{path.name}: no such configuration file")
    size = path.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ConfigError(
            f"{path.name} is {size:,} bytes; a configuration file may be at most {MAX_CONFIG_BYTES:,}"
        )
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise ConfigError(f"{path.name} is not UTF-8 text") from None
    return parse_config(text, path.parent, source=path.name)
