"""Maze layouts, maze files, the solvable-maze generator, and the game environment.

Programmer: Michael Soto
Date: November 15, 2026
Version: 2.1.1
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)
Origin: rewrites TreasureMaze.py, a starter file SNHU supplied for CS 370 Project Two.

Intent:
    Hold everything the agent needs to know about a maze: its layout, the rules for moving,
    the reward for every move, and the checks that decide whether a layout is safe to use.

Decision:
    A maze is validated once, when it is created, and cannot change afterward. Every way in
    (a grid written in code, a .maze file, or the random generator) goes through the same
    Maze.from_grid() checks, so the environment never has to defend itself against a bad
    layout in the middle of training. The reward values are the starter file's, but two of
    its bugs are fixed and covered by regression tests: an invalid move now pays its -0.75
    penalty (the starter set `mode` where it meant `nmode`), and a blocked treasure now gets
    its own clear error instead of a ValueError from list.remove().
"""

from __future__ import annotations

import enum
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Cell codes in a validated grid. The starter file used the floats 1.0 and 0.0 and compared
# them with ==; whole numbers make that comparison exact (code review finding S8).
FREE = 1
WALL = 0

# The value that marks the pirate's cell in an observation, unchanged from the starter file
# so a network trained on the old observations reads the new ones the same way.
PIRATE_MARK = 0.5

# Limits on any maze the program accepts. 32 by 32 keeps a network input under 1,100 values.
MIN_SIDE = 2
MAX_SIDE = 32

# A .maze file for the largest legal maze is about 1 KiB, so anything over 8 KiB is refused
# before it is read into memory.
MAX_FILE_BYTES = 8 * 1024

# A greedy game ends after maze.size * 4 moves, the safety cutoff from the starter notebook.
STEP_LIMIT_FACTOR = 4

# Characters allowed in a .maze file.
FILE_WALL = "#"
FILE_FREE = "."
FILE_TREASURE = "T"
FILE_COMMENT = ";"
MAZE_SUFFIX = ".maze"

Cell = tuple[int, int]


class MazeError(ValueError):
    """A maze layout, maze file or generator setting that cannot be used."""


class Action(enum.IntEnum):
    """The four moves, numbered as in the starter file so saved networks keep their meaning."""

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


# Each action as a (row, column) step.
MOVES: dict[Action, Cell] = {
    Action.LEFT: (0, -1),
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
}
NUM_ACTIONS = len(Action)


class Status(enum.StrEnum):
    """Where a game stands after a move. The values match the starter file's strings."""

    PLAYING = "not_over"
    WIN = "win"
    LOSE = "lose"


class Mode(enum.StrEnum):
    """What the last move did to the pirate."""

    START = "start"
    VALID = "valid"
    INVALID = "invalid"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Rewards:
    """The reward for each kind of move, with the starter file's values as defaults.

    As the starter file explained, every move earns between -1 and 1. Reaching the treasure
    pays the most. Hitting a wall or the edge of the maze costs the most, and returning to a
    visited cell costs a smaller penalty so the pirate does not wander among free cells. Each
    ordinary step costs a little, which favors shorter routes.
    """

    win: float = 1.0
    step: float = -0.04
    revisit: float = -0.25
    invalid: float = -0.75
    # A game is lost once its total reward falls below budget_factor * (number of cells).
    budget_factor: float = -0.5
    # Added to that budget when the pirate cannot move at all.
    blocked_extra: float = -1.0


def shortest_moves(grid: np.ndarray, target: Cell) -> dict[Cell, int]:
    """Breadth-first search out from the target: the fewest moves from every cell that can reach it.

    Works on any 0/1 grid, validated or not, because the validator and the generator both need
    it before a Maze exists. Each cell is expanded once, so it runs in O(rows * cols).
    """
    rows, cols = grid.shape
    distance: dict[Cell, int] = {target: 0}
    frontier: deque[Cell] = deque([target])
    while frontier:
        row, col = frontier.popleft()  # nearest unexpanded cell first
        for d_row, d_col in MOVES.values():
            nxt = (row + d_row, col + d_col)
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and grid[nxt] == FREE and nxt not in distance:
                distance[nxt] = distance[(row, col)] + 1  # one move farther than its neighbor
                frontier.append(nxt)
    return distance


def _as_grid(rows: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    """Converts nested lists or an array to an int8 grid, rejecting anything but 0 and 1."""
    if isinstance(rows, np.ndarray):
        raw = rows
    else:
        lines = [list(line) for line in rows]
        if not lines:
            raise MazeError("the maze has no rows")
        widths = {len(line) for line in lines}
        if len(widths) != 1:
            raise MazeError(f"every row must be the same length; found lengths {sorted(widths)}")
        raw = np.array(lines, dtype=object)
    if raw.ndim != 2:
        raise MazeError(f"a maze must be two-dimensional, not {raw.ndim}-dimensional")
    grid = np.zeros(raw.shape, dtype=np.int8)
    for (row, col), value in np.ndenumerate(raw):
        # bool is a subclass of int, so True would slip through as 1 without this check.
        if isinstance(value, (bool, np.bool_)) or value not in (0, 1):
            raise MazeError(f"cell ({row}, {col}) is {value!r}; only 0 (wall) and 1 (free) are allowed")
        grid[row, col] = int(value)
    return grid


@dataclass(frozen=True, eq=False)
class Maze:
    """A validated, read-only maze: a grid of FREE and WALL cells and a treasure cell.

    Build one with Maze.from_grid(), load_maze(), parse_maze() or generate_maze(); all of them
    run the same checks, so holding a Maze means holding a layout the agent can learn.
    """

    grid: np.ndarray
    target: Cell

    @classmethod
    def from_grid(cls, rows: Iterable[Iterable[float]] | np.ndarray, target: Cell | None = None) -> Maze:
        """Validates a layout and returns it as a Maze, or raises MazeError saying why not.

        The treasure defaults to the bottom-right cell, where the starter file always put it.
        """
        grid = _as_grid(rows)
        n_rows, n_cols = grid.shape
        for name, side in (("rows", n_rows), ("columns", n_cols)):
            if not MIN_SIDE <= side <= MAX_SIDE:
                raise MazeError(f"a maze needs {MIN_SIDE} to {MAX_SIDE} {name}; this one has {side}")
        if target is None:
            target = (n_rows - 1, n_cols - 1)
        target = (int(target[0]), int(target[1]))
        if not (0 <= target[0] < n_rows and 0 <= target[1] < n_cols):
            raise MazeError(f"the treasure cell {target} is outside the {n_rows} x {n_cols} maze")
        # Checked first, before anything else relies on the treasure cell (code review finding
        # S10: the starter file's own check here could never run).
        if grid[target] != FREE:
            raise MazeError(f"the treasure cell {target} is a wall; the treasure must sit on a free cell")
        free = [cell for cell, value in np.ndenumerate(grid) if value == FREE and cell != target]
        if not free:
            raise MazeError("the maze has no free cell for the pirate to start on")
        reachable = shortest_moves(grid, target)
        stranded = [cell for cell in free if cell not in reachable]
        if stranded:
            examples = ", ".join(str(cell) for cell in stranded[:3])
            raise MazeError(
                f"{len(stranded)} free cell(s) cannot reach the treasure, for example {examples}; "
                "every free cell must have a path to the treasure"
            )
        # Read-only from here on, so no later code can open a wall and undo these checks.
        grid.setflags(write=False)
        return cls(grid=grid, target=target)

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns) as plain Python integers."""
        return (int(self.grid.shape[0]), int(self.grid.shape[1]))

    @property
    def size(self) -> int:
        """Number of cells, walls included. It is also the length of an observation."""
        return int(self.grid.size)

    @property
    def free_cells(self) -> list[Cell]:
        """Every cell the pirate may start on: the free cells other than the treasure, row by row."""
        return [
            (int(r), int(c))
            for (r, c), value in np.ndenumerate(self.grid)
            if value == FREE and (r, c) != self.target
        ]

    def is_free(self, cell: Cell) -> bool:
        """True if `cell` lies inside the maze and is not a wall."""
        row, col = cell
        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1] and self.grid[cell] == FREE

    def to_text(self, comments: Iterable[str] = ()) -> str:
        """The maze in .maze file form: '#' wall, '.' free, 'T' treasure, ';' comment lines."""
        lines = [f"{FILE_COMMENT} {text}" for text in comments]
        for row in range(self.grid.shape[0]):
            chars = []
            for col in range(self.grid.shape[1]):
                if (row, col) == self.target:
                    chars.append(FILE_TREASURE)
                else:
                    chars.append(FILE_FREE if self.grid[row, col] == FREE else FILE_WALL)
            lines.append("".join(chars))
        return "\n".join(lines) + "\n"


def parse_maze(data: bytes | str, source: str = "maze text") -> Maze:
    """Parses .maze text into a validated Maze, naming the line and column of any problem.

    Format: one line per row, '#' for a wall, '.' for a free cell, and at most one 'T' for the
    treasure (bottom-right if there is none). Blank lines and lines starting with ';' are
    ignored. Only ASCII is accepted, so look-alike Unicode characters cannot sneak in.
    """
    if isinstance(data, bytes):
        if len(data) > MAX_FILE_BYTES:
            raise MazeError(f"{source} is {len(data):,} bytes; a maze file may be at most {MAX_FILE_BYTES:,}")
        data = data.removeprefix(b"\xef\xbb\xbf")  # a UTF-8 byte order mark some editors add
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError as error:
            raise MazeError(
                f"{source} contains a non-ASCII byte at position {error.start}; "
                "use only '#', '.', 'T' and ';' comments"
            ) from None
    else:
        text = data
    rows: list[list[int]] = []
    target: Cell | None = None
    width: int | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith(FILE_COMMENT):
            continue
        row: list[int] = []
        for col, char in enumerate(stripped):
            if char == FILE_WALL:
                row.append(WALL)
            elif char == FILE_FREE:
                row.append(FREE)
            elif char == FILE_TREASURE:
                if target is not None:
                    raise MazeError(f"{source} line {line_number}: a maze may have only one treasure 'T'")
                target = (len(rows), col)
                row.append(FREE)
            else:
                raise MazeError(
                    f"{source} line {line_number}, column {col + 1}: unexpected character {char!r}; "
                    "use '#' for walls, '.' for free cells and 'T' for the treasure"
                )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise MazeError(
                f"{source} line {line_number}: the row has {len(row)} cells but the first row has {width}"
            )
        rows.append(row)
    if not rows:
        raise MazeError(f"{source} has no maze rows")
    return Maze.from_grid(rows, target)


def load_maze(path: str | Path) -> Maze:
    """Reads and validates a .maze file. The size is checked before the file is read."""
    path = Path(path)
    if path.suffix != MAZE_SUFFIX:
        raise MazeError(f"{path.name}: maze files must end in '{MAZE_SUFFIX}'")
    if not path.is_file():
        raise MazeError(f"{path.name}: no such maze file")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise MazeError(f"{path.name} is {size:,} bytes; a maze file may be at most {MAX_FILE_BYTES:,}")
    return parse_maze(path.read_bytes(), source=path.name)


def generate_maze(
    rows: int,
    cols: int,
    wall_density: float = 0.25,
    seed: int = 0,
    min_free_share: float = 0.5,
    max_attempts: int = 100,
) -> Maze:
    """Generates a random maze in which every free cell can reach the bottom-right treasure.

    Each cell becomes a wall with probability `wall_density`. A breadth-first search from the
    treasure then finds every cell that can reach it, and any free cell cut off from it is
    filled in as a wall, so the result always passes validation. A layout is redrawn (from
    the same seeded generator, so results repeat) when fewer than `min_free_share` of its
    cells remain free. Each attempt costs O(rows * cols).
    """
    for name, side in (("rows", rows), ("cols", cols)):
        if isinstance(side, bool) or not isinstance(side, int) or not MIN_SIDE <= side <= MAX_SIDE:
            raise MazeError(f"{name} must be a whole number from {MIN_SIDE} to {MAX_SIDE}; got {side!r}")
    if not 0.0 <= wall_density <= 0.6:
        raise MazeError(f"wall_density must be from 0.0 to 0.6; got {wall_density!r}")
    if not 0.0 < min_free_share <= 1.0:
        raise MazeError(f"min_free_share must be above 0.0 and at most 1.0; got {min_free_share!r}")
    rng = np.random.default_rng(seed)
    target = (rows - 1, cols - 1)
    needed = max(2, int(np.ceil(min_free_share * rows * cols)))  # the treasure plus one start cell
    for _ in range(max_attempts):
        # A cell is free unless its random draw falls below wall_density; the treasure is always free.
        raw = (rng.random((rows, cols)) >= wall_density).astype(np.int8)
        raw[target] = FREE
        # The validator's own search, run outward from the treasure, finds every cell with a path to it.
        reachable = shortest_moves(raw, target)
        if len(reachable) < needed:
            continue  # too many walls survived; draw a new layout
        grid = np.full((rows, cols), WALL, dtype=np.int8)
        for cell in reachable:
            grid[cell] = FREE  # only cells with a path to the treasure stay open
        return Maze.from_grid(grid, target)
    raise MazeError(
        f"no {rows} x {cols} maze with wall density {wall_density} kept {min_free_share:.0%} of its "
        f"cells open in {max_attempts} attempts; lower the wall density"
    )


class MazeEnv:
    """One game at a time on a validated maze. Replaces the starter TreasureMaze class.

    The pirate starts on a free cell, each step() applies one move and returns the new
    observation, the reward and the game status, and observe() returns the maze as one row of
    numbers: 1.0 free, 0.0 wall, 0.5 the pirate.
    """

    def __init__(self, maze: Maze, rewards: Rewards | None = None) -> None:
        self.maze = maze
        self.rewards = rewards or Rewards()
        # The starter file's minimum reward: a game ends as lost once its total falls below this,
        # so a pirate that wanders cannot keep a game running forever.
        self.min_reward = self.rewards.budget_factor * maze.size
        self._background = maze.grid.astype(np.float32)  # walls 0.0, free cells 1.0
        self._starts = set(maze.free_cells)
        self.reset(maze.free_cells[0])

    def reset(self, start: Cell) -> np.ndarray:
        """Puts the pirate on `start` for a new game and returns the first observation."""
        start = (int(start[0]), int(start[1]))
        if start not in self._starts:
            raise MazeError(
                f"the pirate cannot start on {start}: it is a wall, the treasure, or outside the maze"
            )
        self.pirate: Cell = start
        self.mode = Mode.START
        self.total_reward = 0.0
        self.visited: set[Cell] = set()
        return self.observe()

    def valid_actions(self, cell: Cell | None = None) -> list[Action]:
        """The moves from `cell` (default: the pirate's cell) that land on a free cell."""
        row, col = self.pirate if cell is None else cell
        return [
            action
            for action, (d_row, d_col) in MOVES.items()
            if self.maze.is_free((row + d_row, col + d_col))
        ]

    def step(self, action: int) -> tuple[np.ndarray, float, Status]:
        """Applies one move. Returns (observation, reward, status)."""
        action = Action(int(action))  # raises ValueError for anything but 0 to 3
        self.visited.add(self.pirate)  # the cell being left counts as visited
        valid = self.valid_actions()
        if not valid:
            self.mode = Mode.BLOCKED
        elif action in valid:
            d_row, d_col = MOVES[action]
            self.pirate = (self.pirate[0] + d_row, self.pirate[1] + d_col)
            self.mode = Mode.VALID
        else:
            # The pirate stays put. The starter file assigned 'invalid' to the wrong variable
            # here, so this mode was never recorded (code review finding S9).
            self.mode = Mode.INVALID
        reward = self._reward()
        self.total_reward += reward
        return self.observe(), reward, self.status()

    def _reward(self) -> float:
        """The reward for the move just made."""
        if self.pirate == self.maze.target:
            return self.rewards.win
        if self.mode is Mode.BLOCKED:
            return self.min_reward + self.rewards.blocked_extra
        # Checked before the revisit rule: an invalid move leaves the pirate on a cell it has
        # already visited, so in the starter's order the -0.75 penalty could never be paid.
        if self.mode is Mode.INVALID:
            return self.rewards.invalid
        if self.pirate in self.visited:
            return self.rewards.revisit
        return self.rewards.step

    def status(self) -> Status:
        """LOSE once the reward budget is spent, WIN at the treasure, PLAYING otherwise."""
        if self.total_reward < self.min_reward:
            return Status.LOSE
        if self.pirate == self.maze.target:
            return Status.WIN
        return Status.PLAYING

    def observe(self) -> np.ndarray:
        """The maze as a (1, cells) float32 row with the pirate's cell set to PIRATE_MARK."""
        canvas = self._background.copy()
        canvas[self.pirate] = PIRATE_MARK
        return canvas.reshape(1, -1)
