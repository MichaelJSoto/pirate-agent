"""Shared test fixtures: the CS 370 maze and two small mazes that are easy to reason about.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Give every test the same known mazes, so expected numbers (50 free cells, a farthest cell
    28 moves away) can be checked against the CS 370 notebook's own printed output.

Decision:
    The CS 370 maze is read from mazes/cs370.maze rather than retyped here, so the file the
    command line uses is the same one the tests check.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pirate_agent.maze import Maze, load_maze

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cs370_maze() -> Maze:
    """The 8 x 8 CS 370 maze: treasure at (7, 7), 50 free cells, the farthest 28 moves away."""
    return load_maze(ROOT / "mazes" / "cs370.maze")


@pytest.fixture
def corridor() -> Maze:
    """A 2 x 3 maze: the top row is open, the bottom row has a wall in the middle.

    . . .
    . # T
    """
    return Maze.from_grid([[1, 1, 1], [1, 0, 1]])


@pytest.fixture
def open3() -> Maze:
    """A 3 x 3 maze with no walls and the treasure bottom-right."""
    return Maze.from_grid([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
