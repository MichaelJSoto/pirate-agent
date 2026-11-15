"""Route audit: plays a policy greedily from every free cell and compares it with the shortest path.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)
Origin: the PolicyAudit, route_grid and completion_check cells of the CS 370 notebook.

Intent:
    Answer two questions about a trained pirate with evidence instead of a win-rate number:
    does it reach the treasure from every legal starting cell, and how many of those routes
    are as short as possible?

Decision:
    A policy is any function from an observation to an action, so the same audit judges the
    neural network, a hand-written test policy, or a future agent without importing
    TensorFlow. The fewest possible moves come from one breadth-first search outward from the
    treasure, O(rows * cols), instead of a search from each starting cell.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .maze import STEP_LIMIT_FACTOR, Cell, Maze, MazeEnv, Rewards, Status, shortest_moves

Policy = Callable[[np.ndarray], int]


@dataclass(frozen=True)
class CellResult:
    """How the policy did from one starting cell."""

    cell: Cell
    moves: int | None  # moves taken to reach the treasure, or None if it never did
    fewest: int  # the fewest moves possible from this cell

    @property
    def won(self) -> bool:
        return self.moves is not None

    @property
    def optimal(self) -> bool:
        return self.moves is not None and self.moves == self.fewest


@dataclass(frozen=True)
class AuditReport:
    """The audit of one policy on one maze, cell by cell."""

    results: tuple[CellResult, ...]

    @property
    def cells(self) -> int:
        return len(self.results)

    @property
    def wins(self) -> int:
        return sum(result.won for result in self.results)

    @property
    def optimal(self) -> int:
        return sum(result.optimal for result in self.results)

    @property
    def all_win(self) -> bool:
        return self.wins == self.cells

    def summary(self) -> str:
        """Two lines: how many cells win, how many by a shortest route, and the longest route."""
        text = (
            f"{self.wins} of {self.cells} free cells reach the treasure, "
            f"{self.optimal} of them by a shortest route."
        )
        winners = [result for result in self.results if result.won]
        if winners:
            longest = max(winners, key=lambda result: result.moves or 0)
            text += (
                f"\nLongest route: {longest.moves} moves from {longest.cell}; "
                f"the fewest possible is {longest.fewest}."
            )
        return text

    def to_dict(self) -> dict[str, object]:
        """The totals for a run log."""
        return {"cells": self.cells, "wins": self.wins, "shortest_routes": self.optimal}


def play_greedy(
    env: MazeEnv, policy: Policy, start: Cell, step_limit: int | None = None
) -> tuple[list[Cell], bool]:
    """Plays one game from `start` with no exploration. Returns the cells visited and whether it won."""
    if step_limit is None:
        step_limit = env.maze.size * STEP_LIMIT_FACTOR
    observation = env.reset(start)
    route = [env.pirate]
    status = Status.PLAYING
    for _ in range(step_limit):
        observation, _, status = env.step(policy(observation))
        route.append(env.pirate)
        if status is not Status.PLAYING:
            break
    return route, status is Status.WIN


def audit_policy(maze: Maze, policy: Policy, rewards: Rewards | None = None) -> AuditReport:
    """Plays `policy` from every free cell and records its route length next to the shortest."""
    env = MazeEnv(maze, rewards)
    fewest = shortest_moves(maze.grid, maze.target)
    results = []
    for cell in maze.free_cells:
        route, won = play_greedy(env, policy, cell)
        results.append(CellResult(cell=cell, moves=len(route) - 1 if won else None, fewest=fewest[cell]))
    return AuditReport(tuple(results))


def completion_check(maze: Maze, policy: Policy, rewards: Rewards | None = None) -> bool:
    """True if the policy wins from every free cell. Stops at the first failure."""
    env = MazeEnv(maze, rewards)
    return all(play_greedy(env, policy, cell)[1] for cell in maze.free_cells)


def route_grid(report: AuditReport, maze: Maze) -> str:
    """The maze as text: moves taken from each free cell, '*' where a shorter route exists.

    '#' is a wall, 'T' the treasure and 'x' a cell the policy never solved.
    """
    grid = [["  #" if value == 0 else "   " for value in line] for line in maze.grid]
    grid[maze.target[0]][maze.target[1]] = "  T"
    for result in report.results:
        row, col = result.cell
        if result.moves is None:
            grid[row][col] = "  x"
        else:
            grid[row][col] = f"{result.moves:>2}" + ("*" if result.moves > result.fewest else " ")
    return "\n".join(" ".join(line) for line in grid)


def route_map(maze: Maze, route: list[Cell]) -> str:
    """The maze with one route drawn on it: 'S' start, 'o' path, 'T' treasure, '#' wall."""
    grid = [["#" if value == 0 else "." for value in line] for line in maze.grid]
    for cell in route:
        grid[cell[0]][cell[1]] = "o"
    grid[route[0][0]][route[0][1]] = "S"
    grid[maze.target[0]][maze.target[1]] = "T"
    return "\n".join(" ".join(line) for line in grid)
