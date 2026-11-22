"""Tests for the breadth-first search and the route audit.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Prove the audit is right before trusting what it says about a trained network: a policy
    that follows the breadth-first search must score perfectly, and a bad one must be caught.

Decision:
    The test policies are plain functions, which is possible because the audit accepts any
    observation-to-action callable. No TensorFlow is needed.
"""

from __future__ import annotations

import numpy as np

from pirate_agent.audit import audit_policy, cell_policy, completion_check, play_greedy, route_grid, route_map
from pirate_agent.maze import MOVES, PIRATE_MARK, Action, MazeEnv, shortest_moves


def bfs_policy(maze):
    """A perfect policy: always step to a neighbor one move closer to the treasure."""
    distance = shortest_moves(maze.grid, maze.target)
    cols = maze.shape[1]

    def policy(observation):
        """Finds the pirate from PIRATE_MARK and returns the first move to a cell one step nearer.

        O(1) per move after the one search above: at most four lookups in `distance`.
        """
        index = int(np.argmax(observation[0] == PIRATE_MARK))
        row, col = divmod(index, cols)
        for action, (d_row, d_col) in MOVES.items():
            nxt = (row + d_row, col + d_col)
            if distance.get(nxt, 10**9) == distance[(row, col)] - 1:
                return int(action)
        raise AssertionError("no closer neighbor")

    return policy


def always(action):
    """A policy that ignores the observation and plays `action` from every cell."""
    return lambda observation: int(action)


def test_distances_on_a_small_maze(corridor):
    """Fewest moves from each free cell to the treasure, checked by hand; the wall gets no entry."""
    assert shortest_moves(corridor.grid, corridor.target) == {
        (1, 2): 0,
        (0, 2): 1,
        (0, 1): 2,
        (0, 0): 3,
        (1, 0): 4,
    }


def test_perfect_policy_scores_perfectly(cs370_maze):
    """A policy that follows the search wins all 50 CS 370 cells by a shortest route, longest 28 moves.

    The audit must give this known-perfect policy a perfect score before its verdict on a
    trained network means anything.
    """
    report = audit_policy(cs370_maze, bfs_policy(cs370_maze))
    assert (report.cells, report.wins, report.optimal) == (50, 50, 50)
    assert report.all_win
    assert "Longest route: 28 moves" in report.summary()
    assert completion_check(cs370_maze, bfs_policy(cs370_maze))


def test_bad_policy_is_caught(corridor):
    """A policy that only moves left never wins; the audit, the completion check and the grid all say so."""
    report = audit_policy(corridor, always(Action.LEFT))
    assert report.wins == 0
    assert not completion_check(corridor, always(Action.LEFT))
    assert "  x" in route_grid(report, corridor)


def test_route_grid_flags_a_longer_route(open3):
    """A win in 6 moves where 4 would do counts as not optimal and is starred on the route grid.

    This is how the evidence tells a policy that wins from one that wins by a shortest route.
    """
    # From (0, 0): right, down, left, down, then right twice along the bottom row.
    plan = {(0, 0): Action.RIGHT, (0, 1): Action.DOWN, (1, 1): Action.LEFT, (1, 0): Action.DOWN}

    def wander(observation):
        """Follows `plan` where it names a move; elsewhere heads right, then down the last column."""
        row, col = divmod(int(np.argmax(observation[0] == PIRATE_MARK)), 3)
        default = Action.RIGHT if row == 2 or col < 2 else Action.DOWN
        return int(plan.get((row, col), default))

    report = audit_policy(open3, wander)
    first = report.results[0]
    assert (first.cell, first.moves, first.fewest, first.optimal) == ((0, 0), 6, 4, False)
    assert route_grid(report, open3).splitlines()[0].startswith(" 6*")


def test_play_greedy_stops_at_the_step_limit(corridor):
    """A policy that loops between two cells is cut off at the step limit, so an audit always ends.

    The code review checked loop termination and found this cutoff in the CS 370 audit; the
    test keeps it in place.
    """

    def back_and_forth(observation):
        """Moves right from column 0 and left from any other, so the pirate never leaves the top row."""
        col = int(np.argmax(observation[0] == PIRATE_MARK)) % 3
        return int(Action.RIGHT if col == 0 else Action.LEFT)

    route, won = play_greedy(MazeEnv(corridor), back_and_forth, (0, 1), step_limit=5)
    assert not won
    assert route == [(0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 0)]


def test_route_map_marks_start_path_and_treasure(corridor):
    """The play command's map: 'S' at the start, 'o' along the route, 'T' on the treasure, '#' on walls."""
    env = MazeEnv(corridor)
    route, won = play_greedy(env, bfs_policy(corridor), (1, 0))
    assert won
    assert route_map(corridor, route) == "o o o\nS # T"


def test_cell_policy_matches_the_step_by_step_policy(cs370_maze):
    """The batched lookup table gives the same audit as the per-move policy from one call for all 50 cells."""
    policy = bfs_policy(cs370_maze)
    calls = []

    def batch(observations):
        """Records the size of each batch it receives, then answers every row with the per-move policy."""
        calls.append(len(observations))
        return np.array([policy(row.reshape(1, -1)) for row in observations])

    fast = audit_policy(cs370_maze, cell_policy(cs370_maze, batch))
    slow = audit_policy(cs370_maze, policy)
    assert fast == slow
    assert calls == [50]  # one batched call covers every starting cell
