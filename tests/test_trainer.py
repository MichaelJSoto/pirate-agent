"""Tests for the training loop's stopping rules, exploration schedule and determinism.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Prove the three ways a run can end (solved, stalled, out of budget) happen exactly when
    they should, that epsilon restarts on every run (finding S1), and that a seeded run with
    the real network repeats exactly.

Decision:
    The stopping rules are tested with stand-in agents whose behavior is known in advance (a
    perfect breadth-first policy and one that only walks left), so the exact game a run stops
    on can be asserted without TensorFlow. Two small tests then train the real network.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pirate_agent.config import TrainingConfig
from pirate_agent.maze import MOVES, PIRATE_MARK, Action, shortest_moves
from pirate_agent.trainer import StopReason, Trainer


class PerfectAgent:
    """Always steps one move closer to the treasure; learns nothing."""

    def __init__(self, maze):
        """Runs one breadth-first search from the treasure, O(rows * cols); no batches counted yet."""
        self.distance = shortest_moves(maze.grid, maze.target)
        self.cols = maze.shape[1]
        self.batches = 0

    def greedy_action(self, observation):
        """Finds the pirate from PIRATE_MARK and returns the first move to a cell one step nearer. O(1)."""
        row, col = divmod(int(np.argmax(observation[0] == PIRATE_MARK)), self.cols)
        for action, (d_row, d_col) in MOVES.items():
            if self.distance.get((row + d_row, col + d_col), 10**9) == self.distance[(row, col)] - 1:
                return int(action)
        raise AssertionError("no closer neighbor")

    def greedy_actions(self, observations):
        """Answers the trainer's batched route audit one row at a time with greedy_action()."""
        return np.array([self.greedy_action(row.reshape(1, -1)) for row in observations])

    def learn(self, batch):
        """Counts the minibatch and reports zero loss, so a test can check one learning step per move."""
        self.batches += 1
        return 0.0

    def snapshot(self):
        """A placeholder: the policy never changes, so there are no weights to save."""
        return "weights"

    def restore(self, snapshot):
        """Keeps what it was given in `restored`; the policy itself stays the same."""
        self.restored = snapshot


class LeftAgent(PerfectAgent):
    """Always walks left, so it never reaches a bottom-right treasure."""

    def greedy_action(self, observation):
        """Returns LEFT whatever the observation; the inherited learn() still counts every step."""
        return int(Action.LEFT)


GREEDY = TrainingConfig(
    epsilon_start=0.0, epsilon_min=0.0, win_window=8, patience=20, polish_patience=6, log_every=1000
)


def test_perfect_agent_is_solved_when_the_window_fills(cs370_maze):
    """An agent that always wins is SOLVED on game 8, the first time the win window of 8 is full.

    It takes one learning step per move.
    """
    agent = PerfectAgent(cs370_maze)
    result = Trainer(agent, cs370_maze, GREEDY).train()
    assert result.stop_reason is StopReason.SOLVED
    assert result.epochs == 8  # games 0 to 7 fill the window of 8, all won
    assert result.win_rate == 1.0
    assert agent.batches == result.moves  # one learning step per move


def test_agent_that_never_improves_is_stopped_by_patience(open3):
    """An agent that never wins is stopped as STALLED once `patience` games pass with no better win rate.

    CS 370 declared `patience = 10` and never read it (finding S7); this is the early stopping
    that name promised.
    """
    result = Trainer(LeftAgent(open3), open3, GREEDY).train()
    assert result.stop_reason is StopReason.STALLED
    # The window fills on game 7 (best stays 0.0), then 20 more games without improvement.
    assert result.epochs == 8 + 19
    assert result.best_win_rate == 0.0


def test_budget_ends_a_run_before_patience(open3):
    """With max_epochs = 5, the budget ends the run as BUDGET before the win window can fill."""
    config = dataclasses.replace(GREEDY, max_epochs=5)
    result = Trainer(LeftAgent(open3), open3, config).train()
    assert result.stop_reason is StopReason.BUDGET
    assert result.epochs == 5


def test_epsilon_restarts_on_every_run(open3):
    """Two train() calls on one trainer both start at epsilon 1.0 and decay the same way (finding S1).

    CS 370 kept epsilon in a global, so a second run began where the first had left it.
    """
    config = TrainingConfig(win_window=4, patience=3, max_epochs=30, log_every=1000)
    trainer = Trainer(LeftAgent(open3), open3, config)
    first, second = trainer.train(), trainer.train()
    assert first.history[0].epsilon == second.history[0].epsilon == 1.0
    assert first.history[1].epsilon == pytest.approx(0.995)


def test_exploration_is_counted(open3):
    """The number of random moves is counted, and its share of all moves lies between 0 and 1."""
    config = TrainingConfig(win_window=4, patience=3, max_epochs=10, log_every=1000)
    result = Trainer(LeftAgent(open3), open3, config).train()
    assert 0 < result.explored_moves <= result.moves
    assert 0.0 < result.explored_share <= 1.0


@pytest.mark.tf
def test_real_network_learns_a_small_maze(corridor):
    """The real network and the real loop solve the corridor maze within 400 games, end to end."""
    from pirate_agent.agent import DQNAgent, configure_determinism

    config = TrainingConfig(seed=3, max_epochs=400, patience=200, log_every=1000)
    configure_determinism(config.seed)
    agent = DQNAgent.create(corridor.size, config)
    result = Trainer(agent, corridor, config).train()
    assert result.stop_reason is StopReason.SOLVED


@pytest.mark.tf
def test_seeded_runs_repeat_exactly(corridor):
    """Two seeded runs of the real network match game for game, so any result can be rerun from its seed."""
    from pirate_agent.agent import DQNAgent, configure_determinism

    config = TrainingConfig(seed=11, max_epochs=40, patience=200, log_every=1000)
    histories = []
    for _ in range(2):
        configure_determinism(config.seed)
        agent = DQNAgent.create(corridor.size, config)
        histories.append(Trainer(agent, corridor, config).train().history)
    assert histories[0] == histories[1]


# ---- route polishing (version 2.1) ----------------------------------------------------------------


def wander_plan(detour_at_top_right: bool):
    """Policies for the open 3 x 3 maze that win from every cell but take some long routes.

    Without the detour, 5 of the 8 starting cells take a shortest route; with it, only 4.
    """
    plan = {(0, 0): Action.RIGHT, (0, 1): Action.DOWN, (1, 1): Action.LEFT, (1, 0): Action.DOWN}
    if detour_at_top_right:
        plan[(0, 2)] = Action.LEFT

    def policy(observation):
        """Follows `plan` where it names a move; elsewhere heads right, then down the last column."""
        row, col = divmod(int(np.argmax(observation[0] == PIRATE_MARK)), 3)
        return int(plan.get((row, col), Action.RIGHT if row == 2 or col < 2 else Action.DOWN))

    return policy


class ScriptedAgent:
    """Each route audit uses the next policy in `script`; restore() freezes the policy it names."""

    def __init__(self, script):
        """Starts on the first policy in `script`, with no audits counted and nothing frozen."""
        self.script, self.audits, self.last, self.frozen = script, 0, 0, None

    def _index(self):
        """Index of the policy in use: the frozen one, else one per audit so far, holding at the last."""
        return self.frozen if self.frozen is not None else min(self.audits, len(self.script) - 1)

    def greedy_action(self, observation):
        """Plays the move of the policy in use."""
        return self.script[self._index()](observation)

    def greedy_actions(self, observations):
        """Answers a route audit with the policy in use, notes it for snapshot(), and moves to the next."""
        self.last = self._index()
        self.audits += 1
        return np.array([self.script[self.last](row.reshape(1, -1)) for row in observations])

    def learn(self, batch):
        """Learns nothing; only the route audits change which policy plays."""
        return 0.0

    def snapshot(self):
        """The index of the policy the last audit used, standing in for a copy of the weights."""
        return self.last

    def restore(self, snapshot):
        """Freezes the policy `snapshot` names, the way restoring saved weights would."""
        self.frozen = snapshot


def test_polishing_stops_once_every_route_is_shortest(open3):
    """Once every cell wins (game 8, 5 of 8 routes shortest), polishing runs until all 8 are shortest.

    It then stops as SOLVED two games later. This is the 2.1.0 route polish stage.
    """
    perfect = PerfectAgent(open3).greedy_action
    agent = ScriptedAgent([wander_plan(False), wander_plan(False), perfect])
    result = Trainer(agent, open3, GREEDY).train()
    assert result.stop_reason is StopReason.SOLVED
    assert (result.solved_at, result.first_shortest_routes) == (8, 5)  # the window fills at game 8
    assert (result.shortest_routes, result.routes_complete) == (8, True)
    assert result.epochs == 10  # two more games of polishing


def test_polishing_keeps_the_best_version_when_it_stops_improving(open3):
    """If polishing finds no better version for `polish_patience` games, the best one is restored.

    The run restores the version with 5 shortest routes, not the last one with 4.
    """
    agent = ScriptedAgent([wander_plan(False), wander_plan(True)])  # 5 shortest, then only 4
    result = Trainer(agent, open3, GREEDY).train()
    assert result.stop_reason is StopReason.SOLVED
    assert result.epochs == 8 + GREEDY.polish_patience
    assert (result.shortest_routes, result.routes_complete) == (5, False)
    assert agent.frozen == 0  # the 5-route version was restored, not the last one


def test_a_zero_target_skips_polishing(open3):
    """target_shortest_share = 0.0 ends the run as soon as every cell wins, the 2.0.0 stopping rule."""
    config = dataclasses.replace(GREEDY, target_shortest_share=0.0)
    agent = ScriptedAgent([wander_plan(False)])
    result = Trainer(agent, open3, config).train()
    assert (result.epochs, result.shortest_routes, result.routes_complete) == (8, 5, True)
