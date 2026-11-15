"""Tests for the training loop's stopping rules, exploration schedule and determinism.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
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
        self.distance = shortest_moves(maze.grid, maze.target)
        self.cols = maze.shape[1]
        self.batches = 0

    def greedy_action(self, observation):
        row, col = divmod(int(np.argmax(observation[0] == PIRATE_MARK)), self.cols)
        for action, (d_row, d_col) in MOVES.items():
            if self.distance.get((row + d_row, col + d_col), 10**9) == self.distance[(row, col)] - 1:
                return int(action)
        raise AssertionError("no closer neighbor")

    def learn(self, batch):
        self.batches += 1
        return 0.0


class LeftAgent(PerfectAgent):
    """Always walks left, so it never reaches a bottom-right treasure."""

    def greedy_action(self, observation):
        return int(Action.LEFT)


GREEDY = TrainingConfig(epsilon_start=0.0, epsilon_min=0.0, win_window=8, patience=20, log_every=1000)


def test_perfect_agent_is_solved_when_the_window_fills(cs370_maze):
    agent = PerfectAgent(cs370_maze)
    result = Trainer(agent, cs370_maze, GREEDY).train()
    assert result.stop_reason is StopReason.SOLVED
    assert result.epochs == 8  # games 0 to 7 fill the window of 8, all won
    assert result.win_rate == 1.0
    assert agent.batches == result.moves  # one learning step per move


def test_agent_that_never_improves_is_stopped_by_patience(open3):
    result = Trainer(LeftAgent(open3), open3, GREEDY).train()
    assert result.stop_reason is StopReason.STALLED
    # The window fills on game 7 (best stays 0.0), then 20 more games without improvement.
    assert result.epochs == 8 + 19
    assert result.best_win_rate == 0.0


def test_budget_ends_a_run_before_patience(open3):
    config = dataclasses.replace(GREEDY, max_epochs=5)
    result = Trainer(LeftAgent(open3), open3, config).train()
    assert result.stop_reason is StopReason.BUDGET
    assert result.epochs == 5


def test_epsilon_restarts_on_every_run(open3):
    config = TrainingConfig(win_window=4, patience=3, max_epochs=30, log_every=1000)
    trainer = Trainer(LeftAgent(open3), open3, config)
    first, second = trainer.train(), trainer.train()
    assert first.history[0].epsilon == second.history[0].epsilon == 1.0
    assert first.history[1].epsilon == pytest.approx(0.995)


def test_exploration_is_counted(open3):
    config = TrainingConfig(win_window=4, patience=3, max_epochs=10, log_every=1000)
    result = Trainer(LeftAgent(open3), open3, config).train()
    assert 0 < result.explored_moves <= result.moves
    assert 0.0 < result.explored_share <= 1.0


@pytest.mark.tf
def test_real_network_learns_a_small_maze(corridor):
    from pirate_agent.agent import DQNAgent, configure_determinism

    config = TrainingConfig(seed=3, max_epochs=400, patience=200, log_every=1000)
    configure_determinism(config.seed)
    agent = DQNAgent.create(corridor.size, config)
    result = Trainer(agent, corridor, config).train()
    assert result.stop_reason is StopReason.SOLVED


@pytest.mark.tf
def test_seeded_runs_repeat_exactly(corridor):
    from pirate_agent.agent import DQNAgent, configure_determinism

    config = TrainingConfig(seed=11, max_epochs=40, patience=200, log_every=1000)
    histories = []
    for _ in range(2):
        configure_determinism(config.seed)
        agent = DQNAgent.create(corridor.size, config)
        histories.append(Trainer(agent, corridor, config).train().history)
    assert histories[0] == histories[1]
