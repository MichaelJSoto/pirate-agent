"""Training loop: epsilon-greedy games, replay learning, early stopping and progress logging.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)
Origin: qtrain(), Explorer, ReplayLearner, TargetSync and play_training_game() from the CS 370
notebook.

Intent:
    Train an agent until it wins from every free cell, and when it cannot, stop early and say
    why instead of spending the whole epoch budget.

Decision:
    qtrain() changed a global epsilon, so a second run silently began where the first one
    ended (code review finding S1). Here epsilon is a local value that every call to train()
    resets from the configuration. The declared but never-read `patience` (S7) is now real
    early stopping: once the win-rate window is full, a run that goes `patience` games without
    beating its best win rate stops as "stalled". Games played while the win rate already sits
    at the target do not count as stalled, because those games are waiting on the completion
    check, not failing to learn. The success test is unchanged from CS 370: a full window of
    wins and a passing completion check from every free cell. Progress goes to `logging`
    instead of print (S6). The trainer depends only on the Learner protocol below, so its
    stopping rules can be tested with a stand-in agent and no TensorFlow.
"""

from __future__ import annotations

import enum
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .audit import completion_check
from .config import TrainingConfig
from .maze import Cell, Maze, MazeEnv, Rewards, Status
from .replay import ReplayBuffer, Transition

log = logging.getLogger(__name__)


class Learner(Protocol):
    """What the trainer needs from an agent: a greedy move and one learning step."""

    def greedy_action(self, observation: np.ndarray) -> int: ...

    def learn(self, batch: Sequence[Transition]) -> float: ...


class StopReason(enum.StrEnum):
    SOLVED = "solved"  # a full window of wins and a passing completion check
    STALLED = "stalled"  # `patience` games without a better win rate
    BUDGET = "max_epochs"  # the epoch budget ran out first


@dataclass(frozen=True)
class EpochRecord:
    """One training game."""

    epoch: int
    loss: float
    moves: int
    won: bool
    win_rate: float
    epsilon: float


@dataclass(frozen=True)
class TrainingResult:
    """What a training run did and why it stopped."""

    stop_reason: StopReason
    epochs: int  # games played
    moves: int
    explored_moves: int
    seconds: float
    win_rate: float
    best_win_rate: float
    history: tuple[EpochRecord, ...]

    @property
    def explored_share(self) -> float:
        return self.explored_moves / self.moves if self.moves else 0.0

    def to_dict(self) -> dict[str, Any]:
        """The summary for a run log; the per-game history is written separately."""
        return {
            "stop_reason": str(self.stop_reason),
            "epochs": self.epochs,
            "moves": self.moves,
            "explored_moves": self.explored_moves,
            "explored_share": round(self.explored_share, 4),
            "seconds": round(self.seconds, 2),
            "win_rate": self.win_rate,
            "best_win_rate": self.best_win_rate,
        }


class Explorer:
    """Chooses each move epsilon-greedily and counts how often it explored. Kept from CS 370."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.explored = 0
        self.exploited = 0

    def choose(self, env: MazeEnv, agent: Learner, observation: np.ndarray, epsilon: float) -> int:
        """A random valid move with probability epsilon, else the agent's greedy move."""
        valid = env.valid_actions()
        if valid and self.rng.random() < epsilon:
            self.explored += 1
            return int(valid[int(self.rng.integers(len(valid)))])  # explore: any move onto a free cell
        self.exploited += 1
        return agent.greedy_action(observation)


class Trainer:
    """Trains one agent on one maze with one configuration."""

    def __init__(
        self,
        agent: Learner,
        maze: Maze,
        config: TrainingConfig,
        rewards: Rewards | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.agent = agent
        self.maze = maze
        self.config = config
        self.rewards = rewards or Rewards()
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)

    def train(self) -> TrainingResult:
        cfg = self.config
        env = MazeEnv(self.maze, self.rewards)
        buffer = ReplayBuffer(cfg.memory_for(self.maze.size), self.rng)
        window = cfg.window_for(self.maze.size)
        explorer = Explorer(self.rng)
        starts = self.maze.free_cells
        epsilon = cfg.epsilon_start  # every run starts from the configured value
        recent: deque[bool] = deque(maxlen=window)
        best = 0.0
        stale = 0
        moves = 0
        win_rate = 0.0
        history: list[EpochRecord] = []
        reason = StopReason.BUDGET
        started = time.perf_counter()
        log.info(
            "training on a %d x %d maze: %d starting cells, window %d, patience %d, up to %d games",
            *self.maze.shape,
            len(starts),
            window,
            cfg.patience,
            cfg.max_epochs,
        )
        for epoch in range(cfg.max_epochs):
            loss, game_moves, won = self._play_game(env, buffer, explorer, starts, epsilon)
            moves += game_moves
            recent.append(won)
            full = len(recent) == window
            win_rate = sum(recent) / window if full else 0.0  # 0 until the window fills, as in CS 370
            history.append(EpochRecord(epoch, loss, game_moves, won, win_rate, epsilon))
            if epoch % cfg.log_every == 0:
                log.info(
                    "game %5d | loss %.4f | moves %4d | win rate %.3f | epsilon %.3f | %.1f s",
                    epoch,
                    loss,
                    game_moves,
                    win_rate,
                    epsilon,
                    time.perf_counter() - started,
                )
            # The CS 370 exploration schedule: drop to a small fixed epsilon once the pirate
            # wins most games, otherwise decay a little after every game.
            if win_rate > cfg.settle_win_rate:
                epsilon = cfg.epsilon_settled
            else:
                epsilon = max(epsilon * cfg.epsilon_decay, cfg.epsilon_min)
            if win_rate >= cfg.target_win_rate and completion_check(
                self.maze, self.agent.greedy_action, self.rewards
            ):
                reason = StopReason.SOLVED
                break
            if full:
                if win_rate > best:
                    best, stale = win_rate, 0
                elif win_rate >= cfg.target_win_rate:
                    stale = 0  # at the target and waiting on the completion check: not a stall
                else:
                    stale += 1
                if stale >= cfg.patience:
                    reason = StopReason.STALLED
                    break
        seconds = time.perf_counter() - started
        log.info(
            "stopped (%s) after %d games and %d moves in %.1f s; best win rate %.3f",
            reason,
            len(history),
            moves,
            seconds,
            best,
        )
        return TrainingResult(
            stop_reason=reason,
            epochs=len(history),
            moves=moves,
            explored_moves=explorer.explored,
            seconds=seconds,
            win_rate=win_rate,
            best_win_rate=best,
            history=tuple(history),
        )

    def _play_game(
        self,
        env: MazeEnv,
        buffer: ReplayBuffer,
        explorer: Explorer,
        starts: list[Cell],
        epsilon: float,
    ) -> tuple[float, int, bool]:
        """Plays one game from a random free cell, learning after every move.

        Returns the game's mean loss, its number of moves, and whether the pirate won.
        """
        observation = env.reset(starts[int(self.rng.integers(len(starts)))])
        losses: list[float] = []
        status = Status.PLAYING
        while status is Status.PLAYING:
            action = explorer.choose(env, self.agent, observation, epsilon)
            next_observation, reward, status = env.step(action)
            # Only a win is terminal. Running out of reward budget is a cutoff, not a place in
            # the maze, so its target still looks one move ahead (a CS 370 design decision).
            buffer.add(Transition(observation, action, reward, next_observation, status is Status.WIN))
            losses.append(self.agent.learn(buffer.sample(self.config.batch_size)))
            observation = next_observation
        return float(np.mean(losses)), len(losses), status is Status.WIN
