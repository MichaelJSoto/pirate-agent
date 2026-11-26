"""Training loop: epsilon-greedy games, replay learning, early stopping, route polishing and logging.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)
Origin: qtrain(), Explorer, ReplayLearner, TargetSync and play_training_game() from the CS 370
notebook.

Intent:
    Train an agent until it wins from every free cell by a shortest route, and when it cannot,
    stop early, keep the best version it found, and say why instead of spending the whole
    epoch budget.

Decision:
    qtrain() changed a global epsilon, so a second run silently began where the first one
    ended (code review finding S1). Here epsilon is a local value that every call to train()
    resets from the configuration. The declared but never-read `patience` (S7) is now real
    early stopping: once the win-rate window is full, a run that goes `patience` games without
    beating its best win rate stops as "stalled". Games played while the win rate already sits
    at the target do not count as stalled, because those games are waiting on the route check,
    not failing to learn. Progress goes to `logging` instead of print (S6).

    Version 2.1 adds a polish stage. CS 370 stopped at the first full window of wins with a
    passing completion check, which says nothing about route length, and the 2.0 evidence
    showed the share of shortest routes swinging from 33 to 50 of 50 between seeds. Now, once
    every cell wins, training continues, the route audit runs after every game, and the agent
    is snapshotted whenever it reaches more cells by a shortest route. The run ends when
    `target_shortest_share` of the cells take a shortest route, or after `polish_patience`
    games without improvement, and in that case the best snapshot is restored, so polishing can
    never leave the agent worse than when it first solved the maze. The audit runs as one
    batched prediction for all cells (see audit.cell_policy), which keeps a check per game
    cheap. The trainer depends only on the Learner protocol below, so every stopping rule can
    be tested with a stand-in agent and no TensorFlow.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .audit import AuditReport, audit_policy, cell_policy
from .config import TrainingConfig
from .maze import Cell, Maze, MazeEnv, Rewards, Status
from .replay import ReplayBuffer, Transition

log = logging.getLogger(__name__)


class Learner(Protocol):
    """What the trainer needs from an agent: greedy moves, a learning step, and snapshots."""

    def greedy_action(self, observation: np.ndarray) -> int:
        """The move with the highest Q-value for one observation."""
        ...

    def greedy_actions(self, observations: np.ndarray) -> np.ndarray:
        """The greedy move for every row of a batch, used by the route audit."""
        ...

    def learn(self, batch: Sequence[Transition]) -> float:
        """One learning step on a minibatch of stored moves; returns the loss."""
        ...

    def snapshot(self) -> Any:
        """A copy of whatever restore() needs to bring back the current version."""
        ...

    def restore(self, snapshot: Any) -> None:
        """Puts a version saved by snapshot() back in place."""
        ...


class StopReason(enum.StrEnum):
    """Why a training run ended, as written to the run log."""

    SOLVED = "solved"  # every cell wins; see routes_complete for whether every route is shortest
    STALLED = "stalled"  # `patience` games without a better win rate, before any cell set fully won
    BUDGET = "max_epochs"  # the epoch budget ran out before every cell won


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
    solved_at: int | None = None  # games played when every cell first won
    shortest_routes: int | None = None  # cells reached by a shortest route in the kept version
    first_shortest_routes: int | None = None  # the same count when every cell first won
    routes_complete: bool = False  # True when the kept version met target_shortest_share

    @property
    def explored_share(self) -> float:
        """The share of all moves that were random exploration rather than the greedy move."""
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
            "solved_at": self.solved_at,
            "first_shortest_routes": self.first_shortest_routes,
            "shortest_routes": self.shortest_routes,
            "routes_complete": self.routes_complete,
        }


class Explorer:
    """Chooses each move epsilon-greedily and counts how often it explored. Kept from CS 370."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Starts both move counters at zero; `rng` is the trainer's seeded generator.

        Sharing that generator with the replay memory means one seed fixes every random move
        and every sampled batch.
        """
        self.rng = rng
        self.explored = 0
        self.exploited = 0

    def choose(self, env: MazeEnv, agent: Learner, observation: np.ndarray, epsilon: float) -> int:
        """A random valid move with probability epsilon, else the agent's greedy move."""
        # Exploring picks among valid moves only, as in CS 370, so a random move never wastes a
        # game bumping into a wall.
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
        """Keeps what every run shares: the agent, the maze, the settings, the rewards and one generator.

        Nothing that changes during a run is created here. Each train() call builds its own
        environment, replay memory, epsilon and win history, so a second run cannot start where
        the first one ended (finding S1).
        """
        self.agent = agent
        self.maze = maze
        self.config = config
        self.rewards = rewards or Rewards()
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)

    def audit(self) -> AuditReport:
        """The route audit of the agent's current greedy policy, from one batched prediction."""
        return audit_policy(self.maze, cell_policy(self.maze, self.agent.greedy_actions), self.rewards)

    def train(self) -> TrainingResult:
        """Plays and learns game by game until a stopping rule fires, then returns what happened.

        Three rules can end a run. SOLVED: every cell wins and either enough cells take a
        shortest route or polishing stops improving (the best version is restored). STALLED:
        `patience` games pass, after the win-rate window fills, without a better win rate.
        BUDGET: `max_epochs` games are played first. Epsilon, the replay memory and the win
        history all start fresh on every call, so two runs never share state (finding S1).
        """
        cfg = self.config
        env = MazeEnv(self.maze, self.rewards)
        buffer = ReplayBuffer(cfg.memory_for(self.maze.size), self.rng)
        window = cfg.window_for(self.maze.size)
        explorer = Explorer(self.rng)
        starts = self.maze.free_cells
        needed = math.ceil(cfg.target_shortest_share * len(starts))  # cells that must take a shortest route
        epsilon = cfg.epsilon_start  # every run starts from the configured value
        recent: deque[bool] = deque(maxlen=window)  # the last `window` games, won or lost
        best = 0.0  # best full-window win rate so far, for the stall rule
        stale = 0  # games since that best improved
        moves = 0
        win_rate = 0.0
        history: list[EpochRecord] = []
        reason = StopReason.BUDGET
        solved_at: int | None = None
        first_shortest: int | None = None
        best_shortest = -1  # most cells on a shortest route in any audited version
        best_snapshot: Any = None  # the weights of that version, restored at the end
        polish_stale = 0  # polishing games since best_shortest last improved
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

            # The route audit runs only when the win rate says every cell might win, which is the
            # CS 370 order: win rate first, completion check second.
            improved = False
            if win_rate >= cfg.target_win_rate:
                report = self.audit()
                if report.all_win:
                    if solved_at is None:
                        solved_at, first_shortest = epoch + 1, report.optimal
                        log.info(
                            "every cell wins after %d games (%d of %d by a shortest route); polishing routes",
                            solved_at,
                            report.optimal,
                            report.cells,
                        )
                    if report.optimal > best_shortest:
                        best_shortest, best_snapshot, improved = report.optimal, self.agent.snapshot(), True
                    if report.optimal >= needed:
                        reason = StopReason.SOLVED
                        break
            if solved_at is not None:
                # Polishing: count games that add no shortest route; the stall rule no longer applies.
                polish_stale = 0 if improved else polish_stale + 1
                if polish_stale >= cfg.polish_patience:
                    reason = StopReason.SOLVED
                    break
                continue
            # Early stopping. Patience counts only after the window fills, because the win rate is
            # 0 until then, and a patience of 10 would otherwise stop every run before its first win rate.
            if full:
                if win_rate > best:
                    best, stale = win_rate, 0
                elif win_rate >= cfg.target_win_rate:
                    stale = 0  # at the target and waiting on the route check: not a stall
                else:
                    stale += 1
                if stale >= cfg.patience:
                    reason = StopReason.STALLED
                    break
        if best_snapshot is not None:
            self.agent.restore(best_snapshot)  # never end worse than the best version seen
            reason = StopReason.SOLVED
        seconds = time.perf_counter() - started
        complete = best_shortest >= needed if best_snapshot is not None else False
        log.info(
            "stopped (%s) after %d games and %d moves in %.1f s; best win rate %.3f; shortest routes %s",
            reason,
            len(history),
            moves,
            seconds,
            max(best, win_rate),
            f"{best_shortest} of {len(starts)}" if best_snapshot is not None else "not audited",
        )
        return TrainingResult(
            stop_reason=reason,
            epochs=len(history),
            moves=moves,
            explored_moves=explorer.explored,
            seconds=seconds,
            win_rate=win_rate,
            best_win_rate=max(best, win_rate),
            history=tuple(history),
            solved_at=solved_at,
            shortest_routes=best_shortest if best_snapshot is not None else None,
            first_shortest_routes=first_shortest,
            routes_complete=complete,
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
        # Each game starts on a randomly chosen free cell, as in CS 370.
        observation = env.reset(starts[int(self.rng.integers(len(starts)))])
        losses: list[float] = []
        status = Status.PLAYING
        while status is Status.PLAYING:
            action = explorer.choose(env, self.agent, observation, epsilon)
            next_observation, reward, status = env.step(action)
            # Only a win is terminal. Running out of reward budget is a cutoff, not a place in
            # the maze, so its target still looks one move ahead (a CS 370 design decision).
            buffer.add(Transition(observation, action, reward, next_observation, status is Status.WIN))
            # Learn after every move from a random minibatch of stored moves (experience replay).
            losses.append(self.agent.learn(buffer.sample(self.config.batch_size)))
            observation = next_observation
        return float(np.mean(losses)), len(losses), status is Status.WIN
