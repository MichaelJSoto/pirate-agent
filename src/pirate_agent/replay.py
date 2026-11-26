"""Bounded experience replay memory for the deep Q-learning agent.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)
Origin: replaces the memory half of GameExperience.py, a starter file SNHU supplied for CS 370.

Intent:
    Store the most recent moves the pirate made and hand the learner a random minibatch of
    them, so consecutive, strongly correlated moves are not learned from in order.

Decision:
    The starter kept moves in a Python list and ran `del memory[0]` on every insert once the
    list was full, which shifts every remaining element: O(n) per move (code review finding
    S5). A collections.deque with maxlen drops the oldest move in O(1). Indexing into a deque
    costs O(n / 64) in CPython because it is stored in 64-item blocks, which is negligible at
    the 512-move capacity used here; a preallocated NumPy ring buffer would be the next step
    if the capacity ever grew into the hundreds of thousands. Building training targets moved
    to agent.py, so this class only stores and samples and needs no TensorFlow.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np


class Transition(NamedTuple):
    """One stored move. The starter file called this an "episode"."""

    state: np.ndarray  # (1, cells) observation before the move
    action: int
    reward: float
    next_state: np.ndarray  # (1, cells) observation after the move
    done: bool  # True only when the move reached the treasure


class ReplayBuffer:
    """A first-in, first-out memory of at most `capacity` transitions."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        """Creates an empty memory of `capacity` moves that samples with `rng`.

        The capacity is checked by hand because deque would accept 0 (a memory that keeps
        nothing) and True (as 1) without complaint. The trainer passes in its own seeded
        generator, so one seed fixes every batch.
        """
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be a positive whole number; got {capacity!r}")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = rng

    @property
    def capacity(self) -> int:
        """The most moves the memory holds before the oldest is dropped."""
        return int(self._items.maxlen or 0)

    def __len__(self) -> int:
        """The number of moves stored now, never more than `capacity`. O(1)."""
        return len(self._items)

    def add(self, transition: Transition) -> None:
        """Stores one move; once full, the oldest move is dropped in O(1)."""
        if not isinstance(transition.done, bool):
            # The CS 370 notebook found that storing the status string here made every move
            # look terminal, because any non-empty string is truthy. Refuse anything but a bool.
            raise TypeError(f"done must be a bool, not {type(transition.done).__name__}")
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """A random minibatch, drawn with replacement only while the memory holds fewer moves."""
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {batch_size}")
        count = len(self._items)
        # Early in training the memory can hold fewer moves than a batch needs, so only then may a
        # move be picked twice; after that every move in a batch is distinct.
        picks = self._rng.choice(count, size=batch_size, replace=count < batch_size)
        return [self._items[int(i)] for i in picks]
