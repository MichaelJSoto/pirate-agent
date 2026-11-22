"""Tests for the bounded replay buffer.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Show that the buffer keeps only the newest moves, samples the way the starter did, and
    refuses the status-string mistake the CS 370 notebook documented.

Decision:
    The generator is seeded in every test, so a failing sample can be replayed exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pirate_agent.replay import ReplayBuffer, Transition


def make(i: int, done: bool = False) -> Transition:
    """A move whose state, action and reward all come from `i`, so a sample shows which moves it holds."""
    state = np.full((1, 4), i, dtype=np.float32)
    return Transition(state, i % 4, float(i), state + 1, done)


def test_capacity_drops_the_oldest_first():
    """Once full, the buffer drops its oldest moves first, as the O(1) deque must (finding S5)."""
    buffer = ReplayBuffer(3, np.random.default_rng(0))
    for i in range(5):
        buffer.add(make(i))
    assert len(buffer) == 3
    assert buffer.capacity == 3
    assert min(t.reward for t in buffer.sample(50)) == 2.0  # moves 0 and 1 are gone


def test_sample_without_replacement_once_full_enough():
    """Once the buffer holds enough moves, a batch never repeats one: 10 from 10 returns each exactly once."""
    buffer = ReplayBuffer(10, np.random.default_rng(1))
    for i in range(10):
        buffer.add(make(i))
    batch = buffer.sample(10)
    assert sorted(t.reward for t in batch) == [float(i) for i in range(10)]


def test_sample_with_replacement_while_small():
    """Below a batch's worth of moves, sampling is with replacement, so learning can start at once."""
    buffer = ReplayBuffer(10, np.random.default_rng(2))
    buffer.add(make(7))
    batch = buffer.sample(4)
    assert len(batch) == 4
    assert all(t.reward == 7.0 for t in batch)


def test_same_seed_same_batches():
    """Two buffers with the same seed and moves draw the same batch, so a seeded run repeats exactly."""
    first, second = (ReplayBuffer(8, np.random.default_rng(42)) for _ in range(2))
    for i in range(8):
        first.add(make(i))
        second.add(make(i))
    assert [t.reward for t in first.sample(5)] == [t.reward for t in second.sample(5)]


def test_status_string_is_refused_as_done():
    """A status string such as "not_over" is refused as the done flag.

    Any non-empty string is truthy, so it would mark every move as the end of a game: the
    mistake the CS 370 notebook documented.
    """
    buffer = ReplayBuffer(4, np.random.default_rng(0))
    state = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="done must be a bool"):
        buffer.add(Transition(state, 0, -0.04, state, "not_over"))


@pytest.mark.parametrize("capacity", [0, -1, 2.5, True])
def test_bad_capacity(capacity):
    """Zero, negative, fractional and boolean capacities are all refused with the same ValueError."""
    with pytest.raises(ValueError):
        ReplayBuffer(capacity, np.random.default_rng(0))


def test_empty_and_bad_batch():
    """Sampling an empty buffer, or asking for a batch of zero, fails with a ValueError that says which."""
    buffer = ReplayBuffer(4, np.random.default_rng(0))
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1)
    buffer.add(make(0))
    with pytest.raises(ValueError, match="batch_size"):
        buffer.sample(0)
