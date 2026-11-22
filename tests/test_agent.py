"""Tests for the network, the learning step, and the model file safety checks.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Check that the rebuilt network matches CS 370 (8,708 weights for the 8 x 8 maze), that one
    learning step really learns, and that every unsafe model file is refused before Keras
    reads it.

Decision:
    The attack tests build real malicious files: a model with a Lambda layer (arbitrary Python
    code), a renamed .h5 file, an archive with a smuggled extra entry, and a file changed after
    saving. Each must fail with ModelFileError, never by running anything.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

pytestmark = pytest.mark.tf

keras = pytest.importorskip("keras")

from pirate_agent.agent import (  # noqa: E402
    DQNAgent,
    ModelFileError,
    ModelIntegrityError,
    build_network,
    configure_determinism,
    load_network,
    sha256_bytes,
    sha256_file,
)
from pirate_agent.config import TrainingConfig  # noqa: E402
from pirate_agent.replay import Transition  # noqa: E402


@pytest.fixture
def agent():
    """A seeded agent for a six-cell maze whose target network syncs every 3 gradient steps."""
    configure_determinism(5)
    return DQNAgent.create(6, TrainingConfig(target_sync_every=3))


def batch(size=8):
    """A fixed minibatch of `size` seeded moves in which only the first one ends its game.

    Mixing one terminal move with ordinary ones exercises both forms of the Bellman target:
    the reward alone, and the reward plus the discounted best next Q-value.
    """
    rng = np.random.default_rng(0)
    states = rng.random((size, 1, 6), dtype=np.float32)
    return [Transition(states[i], i % 4, -0.04 * i, states[(i + 1) % size], i == 0) for i in range(size)]


def test_network_matches_cs370():
    """The network for the 8 x 8 maze has the CS 370 model's 8,708 weights.

    Declaring the input with keras.Input instead of the deprecated input_shape (finding S4)
    changed how the network is built, not its size.
    """
    assert build_network(64).count_params() == 8_708


def test_greedy_action_is_the_largest_q_value(agent):
    """The greedy move is the action with the largest predicted Q-value, the rule the audit relies on."""
    observation = np.ones((1, 6), dtype=np.float32)
    assert agent.greedy_action(observation) == int(np.argmax(agent.q_values(observation)))


def test_batched_moves_match_single_moves(agent):
    """One batched prediction picks the same moves as one prediction per cell.

    This is what lets the 2.1.0 route audit use a single network call and still report exactly
    what the step-by-step audit would.
    """
    from pirate_agent.audit import audit_policy, cell_policy
    from pirate_agent.maze import generate_maze

    observations = np.random.default_rng(1).random((20, 6), dtype=np.float32)
    singles = [agent.greedy_action(row.reshape(1, -1)) for row in observations]
    assert agent.greedy_actions(observations).tolist() == singles
    maze = generate_maze(2, 3, wall_density=0.0, seed=0)  # six cells, matching the agent's input
    assert audit_policy(maze, cell_policy(maze, agent.greedy_actions)) == audit_policy(
        maze, agent.greedy_action
    )


def test_snapshot_and_restore(agent):
    """restore() brings back the exact weights snapshot() saved, in both the online and target networks.

    Route polishing depends on this to return to its best version after later games make it worse.
    """
    observation = np.full((1, 6), 0.5, dtype=np.float32)
    before, saved = agent.q_values(observation), agent.snapshot()
    agent.learn(batch())
    assert not np.allclose(agent.q_values(observation), before)
    agent.restore(saved)
    assert np.allclose(agent.q_values(observation), before)
    assert all(
        np.array_equal(a, b)
        for a, b in zip(agent.network.get_weights(), agent.target_network.get_weights(), strict=True)
    )


def test_learning_lowers_the_loss_on_a_fixed_batch(agent):
    """Sixty learning steps on one batch, with the targets held still, at least halve the loss."""
    fixed = batch()
    agent.target_sync_every = 10**9  # keep the targets still for this test
    losses = [agent.learn(fixed) for _ in range(60)]
    assert losses[-1] < losses[0] * 0.5


def test_target_network_syncs_on_schedule(agent):
    """The target network copies the online weights on every third update and not before.

    Between syncs the targets the online network chases hold still, which is the point of
    having a target network at all.
    """

    def same():
        """True when the online and target networks hold identical weights, layer by layer."""
        return all(
            np.array_equal(a, b)
            for a, b in zip(agent.network.get_weights(), agent.target_network.get_weights(), strict=True)
        )

    agent.learn(batch())
    agent.learn(batch())
    assert not same()  # two updates, sync every three
    agent.learn(batch())
    assert same()


def test_save_and_load_round_trip(agent, tmp_path):
    """A saved model loads back through every safety check with the same Q-values.

    save() returns the SHA-256 that the run log records. The CS 370 model was never saved at
    all, so every result disappeared with the notebook kernel (finding S6).
    """
    path = tmp_path / "model.keras"
    digest = agent.save(path)
    assert digest == sha256_file(path)
    loaded = DQNAgent.from_file(path, digest, TrainingConfig(), input_size=6)
    observation = np.full((1, 6), 0.5, dtype=np.float32)
    assert np.allclose(loaded.q_values(observation), agent.q_values(observation))


def test_changed_file_is_refused(agent, tmp_path):
    """A model with one byte flipped after saving fails the SHA-256 check before Keras opens it (CWE-502)."""
    path = tmp_path / "model.keras"
    digest = agent.save(path)
    data = bytearray(path.read_bytes())
    data[-10] ^= 0xFF  # flip one byte after saving
    path.write_bytes(bytes(data))
    with pytest.raises(ModelIntegrityError, match="does not match its run log"):
        load_network(path, digest)


def test_h5_and_other_suffixes_are_refused(agent, tmp_path):
    """A valid model renamed to .h5 is refused by its suffix, even though its hash matches (CWE-502).

    Only a .keras archive can be opened and inspected entry by entry before Keras reads it.
    """
    path = tmp_path / "model.keras"
    digest = agent.save(path)
    renamed = tmp_path / "model.h5"
    renamed.write_bytes(path.read_bytes())
    with pytest.raises(ModelFileError, match="only .keras files are loaded"):
        load_network(renamed, digest)


def test_smuggled_archive_entry_is_refused(agent, tmp_path):
    """An archive with an extra file added, here a Python script, is refused even with a matching hash.

    A .keras file may hold only its three known entries (CWE-502), so the hash check alone is
    not trusted to catch a file that was built hostile from the start.
    """
    path = tmp_path / "model.keras"
    agent.save(path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(buffer, "w") as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.writestr("payload.py", "print('should never run')")
    tampered = tmp_path / "tampered.keras"
    tampered.write_bytes(buffer.getvalue())
    with pytest.raises(ModelFileError, match="must hold exactly"):
        load_network(tampered, sha256_bytes(buffer.getvalue()))


def test_lambda_layer_is_refused_before_keras_reads_it(tmp_path):
    """A real model with a Lambda layer, which can carry arbitrary Python code, is refused by the allowlist.

    The layer classes are read from the archive's config.json, so Keras never deserializes the
    hostile layer (CWE-502).
    """
    hostile = keras.Sequential(
        [keras.Input(shape=(6,)), keras.layers.Lambda(lambda x: x * 2), keras.layers.Dense(4)]
    )
    path = tmp_path / "hostile.keras"
    hostile.save(path)
    with pytest.raises(ModelFileError, match="'Lambda'"):
        load_network(path, sha256_file(path))


def test_wrong_maze_size_is_refused(agent, tmp_path):
    """A model trained on a 6-cell maze is refused for a 64-cell maze when it loads, not in mid-training."""
    path = tmp_path / "model.keras"
    digest = agent.save(path)
    with pytest.raises(ModelFileError, match="maze of 6 cells, not 64"):
        load_network(path, digest, input_size=64)


def test_saving_as_h5_is_refused(agent, tmp_path):
    """save() refuses the .h5 suffix too, so the project never writes a format it would refuse to read."""
    with pytest.raises(ModelFileError, match="saved only as .keras"):
        agent.save(tmp_path / "model.h5")
