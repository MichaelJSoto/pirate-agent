"""Tests for the network, the learning step, and the model file safety checks.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.1.0
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
    configure_determinism(5)
    return DQNAgent.create(6, TrainingConfig(target_sync_every=3))


def batch(size=8):
    rng = np.random.default_rng(0)
    states = rng.random((size, 1, 6), dtype=np.float32)
    return [Transition(states[i], i % 4, -0.04 * i, states[(i + 1) % size], i == 0) for i in range(size)]


def test_network_matches_cs370():
    assert build_network(64).count_params() == 8_708


def test_greedy_action_is_the_largest_q_value(agent):
    observation = np.ones((1, 6), dtype=np.float32)
    assert agent.greedy_action(observation) == int(np.argmax(agent.q_values(observation)))


def test_batched_moves_match_single_moves(agent):
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
    fixed = batch()
    agent.target_sync_every = 10**9  # keep the targets still for this test
    losses = [agent.learn(fixed) for _ in range(60)]
    assert losses[-1] < losses[0] * 0.5


def test_target_network_syncs_on_schedule(agent):
    def same():
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
    path = tmp_path / "model.keras"
    digest = agent.save(path)
    assert digest == sha256_file(path)
    loaded = DQNAgent.from_file(path, digest, TrainingConfig(), input_size=6)
    observation = np.full((1, 6), 0.5, dtype=np.float32)
    assert np.allclose(loaded.q_values(observation), agent.q_values(observation))


def test_changed_file_is_refused(agent, tmp_path):
    path = tmp_path / "model.keras"
    digest = agent.save(path)
    data = bytearray(path.read_bytes())
    data[-10] ^= 0xFF  # flip one byte after saving
    path.write_bytes(bytes(data))
    with pytest.raises(ModelIntegrityError, match="does not match its run log"):
        load_network(path, digest)


def test_h5_and_other_suffixes_are_refused(agent, tmp_path):
    path = tmp_path / "model.keras"
    digest = agent.save(path)
    renamed = tmp_path / "model.h5"
    renamed.write_bytes(path.read_bytes())
    with pytest.raises(ModelFileError, match="only .keras files are loaded"):
        load_network(renamed, digest)


def test_smuggled_archive_entry_is_refused(agent, tmp_path):
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
    hostile = keras.Sequential(
        [keras.Input(shape=(6,)), keras.layers.Lambda(lambda x: x * 2), keras.layers.Dense(4)]
    )
    path = tmp_path / "hostile.keras"
    hostile.save(path)
    with pytest.raises(ModelFileError, match="'Lambda'"):
        load_network(path, sha256_file(path))


def test_wrong_maze_size_is_refused(agent, tmp_path):
    path = tmp_path / "model.keras"
    digest = agent.save(path)
    with pytest.raises(ModelFileError, match="maze of 6 cells, not 64"):
        load_network(path, digest, input_size=64)


def test_saving_as_h5_is_refused(agent, tmp_path):
    with pytest.raises(ModelFileError, match="saved only as .keras"):
        agent.save(tmp_path / "model.h5")
