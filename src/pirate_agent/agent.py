"""Deep Q-network agent: the network, its target copy, one learning step, and safe saving and loading.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)
Origin: build_model() and train_step() from the CS 370 notebook, and get_data() from the
GameExperience.py starter file.

Intent:
    Own everything about the neural network: how it is built, how it picks a move, how it
    learns from a minibatch, and how it is written to and read from disk.

Decision:
    The agent owns its online network, target network and optimizer, so nothing depends on a
    global `model` (code review finding S1) and two agents can train side by side. The
    network declares its input with keras.Input instead of the `input_shape` argument Keras 3
    warns about, and it is not compiled, because CS 370 compiled an optimizer that training
    never used (S4). One compiled TensorFlow function builds the Bellman targets and takes
    the gradient step, where CS 370 made three separate calls per move.

    A model file is untrusted input (CWE-502, deserialization of untrusted data). Before
    Keras reads one, the bytes are read once, their SHA-256 must match the run log, the
    archive may hold only the three files Keras writes, and its configuration may name only
    the classes this network uses. Only then is a private copy of those same bytes loaded,
    with Keras's safe_mode on as a second line of defense. Loading a private copy means the
    file cannot be swapped between the check and the load.
"""
# ruff: noqa: E402  (TensorFlow reads these settings once, when it is first imported)

from __future__ import annotations

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU only, so a seeded run repeats on any machine
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # one fixed order of floating-point operations
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's informational messages
logging.getLogger("tensorflow").setLevel(logging.ERROR)  # and its notice about GPUs on Windows

import hashlib
import hmac
import io
import json
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from .config import TrainingConfig
from .maze import NUM_ACTIONS
from .replay import Transition

MODEL_SUFFIX = ".keras"
MAX_MODEL_BYTES = 20 * 1024 * 1024  # a 32 x 32 maze network is about 8.5 MB; refuse anything larger
ARCHIVE_ENTRIES = frozenset({"metadata.json", "config.json", "model.weights.h5"})
# Every (module, class) pair a saved network of this design contains. Anything else, such as a
# Lambda layer (arbitrary Python) or a TFSMLayer (an external SavedModel), is refused.
ALLOWED_CLASSES = frozenset(
    {
        ("keras", "Sequential"),
        ("keras", "DTypePolicy"),
        ("keras.layers", "InputLayer"),
        ("keras.layers", "Dense"),
        ("keras.layers", "PReLU"),
        ("keras.initializers", "GlorotUniform"),
        ("keras.initializers", "Zeros"),
    }
)


class ModelFileError(ValueError):
    """A model file that is not safe or not suitable to load."""


class ModelIntegrityError(ModelFileError):
    """A model file whose bytes do not match the hash recorded when it was saved."""


def configure_determinism(seed: int) -> None:
    """Seeds Python, NumPy and TensorFlow together and asks TensorFlow for deterministic kernels."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_network(input_size: int, num_actions: int = NUM_ACTIONS) -> keras.Sequential:
    """The CS 370 network: two hidden layers as wide as the maze, PReLU activations, one Q-value per move.

    For the 8 x 8 maze that is 8,708 weights, the same count CS 370 reported.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            keras.layers.Dense(input_size),
            keras.layers.PReLU(),
            keras.layers.Dense(input_size),
            keras.layers.PReLU(),
            keras.layers.Dense(num_actions),
        ]
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _check_classes(node: Any) -> None:
    """Walks a Keras configuration and refuses any class outside ALLOWED_CLASSES."""
    if isinstance(node, dict):
        if "class_name" in node:
            pair = (node.get("module"), node.get("class_name"))
            if pair not in ALLOWED_CLASSES:
                raise ModelFileError(
                    f"the model uses {pair[1]!r} from {pair[0]!r}, which this program does not load"
                )
            if node.get("registered_name") is not None:
                raise ModelFileError(f"the model uses a custom object {node['registered_name']!r}")
        for value in node.values():
            _check_classes(value)
    elif isinstance(node, list):
        for value in node:
            _check_classes(value)


def inspect_model_bytes(data: bytes, expected_sha256: str, name: str = "model") -> None:
    """Every check that runs before Keras sees a model file. Raises ModelFileError on any failure."""
    if len(data) > MAX_MODEL_BYTES:
        raise ModelFileError(f"{name} is {len(data):,} bytes; the limit is {MAX_MODEL_BYTES:,}")
    digest = sha256_bytes(data)
    if not hmac.compare_digest(digest, str(expected_sha256).lower()):
        raise ModelIntegrityError(
            f"{name} does not match its run log (SHA-256 {digest[:12]}..., expected "
            f"{str(expected_sha256)[:12]}...); the file may have been changed or replaced"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            if names != ARCHIVE_ENTRIES:
                raise ModelFileError(
                    f"{name} holds {sorted(names)}; a model file must hold exactly {sorted(ARCHIVE_ENTRIES)}"
                )
            if sum(info.file_size for info in archive.infolist()) > 4 * MAX_MODEL_BYTES:
                raise ModelFileError(f"{name} expands to more than {4 * MAX_MODEL_BYTES:,} bytes")
            config = json.loads(archive.read("config.json"))
    except zipfile.BadZipFile:
        raise ModelFileError(f"{name} is not a Keras .keras archive") from None
    except json.JSONDecodeError:
        raise ModelFileError(f"{name} has an unreadable config.json") from None
    _check_classes(config)


def load_network(path: str | Path, expected_sha256: str, input_size: int | None = None) -> keras.Model:
    """Loads a saved network only after inspect_model_bytes() passes. Returns the uncompiled network."""
    path = Path(path)
    if path.suffix != MODEL_SUFFIX:
        raise ModelFileError(
            f"{path.name}: only {MODEL_SUFFIX} files are loaded; .h5 and pickle files are refused"
        )
    if not path.is_file():
        raise ModelFileError(f"{path.name}: no such model file")
    if path.stat().st_size > MAX_MODEL_BYTES:
        raise ModelFileError(f"{path.name} is larger than {MAX_MODEL_BYTES:,} bytes")
    data = path.read_bytes()  # read once; every check and the load use these same bytes
    inspect_model_bytes(data, expected_sha256, path.name)
    with tempfile.TemporaryDirectory() as private:
        copy = Path(private) / ("checked" + MODEL_SUFFIX)
        copy.write_bytes(data)
        network = keras.saving.load_model(copy, compile=False, safe_mode=True)
    if network.output_shape[-1] != NUM_ACTIONS:
        raise ModelFileError(f"{path.name} returns {network.output_shape[-1]} values; expected {NUM_ACTIONS}")
    if input_size is not None and network.input_shape[-1] != input_size:
        raise ModelFileError(
            f"{path.name} was trained on a maze of {network.input_shape[-1]} cells, not {input_size}"
        )
    return network


class DQNAgent:
    """A deep Q-learning agent with a target network, trained one minibatch at a time."""

    def __init__(
        self,
        network: keras.Model,
        learning_rate: float = 0.001,
        discount: float = 0.95,
        target_sync_every: int = 50,
    ) -> None:
        if network.output_shape[-1] != NUM_ACTIONS:
            raise ValueError(f"the network must return {NUM_ACTIONS} Q-values")
        self.network = network
        self.target_network = keras.models.clone_model(network)
        self.target_network.set_weights(network.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer.build(network.trainable_variables)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.discount = float(discount)
        self.target_sync_every = int(target_sync_every)
        self.updates = 0  # gradient steps taken so far
        self._predict = tf.function(self._predict_step, reduce_retracing=True)
        self._learn = tf.function(self._learn_step, reduce_retracing=True)

    @classmethod
    def create(cls, input_size: int, config: TrainingConfig) -> DQNAgent:
        """A new agent with a freshly initialized network for a maze of `input_size` cells."""
        return cls(build_network(input_size), config.learning_rate, config.discount, config.target_sync_every)

    @classmethod
    def from_file(
        cls, path: str | Path, expected_sha256: str, config: TrainingConfig, input_size: int | None = None
    ) -> DQNAgent:
        """An agent around a network loaded with every safety check in load_network()."""
        network = load_network(path, expected_sha256, input_size)
        return cls(network, config.learning_rate, config.discount, config.target_sync_every)

    @property
    def input_size(self) -> int:
        return int(self.network.input_shape[-1])

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """The four Q-values for one observation."""
        batch = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return self._predict(tf.constant(batch)).numpy()[0]

    def greedy_action(self, observation: np.ndarray) -> int:
        """The move with the highest Q-value. This is the trained pirate's policy."""
        return int(np.argmax(self.q_values(observation)))

    def learn(self, batch: Sequence[Transition]) -> float:
        """One gradient step on a minibatch; refreshes the target network on schedule. Returns the loss."""
        states = np.concatenate([t.state for t in batch]).astype(np.float32, copy=False)
        next_states = np.concatenate([t.next_state for t in batch]).astype(np.float32, copy=False)
        actions = np.fromiter((t.action for t in batch), dtype=np.int32, count=len(batch))
        rewards = np.fromiter((t.reward for t in batch), dtype=np.float32, count=len(batch))
        dones = np.fromiter((t.done for t in batch), dtype=np.float32, count=len(batch))
        loss = self._learn(
            tf.constant(states),
            tf.constant(actions),
            tf.constant(rewards),
            tf.constant(next_states),
            tf.constant(dones),
        )
        self.updates += 1
        if self.updates % self.target_sync_every == 0:
            self.sync_target()
        return float(loss)

    def sync_target(self) -> None:
        """Copies the online network's weights into the target network."""
        self.target_network.set_weights(self.network.get_weights())

    def save(self, path: str | Path) -> str:
        """Writes the network as a .keras file and returns its SHA-256 for the run log."""
        path = Path(path)
        if path.suffix != MODEL_SUFFIX:
            raise ModelFileError(f"{path.name}: models are saved only as {MODEL_SUFFIX} files")
        self.network.save(path)
        return sha256_file(path)

    def _predict_step(self, batch: tf.Tensor) -> tf.Tensor:
        return self.network(batch, training=False)

    def _learn_step(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
    ) -> tf.Tensor:
        # The Bellman target, exactly as GameExperience.get_data() built it: the online network's
        # current Q-values, with the taken action's entry replaced by reward + discount * the
        # target network's best next Q-value (or by the reward alone when the move won).
        q_values = self.network(states, training=False)
        q_next = self.target_network(next_states, training=False)
        bellman = rewards + (1.0 - dones) * self.discount * tf.reduce_max(q_next, axis=1)
        taken = tf.one_hot(actions, NUM_ACTIONS, dtype=q_values.dtype)
        targets = tf.stop_gradient(q_values * (1.0 - taken) + taken * bellman[:, None])
        with tf.GradientTape() as tape:
            loss = self.loss_fn(targets, self.network(states, training=True))
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables, strict=True))
        return loss
