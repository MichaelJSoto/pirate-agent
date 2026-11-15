"""Command-line interface: train, evaluate, play and generate.

Programmer: Michael Soto
Date: November 15, 2026
Version: 2.1.1
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Let another developer use the agent without opening a notebook: train from a
    configuration file, check a saved run, watch the trained pirate play from any cell, and
    create new mazes.

Decision:
    Every command validates its input and fails with one plain sentence and a distinct exit
    code instead of a stack trace: 2 for a usage error (argparse's own code), 3 for a bad
    maze, configuration or run log, 4 for a model file that fails its safety checks, and 5
    when an evaluated model does not win from every cell. TensorFlow is imported only inside
    the commands that need it, so `generate` and `--help` start at once. Progress goes to
    logging; the results a command exists to show are printed to standard output.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .audit import audit_policy, play_greedy, route_grid, route_map
from .config import AppConfig, ConfigError, TrainingConfig, load_config
from .maze import MAZE_SUFFIX, Cell, Maze, MazeEnv, MazeError, generate_maze, load_maze, shortest_moves
from .runlog import (
    HISTORY_NAME,
    MODEL_NAME,
    RUN_LOG_NAME,
    SCHEMA_VERSION,
    RunLogError,
    make_run_dir,
    maze_from_log,
    maze_record,
    new_run_id,
    package_versions,
    read_run_log,
    write_history,
    write_run_log,
)

# Exit codes, so a script calling this program can tell the failures apart (2 is argparse's own).
EXIT_OK = 0
EXIT_INPUT = 3
EXIT_MODEL = 4
EXIT_AUDIT = 5

log = logging.getLogger("pirate_agent")


def _maze_from_config(config: AppConfig) -> tuple[Maze, str]:
    """The maze a configuration names, with a short description of where it came from."""
    if config.maze_file is not None:
        return load_maze(config.maze_file), config.maze_file.name
    settings = config.generate
    if settings is None:  # parse_config already guarantees one source; this keeps it explicit
        raise ConfigError("the configuration names no maze")
    maze = generate_maze(settings.rows, settings.cols, settings.wall_density, settings.seed)
    size = f"{settings.rows} x {settings.cols}"
    return maze, f"generated {size}, wall density {settings.wall_density}, seed {settings.seed}"


def _shown(path: Path) -> str:
    """A path relative to the working folder when possible, so output never shows a home folder."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return Path(path).name


def _parse_cell(text: str) -> Cell:
    """'ROW,COL' as a cell, for --start."""
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise MazeError(f"--start must look like ROW,COL with whole numbers, for example 0,0; got {text!r}")
    return int(parts[0]), int(parts[1])


def _load_run(run_dir: Path) -> tuple[dict[str, Any], Maze, Any]:
    """Reads a run folder and loads its model with every safety check. TensorFlow loads here."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RunLogError(f"{run_dir} is not a run folder")
    record = read_run_log(run_dir / RUN_LOG_NAME)
    maze = maze_from_log(record)
    from .agent import DQNAgent

    agent = DQNAgent.from_file(run_dir / MODEL_NAME, record["model"]["sha256"], TrainingConfig(), maze.size)
    return record, maze, agent


def cmd_train(args: argparse.Namespace) -> int:
    """`train`: trains and audits a new agent, then saves its model, history and run log in a new folder."""
    config = load_config(args.config)
    training = config.training if args.seed is None else dataclasses.replace(config.training, seed=args.seed)
    maze, source = _maze_from_config(config)  # validated before TensorFlow is even imported
    from .agent import DQNAgent, configure_determinism
    from .trainer import Trainer

    # Seed everything before the network is built, so its starting weights repeat too.
    configure_determinism(training.seed)
    agent = DQNAgent.create(maze.size, training)
    result = Trainer(agent, maze, training).train()
    report = audit_policy(maze, agent.greedy_action)

    # The model's SHA-256 goes into the run log, which is what evaluate and play check it against.
    run_id = new_run_id(training.seed)
    run_dir = make_run_dir(config.runs_dir, run_id)
    digest = agent.save(run_dir / MODEL_NAME)
    write_history(run_dir / HISTORY_NAME, result.history)
    write_run_log(
        run_dir / RUN_LOG_NAME,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "config_file": Path(args.config).name,
            "versions": package_versions(),
            "maze": maze_record(maze, source),
            "training": training.to_dict(),
            "result": result.to_dict(),
            "audit": report.to_dict(),
            "model": {
                "file": MODEL_NAME,
                "sha256": digest,
                "bytes": (run_dir / MODEL_NAME).stat().st_size,
                "parameters": int(agent.network.count_params()),
            },
        },
    )
    print(f"Run {run_id} saved in {_shown(run_dir)}")
    print(
        f"Stopped: {result.stop_reason} after {result.epochs:,} games and {result.moves:,} moves "
        f"in {result.seconds:.1f} s ({result.explored_share:.1%} of moves explored)"
    )
    if result.solved_at is not None:
        print(
            f"Every cell won after {result.solved_at:,} games; polishing took the shortest routes from "
            f"{result.first_shortest_routes} to {result.shortest_routes} of {report.cells}"
        )
    print(report.summary())
    print()
    print(route_grid(report, maze))
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    """`evaluate`: reloads a saved run's model with every safety check and audits it from every cell."""
    from .agent import ModelFileError

    try:
        record, maze, agent = _load_run(args.run)
    except ModelFileError as error:
        log.error("model refused: %s", error)
        return EXIT_MODEL
    report = audit_policy(maze, agent.greedy_action)
    print(f"Run {record['run_id']}: model hash verified, loaded with safe mode on")
    print(report.summary())
    print()
    print(route_grid(report, maze))
    # A fresh audit that disagrees with the logged one means the run cannot be reproduced as recorded.
    logged = record.get("audit")
    if isinstance(logged, dict) and logged != report.to_dict():
        log.warning("this audit differs from the one in the run log: %s", logged)
    return EXIT_OK if report.all_win else EXIT_AUDIT


def cmd_play(args: argparse.Namespace) -> int:
    """`play`: draws the trained pirate's route from one starting cell next to the shortest possible."""
    from .agent import ModelFileError

    start = _parse_cell(args.start)
    try:
        _, maze, agent = _load_run(args.run)
    except ModelFileError as error:
        log.error("model refused: %s", error)
        return EXIT_MODEL
    env = MazeEnv(maze)
    route, won = play_greedy(env, agent.greedy_action, start)  # reset() rejects a bad start cell
    fewest = shortest_moves(maze.grid, maze.target)[start]
    print(route_map(maze, route))
    outcome = "Reached" if won else "Did not reach"
    print(f"{outcome} the treasure from {start} in {len(route) - 1} moves; the fewest possible is {fewest}.")
    return EXIT_OK if won else EXIT_AUDIT


def cmd_generate(args: argparse.Namespace) -> int:
    """`generate`: writes a random .maze file in which every free cell can reach the treasure."""
    maze = generate_maze(args.rows, args.cols, args.density, args.seed)
    out = Path(args.out)
    if out.suffix != MAZE_SUFFIX:
        raise MazeError(f"{out.name}: maze files must end in '{MAZE_SUFFIX}'")
    # Never overwrite a maze file by accident; replacing one takes an explicit --force.
    if out.exists() and not args.force:
        raise MazeError(f"{out.name} already exists; pass --force to replace it")
    note = f"Generated {args.rows} x {args.cols} maze, wall density {args.density}, seed {args.seed}"
    out.write_text(maze.to_text(comments=[note]), encoding="ascii", newline="\n")
    print(maze.to_text(), end="")
    print(f"{len(maze.free_cells) + 1} of {maze.size} cells are open. Saved to {_shown(out)}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """The argument parser: global options, then one subcommand per task, each tied to its handler."""
    parser = argparse.ArgumentParser(
        prog="pirate-agent",
        description="Train and check a deep Q-learning pirate that finds the treasure in a maze.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="show warnings and errors only")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    train = commands.add_parser("train", help="train a new agent from a configuration file")
    train.add_argument("--config", required=True, type=Path, help="a .toml configuration file")
    train.add_argument("--seed", type=int, help="override the configuration's training seed")
    train.set_defaults(handler=cmd_train)

    evaluate = commands.add_parser("evaluate", help="check a saved run's model from every free cell")
    evaluate.add_argument("--run", required=True, type=Path, help="a run folder written by train")
    evaluate.set_defaults(handler=cmd_evaluate)

    play = commands.add_parser("play", help="show the trained pirate's route from one cell")
    play.add_argument("--run", required=True, type=Path, help="a run folder written by train")
    play.add_argument("--start", required=True, help="the starting cell as ROW,COL, for example 0,0")
    play.set_defaults(handler=cmd_play)

    generate = commands.add_parser("generate", help="write a random maze in which every free cell can win")
    generate.add_argument("--rows", type=int, default=8)
    generate.add_argument("--cols", type=int, default=8)
    generate.add_argument("--density", type=float, default=0.25, help="chance a cell starts as a wall")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--out", required=True, help="the .maze file to write")
    generate.add_argument("--force", action="store_true", help="replace the file if it exists")
    generate.set_defaults(handler=cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs one command and returns its exit code; an input error becomes one log line, not a traceback."""
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        return args.handler(args)
    # A bad maze, configuration or run log is the user's input, not a bug, so it gets one plain
    # sentence and exit code 3 instead of a stack trace.
    except (MazeError, ConfigError, RunLogError) as error:
        log.error("%s", error)
        return EXIT_INPUT
    except KeyboardInterrupt:
        log.error("stopped by the user")
        return 130
