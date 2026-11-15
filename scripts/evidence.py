"""Reproduces every number the README and the enhancement narrative report.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Turn each claim about the enhancement into a file anyone can regenerate: a ten-seed sweep
    on the CS 370 maze, a run that early stopping ends, a run on a generated maze, the replay
    buffer timing, and the dependency audit of the old and new version pins.

Decision:
    Results are written as JSON (for checking) and one Markdown summary (for reading) under
    docs/evidence/. Seeds are fixed and listed, and no output contains a machine path, so the
    files can be committed and compared between runs.

Run:  python scripts/evidence.py            (about five minutes on a laptop CPU)
      python scripts/evidence.py --only replay audit
      <CS 370 venv>/python scripts/evidence.py --only reference   (the same run on the old libraries)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pirate_agent.audit import audit_policy  # noqa: E402
from pirate_agent.config import load_config  # noqa: E402
from pirate_agent.maze import generate_maze, load_maze  # noqa: E402

OUT = ROOT / "docs" / "evidence"
SEEDS = [1671, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # the CS 370 seed first, then nine more
ORIGINAL_PINS = ["numpy==1.26.0", "tensorflow==2.17.0", "keras==3.5.0"]  # the CS 370 environment
log = logging.getLogger("evidence")


def train_once(maze, training):
    """One seeded training run; returns the result and the audit."""
    from pirate_agent.agent import DQNAgent, configure_determinism
    from pirate_agent.trainer import Trainer

    configure_determinism(training.seed)
    agent = DQNAgent.create(maze.size, training)
    result = Trainer(agent, maze, training).train()
    report = audit_policy(maze, agent.greedy_action)
    return result, report


def row(seed, result, report):
    return {
        "seed": seed,
        "stop_reason": str(result.stop_reason),
        "games": result.epochs,
        "moves": result.moves,
        "seconds": round(result.seconds, 1),
        "explored_share": round(result.explored_share, 3),
        "wins": report.wins,
        "cells": report.cells,
        "shortest_routes": report.optimal,
    }


def seed_sweep():
    config = load_config(ROOT / "configs" / "default.toml")
    maze = load_maze(config.maze_file)
    rows = []
    for seed in SEEDS:
        result, report = train_once(maze, dataclasses.replace(config.training, seed=seed))
        rows.append(row(seed, result, report))
        log.info("seed %s: %s", seed, rows[-1])
    games = [r["games"] for r in rows]
    summary = {
        "maze": "cs370.maze",
        "seeds": SEEDS,
        "runs": rows,
        "solved": sum(r["stop_reason"] == "solved" for r in rows),
        "all_cells_win": sum(r["wins"] == r["cells"] for r in rows),
        "games_median": statistics.median(games),
        "games_min": min(games),
        "games_max": max(games),
        "seconds_mean": round(statistics.mean(r["seconds"] for r in rows), 1),
        "shortest_routes_min": min(r["shortest_routes"] for r in rows),
        "shortest_routes_max": max(r["shortest_routes"] for r in rows),
    }
    return summary


def reference():
    """The reference run (CS 370 maze, seed 1671) on whichever TensorFlow and Keras are installed.

    Run it under this project's environment and under the CS 370 one: with the same libraries
    on both sides, the time per move separates the code change from the library upgrade.
    """
    import keras
    import tensorflow

    config = load_config(ROOT / "configs" / "default.toml")
    maze = load_maze(config.maze_file)
    result, report = train_once(maze, config.training)
    return {
        "versions": {"tensorflow": tensorflow.__version__, "keras": keras.__version__},
        **row(config.training.seed, result, report),
        "ms_per_move": round(1000 * result.seconds / result.moves, 2),
    }


def summary():
    """docs/evidence/SUMMARY.md: every evidence file as one readable page."""

    def load(name):
        return json.loads((OUT / name).read_text(encoding="utf-8"))

    sweep, stall_run, gen, audit = (
        load("sweep.json"),
        load("stall.json"),
        load("generated.json"),
        load("audit.json"),
    )
    original = load("original.json")
    references = [load(path.name) for path in sorted(OUT.glob("reference-keras-*.json"))]
    lines = [
        "# Evidence summary",
        "",
        "Regenerate with `python scripts/evidence.py` and `scripts/time_original.py`.",
        "",
    ]
    lines += [
        "## Training speed on one computer (CS 370 maze, seed 1671)",
        "",
        "| Code | TensorFlow / Keras | Games | Moves | Seconds | ms per move | Cells won | Shortest routes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    o = original
    won = o["audit"].split(" of ")[0] if o["audit"] else "?"
    shortest = o["audit"].split(", ")[1].split(" of ")[0] if o["audit"] else "?"
    lines.append(
        f"| CS 370 notebook | {o['versions']['tensorflow']} / {o['versions']['keras']} | {o['games']} | "
        f"{o['moves']:,} | {o['train_seconds']} | {o['ms_per_move']} | {won} of 50 | {shortest} |"
    )
    for r in references:
        lines.append(
            f"| pirate_agent 2.0.0 | {r['versions']['tensorflow']} / {r['versions']['keras']} | "
            f"{r['games']} | {r['moves']:,} | {r['seconds']} | {r['ms_per_move']} | "
            f"{r['wins']} of {r['cells']} | {r['shortest_routes']} |"
        )
    lines += [
        "",
        "## Ten seeds on the CS 370 maze",
        "",
        "| Seed | Stop | Games | Moves | Seconds | Cells won | Shortest routes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sweep["runs"]:
        lines.append(
            f"| {r['seed']} | {r['stop_reason']} | {r['games']} | {r['moves']:,} | {r['seconds']} | "
            f"{r['wins']} of {r['cells']} | {r['shortest_routes']} |"
        )
    lines += [
        "",
        f"Solved {sweep['solved']} of {len(sweep['runs'])}; every cell won in {sweep['all_cells_win']}; "
        f"median {sweep['games_median']:g} games (range {sweep['games_min']} to {sweep['games_max']}); "
        f"mean {sweep['seconds_mean']} s per run.",
        "",
    ]
    lines += [
        "## Early stopping on a run that cannot learn",
        "",
        f"Learning rate {stall_run['learning_rate']:g}, patience {stall_run['patience']}: stopped as "
        f"{stall_run['stop_reason']} after {stall_run['games']} games ({stall_run['seconds']} s) instead of "
        f"up to {stall_run['max_epochs']:,}; best win rate {stall_run['best_win_rate']:.3f}.",
        "",
    ]
    lines += [
        "## A generated maze",
        "",
        "```",
        *gen["layout"],
        "```",
        "",
        f"{gen['open_cells']} open cells, seed {gen['maze']['seed']}: {gen['stop_reason']} after "
        f"{gen['games']} games in {gen['seconds']} s; {gen['wins']} of {gen['cells']} cells won, "
        f"{gen['shortest_routes']} by a shortest route.",
        "",
    ]
    lines += [
        "## Replay memory: nanoseconds per insert into a full memory",
        "",
        "| Capacity | list with del [0] | deque(maxlen) | Speedup |",
        "|---|---|---|---|",
    ]
    for r in load("replay.json"):
        lines.append(
            f"| {r['capacity']:,} | {r['list_ns_per_insert']:,} | {r['deque_ns_per_insert']} | "
            f"{r['speedup']}x |"
        )
    lines += [
        "",
        "## Dependency audit (pip-audit)",
        "",
        f"CS 370 pins ({', '.join(audit['original_pins'])}): {audit['original_advisory_count']} advisories.",
    ]
    for package, ids in audit["original_findings"].items():
        lines.append(f"- {package}: {len(ids)} ({', '.join(ids)})")
    lines += [
        "",
        f"Current pins ({', '.join(audit['current_pins'])}): {audit['current_advisory_count']} advisories.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return None


def stall():
    """A misconfigured run (a learning rate of 1e-9 cannot learn) that early stopping ends."""
    config = load_config(ROOT / "configs" / "default.toml")
    maze = load_maze(config.maze_file)
    training = dataclasses.replace(config.training, learning_rate=1e-9)
    result, report = train_once(maze, training)
    return {
        "learning_rate": training.learning_rate,
        "patience": training.patience,
        "max_epochs": training.max_epochs,
        **row(training.seed, result, report),
        "best_win_rate": result.best_win_rate,
        "games_not_played": training.max_epochs - result.epochs,
    }


def generated():
    config = load_config(ROOT / "configs" / "generated.toml")
    settings = config.generate
    maze = generate_maze(settings.rows, settings.cols, settings.wall_density, settings.seed)
    result, report = train_once(maze, config.training)
    return {
        "maze": dataclasses.asdict(settings),
        "layout": maze.to_text().splitlines(),
        "open_cells": len(maze.free_cells) + 1,
        **row(config.training.seed, result, report),
    }


def replay_timing():
    """Seconds per insert into a full memory: the starter's list with del [0] against a deque."""
    results = []
    for capacity in (512, 10_000, 100_000):
        inserts = 20_000
        timings = {}
        for name in ("list", "deque"):
            memory = list(range(capacity)) if name == "list" else deque(range(capacity), maxlen=capacity)
            started = time.perf_counter()
            for item in range(inserts):
                memory.append(item)
                if name == "list" and len(memory) > capacity:
                    del memory[0]  # what GameExperience.remember() did on every move
            timings[name] = (time.perf_counter() - started) / inserts
        results.append(
            {
                "capacity": capacity,
                "inserts": inserts,
                "list_ns_per_insert": round(timings["list"] * 1e9),
                "deque_ns_per_insert": round(timings["deque"] * 1e9),
                "speedup": round(timings["list"] / timings["deque"], 1),
            }
        )
    return results


def dependency_audit():
    """pip-audit on the CS 370 pins and on requirements.txt."""

    def audit(lines):
        with tempfile.TemporaryDirectory() as folder:
            req = Path(folder) / "requirements.txt"
            req.write_text("\n".join(lines) + "\n", encoding="utf-8")
            done = subprocess.run(  # noqa: S603 (a fixed command: this Python running pip-audit)
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "-r",
                    str(req),
                    "-f",
                    "json",
                    "--progress-spinner",
                    "off",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        report = json.loads(done.stdout)
        found = {}
        for dependency in report.get("dependencies", []):
            ids = sorted({v["id"] for v in dependency.get("vulns", [])})
            if ids:
                found[f"{dependency['name']} {dependency['version']}"] = ids
        return found

    current = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines() if "==" in line]
    old, new = audit(ORIGINAL_PINS), audit(current)
    return {
        "original_pins": ORIGINAL_PINS,
        "original_findings": old,
        "original_advisory_count": sum(len(ids) for ids in old.values()),
        "current_pins": current,
        "current_findings": new,
        "current_advisory_count": sum(len(ids) for ids in new.values()),
    }


JOBS = {
    "reference": reference,
    "sweep": seed_sweep,
    "stall": stall,
    "generated": generated,
    "replay": replay_timing,
    "audit": dependency_audit,
    "summary": summary,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=sorted(JOBS), help="run only these jobs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("pirate_agent").setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(JOBS):
        log.info("running %s", name)
        data = JOBS[name]()
        if data is None:
            continue  # the summary writes its own Markdown file
        file_name = f"{name}.json"
        if name == "reference":
            file_name = f"reference-keras-{data['versions']['keras']}.json"
        (OUT / file_name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
        log.info("wrote docs/evidence/%s", file_name)


if __name__ == "__main__":
    main()
