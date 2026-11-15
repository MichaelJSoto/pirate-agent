"""Draws the README figures from the reference run: the training curve and the learned policy.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.1.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Show a reader what the agent learned without asking them to run anything: how the win
    rate rose game by game, and which move the trained pirate picks in every free cell.

Decision:
    Each figure is drawn twice, for GitHub's light and dark themes, from the same colors the
    CS 370 notebook used (blue for the agent, orange for a warning), which pass a color
    vision deficiency check on both surfaces. The model is loaded through the checked loader,
    like any other model file. PNG metadata names no software.

Run:  python scripts/figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from pirate_agent.audit import audit_policy  # noqa: E402
from pirate_agent.maze import MOVES, MazeEnv  # noqa: E402
from pirate_agent.runlog import maze_from_log, read_run_log  # noqa: E402

RUN = ROOT / "pretrained" / "cs370-seed1671"
OUT = ROOT / "docs" / "images"
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "wall": "#52514e", "series": "#2a78d6", "flag": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#34332f",
             "wall": "#8a8985", "series": "#3987e5", "flag": "#d95926"},
}  # fmt: skip


def save(fig, name: str) -> None:
    fig.savefig(
        OUT / name, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), metadata={"Software": None}
    )
    plt.close(fig)


def training_curve(theme: dict[str, str], record: dict) -> plt.Figure:
    with (RUN / "history.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    games = np.array([int(r["epoch"]) for r in rows]) + 1
    rate = np.array([float(r["win_rate"]) for r in rows]) * 100
    window = record["training"]["win_window"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=theme["surface"])
    ax.set_facecolor(theme["surface"])
    ax.plot(games[window - 1 :], rate[window - 1 :], color=theme["series"], linewidth=2)
    ax.set_xlim(0, games[-1] + 5)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["muted"])
    ax.set_xlabel("Training game", color=theme["muted"], fontsize=9)
    seed = record["training"]["seed"]
    ax.set_title(
        f"Win rate over the last {window} games, seed {seed}", loc="left", color=theme["ink"], fontsize=11
    )
    last = games[-1]
    solved = record["result"].get("solved_at") or last
    label = f"Every cell wins at game {solved}."
    if last > solved:
        label += f"\nShaded: polishing routes until game {last}."
        ax.axvspan(solved, last, color=theme["grid"], linewidth=0)  # the polish stage
    # The top-left corner stays empty: the win rate has no value until the window fills.
    ax.text(window, 92, label, color=theme["ink"], fontsize=9, va="top")
    ax.text(window, 4, f"no rate until the first {window} games are played", color=theme["muted"], fontsize=8)
    return fig


def policy_map(theme: dict[str, str], record: dict) -> plt.Figure:
    from pirate_agent.agent import DQNAgent
    from pirate_agent.config import TrainingConfig

    maze = maze_from_log(record)
    agent = DQNAgent.from_file(RUN / "model.keras", record["model"]["sha256"], TrainingConfig(), maze.size)
    report = audit_policy(maze, agent.greedy_action)
    env = MazeEnv(maze)
    arrows = []
    for cell in maze.free_cells:
        d_row, d_col = MOVES[agent.greedy_action(env.reset(cell))]
        arrows.append((cell[1], cell[0], d_col, d_row))
    x, y, u, v = np.array(arrows, dtype=float).T
    fig, ax = plt.subplots(figsize=(4.6, 4.9), facecolor=theme["surface"])
    ax.imshow(maze.grid, cmap=ListedColormap([theme["wall"], theme["surface"]]), vmin=0, vmax=1)
    ax.quiver(
        x,
        y,
        u,
        v,
        color=theme["series"],
        angles="xy",
        scale_units="xy",
        scale=2.4,
        pivot="middle",
        width=0.012,
    )
    flagged = [r.cell for r in report.results if not r.optimal]
    for row, col in flagged:
        ax.add_patch(
            Rectangle((col - 0.4, row - 0.4), 0.8, 0.8, fill=False, edgecolor=theme["flag"], linewidth=2.2)
        )
    ax.text(
        maze.target[1], maze.target[0], "T", ha="center", va="center", color=theme["ink"], fontweight="bold"
    )
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color=theme["surface"], linewidth=2)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme["wall"])
    ax.set_title("Trained move from every free cell", loc="left", color=theme["ink"], fontsize=11)
    if flagged:
        handle = Line2D(
            [], [], color=theme["flag"], linewidth=2, label=f"longer than shortest ({len(flagged)} cells)"
        )
        legend = ax.legend(
            handles=[handle], loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9
        )
        for text in legend.get_texts():
            text.set_color(theme["ink"])
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = read_run_log(RUN / "run.json")
    for name, theme in THEMES.items():
        save(training_curve(theme, record), f"training-curve-{name}.png")
        save(policy_map(theme, record), f"policy-{name}.png")
    print(json.dumps(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
