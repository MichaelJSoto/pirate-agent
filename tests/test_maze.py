"""Tests for maze validation, .maze files, the generator, and the game rules.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Pin down the rules of the game and every way a maze can be rejected, including regression
    tests for the two bugs the code review found in the SNHU starter file (findings S9, S10).

Decision:
    Each rejection test checks the wording of the error as well as its type, because a clear
    message is part of the security requirement: the user must be able to tell why a file was
    refused without reading a stack trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from pirate_agent.maze import (
    MAX_FILE_BYTES,
    PIRATE_MARK,
    Action,
    Maze,
    MazeEnv,
    MazeError,
    Mode,
    Rewards,
    Status,
    generate_maze,
    load_maze,
    parse_maze,
    shortest_moves,
)

# ---- the CS 370 maze ------------------------------------------------------------------------


def test_cs370_maze_matches_the_notebook(cs370_maze):
    assert cs370_maze.shape == (8, 8)
    assert cs370_maze.target == (7, 7)
    assert len(cs370_maze.free_cells) == 50  # "50 of 50 free cells" in the CS 370 audit
    assert max(shortest_moves(cs370_maze.grid, cs370_maze.target).values()) == 28


def test_maze_grid_is_read_only(cs370_maze):
    with pytest.raises(ValueError):
        cs370_maze.grid[0, 0] = 0


# ---- validation ------------------------------------------------------------------------------


def test_blocked_treasure_gets_a_clear_error():
    # Regression for finding S10: the starter crashed with "list.remove(x): x not in list".
    with pytest.raises(MazeError, match=r"treasure cell \(1, 1\) is a wall"):
        Maze.from_grid([[1, 1], [1, 0]])


def test_unreachable_free_cell_is_rejected():
    with pytest.raises(MazeError, match=r"1 free cell\(s\) cannot reach the treasure, for example \(0, 0\)"):
        Maze.from_grid([[1, 0, 1], [0, 1, 1], [1, 1, 1]])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[1, 1, 1]], "2 to 32 rows"),
        ([[1] * 33] * 2, "2 to 32 columns"),
        ([[1, 1], [1]], "same length"),
        ([[1, 2], [1, 1]], r"cell \(0, 1\) is 2"),
        ([[1, 0.5], [1, 1]], r"cell \(0, 1\) is 0.5"),
        ([[True, 1], [1, 1]], r"cell \(0, 0\) is True"),
        ([[0, 0], [0, 1]], "no free cell"),
    ],
)
def test_bad_layouts_are_rejected(rows, message):
    with pytest.raises(MazeError, match=message):
        Maze.from_grid(rows)


def test_nan_cell_is_rejected():
    with pytest.raises(MazeError, match="only 0"):
        Maze.from_grid(np.array([[1.0, np.nan], [1.0, 1.0]]))


def test_float_grid_from_the_notebook_is_accepted():
    maze = Maze.from_grid(np.array([[1.0, 0.0], [1.0, 1.0]]))
    assert maze.grid.dtype == np.int8


def test_treasure_outside_the_maze_is_rejected():
    with pytest.raises(MazeError, match="outside"):
        Maze.from_grid([[1, 1], [1, 1]], target=(5, 5))


# ---- .maze files ---------------------------------------------------------------------------------


def test_parse_with_comments_marker_and_bom():
    maze = parse_maze(b"\xef\xbb\xbf; a comment\n\n..T\n.#.\n")
    assert maze.target == (0, 2)
    assert maze.grid.tolist() == [[1, 1, 1], [1, 0, 1]]


def test_round_trip_through_text(cs370_maze):
    again = parse_maze(cs370_maze.to_text(comments=["round trip"]))
    assert np.array_equal(again.grid, cs370_maze.grid)
    assert again.target == cs370_maze.target


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("..\n.x\n", r"line 2, column 2: unexpected character 'x'"),
        ("...\n..\n", "line 2: the row has 2 cells but the first row has 3"),
        ("T.\n.T\n", "only one treasure"),
        ("; only comments\n", "no maze rows"),
        (".. .\n....\n", "unexpected character ' '"),
    ],
)
def test_bad_maze_text_is_rejected_with_its_location(text, message):
    with pytest.raises(MazeError, match=message):
        parse_maze(text, source="bad.maze")


def test_non_ascii_bytes_are_rejected():
    with pytest.raises(MazeError, match="non-ASCII"):
        parse_maze(b".\xe2\x80\xa7\n..\n")  # UTF-8 bytes for a hyphenation point


def test_oversized_file_is_refused_before_reading(tmp_path):
    path = tmp_path / "huge.maze"
    path.write_bytes(b"." * (MAX_FILE_BYTES + 1))
    with pytest.raises(MazeError, match="at most 8,192"):
        load_maze(path)


def test_wrong_suffix_and_missing_file(tmp_path):
    with pytest.raises(MazeError, match="must end in '.maze'"):
        load_maze(tmp_path / "maze.txt")
    with pytest.raises(MazeError, match="no such maze file"):
        load_maze(tmp_path / "missing.maze")


# ---- generator -------------------------------------------------------------------------------


def test_generator_repeats_for_a_seed_and_always_validates():
    first = generate_maze(8, 8, wall_density=0.3, seed=7)
    again = generate_maze(8, 8, wall_density=0.3, seed=7)
    assert np.array_equal(first.grid, again.grid)
    for seed in range(50):
        maze = generate_maze(8, 8, wall_density=0.3, seed=seed)
        reach = shortest_moves(maze.grid, maze.target)
        assert all(cell in reach for cell in maze.free_cells)  # every free cell reaches the treasure
        assert len(maze.free_cells) + 1 >= 32  # at least half the cells stay open


def test_generator_rejects_bad_settings():
    with pytest.raises(MazeError, match="rows must be"):
        generate_maze(1, 8)
    with pytest.raises(MazeError, match="wall_density"):
        generate_maze(8, 8, wall_density=0.9)
    with pytest.raises(MazeError, match="lower the wall density"):
        generate_maze(8, 8, wall_density=0.6, min_free_share=1.0)


# ---- game rules --------------------------------------------------------------------------------


def test_observation_marks_the_pirate(corridor):
    env = MazeEnv(corridor)
    obs = env.reset((0, 0))
    assert obs.shape == (1, 6)
    assert obs.dtype == np.float32
    assert obs[0].tolist() == [PIRATE_MARK, 1.0, 1.0, 1.0, 0.0, 1.0]


def test_valid_actions_respect_walls_and_edges(corridor):
    env = MazeEnv(corridor)
    assert set(env.valid_actions((0, 0))) == {Action.RIGHT, Action.DOWN}
    assert set(env.valid_actions((0, 1))) == {Action.LEFT, Action.RIGHT}  # a wall below


def test_step_rewards_and_win(corridor):
    env = MazeEnv(corridor)
    env.reset((0, 0))
    _, reward, status = env.step(Action.RIGHT)
    assert (reward, status, env.mode) == (-0.04, Status.PLAYING, Mode.VALID)
    _, reward, _ = env.step(Action.LEFT)
    assert reward == -0.25  # back onto a visited cell
    env.step(Action.RIGHT)
    env.step(Action.RIGHT)
    _, reward, status = env.step(Action.DOWN)
    assert (reward, status) == (1.0, Status.WIN)


def test_invalid_move_pays_its_penalty(corridor):
    # Regression for finding S9: the starter returned -0.25 here and left the mode unchanged.
    env = MazeEnv(corridor)
    env.reset((0, 1))
    _, reward, status = env.step(Action.DOWN)  # into the wall
    assert env.pirate == (0, 1)
    assert env.mode is Mode.INVALID
    assert reward == -0.75
    assert status is Status.PLAYING


def test_game_is_lost_when_the_budget_runs_out(corridor):
    env = MazeEnv(corridor, Rewards())
    env.reset((0, 1))
    statuses = [env.step(Action.UP)[2] for _ in range(5)]  # off the top edge, five times
    assert env.total_reward == pytest.approx(-3.75)
    assert statuses[-1] is Status.LOSE  # -3.75 is below the budget of -0.5 * 6 = -3.0


def test_reset_rejects_walls_and_the_treasure(corridor):
    env = MazeEnv(corridor)
    with pytest.raises(MazeError, match="cannot start"):
        env.reset((1, 1))
    with pytest.raises(MazeError, match="cannot start"):
        env.reset((1, 2))


def test_unknown_action_is_rejected(corridor):
    env = MazeEnv(corridor)
    with pytest.raises(ValueError):
        env.step(4)
