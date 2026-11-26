"""Tests for maze validation, .maze files, the generator, and the game rules.

Programmer: Michael Soto
Date: November 22, 2026
Version: 2.1.2
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
    """The shipped maze has the notebook's shape, treasure, 50 free cells and 28-move farthest cell."""
    assert cs370_maze.shape == (8, 8)
    assert cs370_maze.target == (7, 7)
    assert len(cs370_maze.free_cells) == 50  # "50 of 50 free cells" in the CS 370 audit
    assert max(shortest_moves(cs370_maze.grid, cs370_maze.target).values()) == 28


def test_maze_grid_is_read_only(cs370_maze):
    """A validated grid cannot be edited afterward, so no later code can undo its validation."""
    with pytest.raises(ValueError):
        cs370_maze.grid[0, 0] = 0


# ---- validation ------------------------------------------------------------------------------


def test_blocked_treasure_gets_a_clear_error():
    """A maze whose treasure is a wall is refused with a message that names the cell (finding S10).

    The starter file crashed here with "list.remove(x): x not in list" before its own check
    for a blocked treasure could run.
    """
    with pytest.raises(MazeError, match=r"treasure cell \(1, 1\) is a wall"):
        Maze.from_grid([[1, 1], [1, 0]])


def test_unreachable_free_cell_is_rejected():
    """A free cell with no path to the treasure is refused and named, since no policy could win from it."""
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
    """Each malformed grid is refused with a message naming the problem and, where there is one, the cell.

    Only 0 and 1 are accepted, not 0.5 or True, because whole-number cells are what make the
    wall checks exact (finding S8).
    """
    with pytest.raises(MazeError, match=message):
        Maze.from_grid(rows)


def test_nan_cell_is_rejected():
    """A NaN cell is refused; the starter's float checks (finding S8) would see neither wall nor free."""
    with pytest.raises(MazeError, match="only 0"):
        Maze.from_grid(np.array([[1.0, np.nan], [1.0, 1.0]]))


def test_float_grid_from_the_notebook_is_accepted():
    """The notebook's float grid of 1.0 and 0.0 still loads, stored as whole numbers (finding S8)."""
    maze = Maze.from_grid(np.array([[1.0, 0.0], [1.0, 1.0]]))
    assert maze.grid.dtype == np.int8


def test_treasure_outside_the_maze_is_rejected():
    """A treasure outside the grid is refused when the maze is built, not with an IndexError later."""
    with pytest.raises(MazeError, match="outside"):
        Maze.from_grid([[1, 1], [1, 1]], target=(5, 5))


# ---- .maze files ---------------------------------------------------------------------------------


def test_parse_with_comments_marker_and_bom():
    """A .maze file may open with a UTF-8 byte order mark and hold ';' comments, blank lines and a 'T'."""
    maze = parse_maze(b"\xef\xbb\xbf; a comment\n\n..T\n.#.\n")
    assert maze.target == (0, 2)
    assert maze.grid.tolist() == [[1, 1, 1], [1, 0, 1]]


def test_round_trip_through_text(cs370_maze):
    """Writing the CS 370 maze as text and parsing it back gives the same grid and treasure."""
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
    """Bad .maze text is refused with the line, and the column where it applies, that a user must fix."""
    with pytest.raises(MazeError, match=message):
        parse_maze(text, source="bad.maze")


def test_non_ascii_bytes_are_rejected():
    """Bytes outside ASCII are refused, so a look-alike character cannot pass for a maze cell."""
    with pytest.raises(MazeError, match="non-ASCII"):
        parse_maze(b".\xe2\x80\xa7\n..\n")  # UTF-8 bytes for a hyphenation point


def test_oversized_file_is_refused_before_reading(tmp_path):
    """A .maze file over 8,192 bytes is refused from its size on disk, before any of it is read."""
    path = tmp_path / "huge.maze"
    path.write_bytes(b"." * (MAX_FILE_BYTES + 1))
    with pytest.raises(MazeError, match="at most 8,192"):
        load_maze(path)


def test_wrong_suffix_and_missing_file(tmp_path):
    """load_maze() refuses a name without the .maze suffix and reports a missing file plainly."""
    with pytest.raises(MazeError, match="must end in '.maze'"):
        load_maze(tmp_path / "maze.txt")
    with pytest.raises(MazeError, match="no such maze file"):
        load_maze(tmp_path / "missing.maze")


# ---- generator -------------------------------------------------------------------------------


def test_generator_repeats_for_a_seed_and_always_validates():
    """A seed always gives the same maze, and 50 seeds each give a solvable maze at least half open."""
    first = generate_maze(8, 8, wall_density=0.3, seed=7)
    again = generate_maze(8, 8, wall_density=0.3, seed=7)
    assert np.array_equal(first.grid, again.grid)
    for seed in range(50):
        maze = generate_maze(8, 8, wall_density=0.3, seed=seed)
        reach = shortest_moves(maze.grid, maze.target)
        assert all(cell in reach for cell in maze.free_cells)  # every free cell reaches the treasure
        assert len(maze.free_cells) + 1 >= 32  # at least half the cells stay open


def test_generator_rejects_bad_settings():
    """The generator refuses too few rows, too many walls and settings it cannot meet, naming the fix."""
    with pytest.raises(MazeError, match="rows must be"):
        generate_maze(1, 8)
    with pytest.raises(MazeError, match="wall_density"):
        generate_maze(8, 8, wall_density=0.9)
    with pytest.raises(MazeError, match="lower the wall density"):
        generate_maze(8, 8, wall_density=0.6, min_free_share=1.0)


# ---- game rules --------------------------------------------------------------------------------


def test_observation_marks_the_pirate(corridor):
    """The network's input: one float32 row, 1.0 free, 0.0 wall and PIRATE_MARK on the pirate."""
    env = MazeEnv(corridor)
    obs = env.reset((0, 0))
    assert obs.shape == (1, 6)
    assert obs.dtype == np.float32
    assert obs[0].tolist() == [PIRATE_MARK, 1.0, 1.0, 1.0, 0.0, 1.0]


def test_valid_actions_respect_walls_and_edges(corridor):
    """valid_actions() leaves out moves off the edge and into a wall, so exploration never wastes a move."""
    env = MazeEnv(corridor)
    assert set(env.valid_actions((0, 0))) == {Action.RIGHT, Action.DOWN}
    assert set(env.valid_actions((0, 1))) == {Action.LEFT, Action.RIGHT}  # a wall below


def test_step_rewards_and_win(corridor):
    """The starter's reward scale: a move costs 0.04, a revisit 0.25, and the treasure pays 1.0 and wins."""
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
    """A move into a wall leaves the pirate in place, records Mode.INVALID and costs 0.75 (finding S9).

    The starter file set `mode` where it meant `nmode`, so it never recorded an invalid move
    and charged 0.25 instead.
    """
    env = MazeEnv(corridor)
    env.reset((0, 1))
    _, reward, status = env.step(Action.DOWN)  # into the wall
    assert env.pirate == (0, 1)
    assert env.mode is Mode.INVALID
    assert reward == -0.75
    assert status is Status.PLAYING


def test_game_is_lost_when_the_budget_runs_out(corridor):
    """Five invalid moves total -3.75, below the budget of -0.5 per cell (-3.0 here), so the game is lost.

    The budget is what stops a wandering pirate from playing one game forever.
    """
    env = MazeEnv(corridor, Rewards())
    env.reset((0, 1))
    statuses = [env.step(Action.UP)[2] for _ in range(5)]  # off the top edge, five times
    assert env.total_reward == pytest.approx(-3.75)
    assert statuses[-1] is Status.LOSE


def test_reset_rejects_walls_and_the_treasure(corridor):
    """A game cannot start on a wall or on the treasure, only on a free cell the pirate can play from."""
    env = MazeEnv(corridor)
    with pytest.raises(MazeError, match="cannot start"):
        env.reset((1, 1))
    with pytest.raises(MazeError, match="cannot start"):
        env.reset((1, 2))


def test_unknown_action_is_rejected(corridor):
    """An action number outside 0 to 3 raises ValueError instead of moving the pirate somewhere undefined."""
    env = MazeEnv(corridor)
    with pytest.raises(ValueError):
        env.step(4)
