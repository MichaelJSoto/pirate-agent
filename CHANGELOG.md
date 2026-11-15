# Changelog

All notable changes to this project. Versions follow [Semantic Versioning](https://semver.org/).

## [2.1.0] - 2026-11-10

### Added
- A route polish stage. Once every cell wins, training continues and audits the routes after
  every game, keeps a snapshot of the version with the most shortest routes, and stops when
  `target_shortest_share` of the cells take a shortest route or after `polish_patience` games
  without improvement, restoring the best snapshot. The 2.0.0 evidence showed shortest routes
  ranging from 33 to 50 of 50 between seeds, because the CS 370 stopping rule asked only for wins.
- `audit.cell_policy()`: one batched prediction for every cell replaces a prediction per move
  during an audit, because an observation encodes only the pirate's cell.
- `DQNAgent.greedy_actions()`, `snapshot()` and `restore()`.
- Run logs record when every cell first won and the shortest-route count before and after polishing.

### Changed
- `target_shortest_share = 0.0` reproduces the 2.0.0 stopping rule.

## [2.0.0] - 2026-11-10

The CS 370 notebook rebuilt as a tested, configurable and secured Python package for CS 499.
Code review finding numbers (S1 to S12) refer to the Milestone One code review.

### Added
- `pirate_agent` package with one module per job: `maze`, `replay`, `agent`, `trainer`, `audit`,
  `config`, `runlog` and `cli`.
- Command line: `train`, `evaluate`, `play` and `generate`, with distinct exit codes.
- `.maze` files and a random solvable-maze generator, so training is no longer tied to one maze.
- TOML configuration with strict validation of every key, type and range.
- Run folders with the `.keras` model, a per-game history and a JSON run log (S6).
- Real early stopping with `patience` (S7) and an exploration schedule that restarts on every run (S1).
- 132 automated tests and a GitHub Actions workflow running Ruff, codespell, pytest and pip-audit (S12).
- README, contributing guide and pull request template.
- `pretrained/cs370-seed1671`, a reference run on the CS 370 maze.

### Changed
- The agent owns its network, target network and optimizer; no global state (S1).
- The network declares its input with `keras.Input` and is not compiled with an unused optimizer (S4).
- The replay memory is a `collections.deque` with `maxlen`, so inserts are O(1) (S5).
- One compiled TensorFlow function builds the Bellman targets and takes the gradient step.
- Settings are named in `TrainingConfig` instead of magic numbers (S2); unused imports removed (S3).
- `print` progress replaced by `logging` (S6).

### Fixed
- An invalid move now pays the -0.75 penalty the SNHU starter file describes (S9).
- A blocked treasure is reported as such, instead of crashing in `list.remove()` (S10).
- Cell values are whole numbers, so wall checks no longer compare floats (S8).

### Security
- TensorFlow 2.17 and Keras 3.5 replaced by TensorFlow 2.21.0 and Keras 3.15.1, pinned. pip-audit
  reports 19 known advisories against the CS 370 pins (18 in Keras 3.5.0, including safe-mode
  bypasses) and none against the new pins (S11).
- Model files are checked before Keras reads them: SHA-256 against the run log, archive contents,
  and an allowlist of layer classes; then a private copy loads with `safe_mode=True` (CWE-502).
- Maze, configuration and run-log files are size-capped and validated before use.
- No packages are installed at run time (S11).

## [1.0.0] - 2026-10-18

CS 370 Project Two: the deep Q-learning notebook as submitted, kept unchanged in `original/`.
