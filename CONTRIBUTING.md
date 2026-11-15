# Contributing to Pirate Agent

Thank you for helping. This guide covers how to set up the project, how a change moves from a
branch to `main`, and the standards every pull request is reviewed against.

## Set up

```
python -m venv .venv
.venv\Scripts\activate                 # Windows
source .venv/bin/activate              # macOS and Linux
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps   # makes the pirate-agent command available
python -m pytest
```

The full suite takes about ten seconds on a laptop CPU. Tests marked `tf` load TensorFlow; the
rest do not, so `python -m pytest -m "not tf"` gives a two-second check while you work.

## How a change reaches `main`

1. Branch from `main` with a short, descriptive name: `feature/<topic>` or `fix/<topic>`.
2. Keep each branch to one idea. A reviewer should be able to read the whole diff in one sitting
   (a few hundred lines is a good limit).
3. Before you push, run the same checks CI runs:

   ```
   ruff format .
   ruff check .
   codespell
   python -m pytest
   pip-audit --strict -r requirements.txt
   ```

4. Open a pull request. The template asks for the problem, the decision you made and the
   alternatives you rejected, how you tested it, and what it changes for security. That
   write-up is the record future maintainers will read, so give the reasons, not just the
   result.
5. CI must pass, and one review must approve, before the branch is merged.

## Code standards

**Every file starts with a header** that names the programmer, date and version, and states the
file's intent and the main decision behind it. Python files use the module docstring:

```python
"""One line on what this module is for.

Programmer: Your Name
Date: Month Day, Year
Version: 2.0.0
Course: (if it applies)

Intent:
    What problem this file solves, in two or three sentences.

Decision:
    The design choice a reader would otherwise question, and why it was made.
"""
```

TOML, YAML and requirements files use the same fields in `#` comments.

**Comments explain intent and decisions**, not what a line plainly does. "Checked before the
revisit rule, or the -0.75 penalty could never be paid" is useful; "check if invalid" is not.

**Tests come with every behavior change.** A bug fix starts with a test that fails on the old
code. Use the fixtures in `tests/conftest.py` so expected numbers stay traceable to the
CS 370 notebook.

**Errors are for people.** Raise `MazeError`, `ConfigError`, `RunLogError` or `ModelFileError`
with one sentence that says what was wrong and how to fix it. The command line turns them
into exit codes; nobody should need a stack trace to understand a bad input.

## Security checklist for reviewers

- Anything read from disk (a maze, a configuration, a run log, a model) is untrusted. It is
  size-checked before it is read and validated before it is used.
- Models load only through `pirate_agent.agent.load_network()`, which checks the SHA-256, the
  archive contents and the layer classes before Keras sees the file. Never call
  `keras.models.load_model()` directly, and never load `.h5` or pickle files.
- No `pickle`, `eval`, `exec` or `subprocess` in the package. Ruff's security rules (`S`) flag
  them in CI.
- A new or upgraded dependency is pinned to an exact version in `requirements.txt` and must pass
  `pip-audit --strict`.
- Output and logs never include absolute paths, user names or other personal data.

## Reviewing

Review the change, not the person. Be specific ("line 42 reads the file before checking its
size"), say whether a comment blocks the merge or is a suggestion, and approve once the blocking
points are fixed. Authors: reply to every comment, even if only to say it is done.
