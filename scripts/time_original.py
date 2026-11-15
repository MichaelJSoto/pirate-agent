"""Runs the original CS 370 notebook's training code unchanged and records how long it takes.

Programmer: Michael Soto
Date: November 15, 2026
Version: 2.1.1
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Give the enhanced package a fair baseline: the CS 370 code, trained with the same seed on
    the same computer, instead of comparing a laptop with Codio's virtual machine.

Decision:
    The notebook's code cells are executed in order from original/, with only the Jupyter
    magics (!pip, %matplotlib, %config) removed, and the run stops after the route audit.
    Run it with the CS 370 environment (TensorFlow 2.17, Keras 3.5), not this project's, so
    the original code meets the libraries it was written for. Writes docs/evidence/original.json.

Run:  <CS 370 venv>/python scripts/time_original.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "original"
os.environ.setdefault("MPLBACKEND", "Agg")  # draw figures off screen
LAST_CELL = "audit_rows = PolicyAudit(model, maze).rows()"  # the route audit cell ends the run


def code_cells():
    """Yields each code cell of the original notebook, minus shell (!) and magic (%) lines."""
    notebook = json.loads((ORIGINAL / "Soto_Michael_ProjectTwo.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        lines = [line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))]
        yield "\n".join(lines), source


def main() -> None:
    """Runs the notebook's cells up to the route audit, times the qtrain() cell, and writes original.json."""
    sys.path.insert(0, str(ORIGINAL))  # so `from TreasureMaze import TreasureMaze` finds the starter files
    os.chdir(ORIGINAL)
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    printed = io.StringIO()
    train_seconds = None
    with contextlib.redirect_stdout(printed):
        for code, source in code_cells():
            started = time.perf_counter()
            exec(compile(code, "<notebook cell>", "exec"), namespace)  # noqa: S102 (our own notebook)
            if "qtrain(model, maze" in source:
                train_seconds = time.perf_counter() - started
            if LAST_CELL in source:
                break
    output = printed.getvalue()
    reached = re.search(r"Reached 100% win rate at epoch (\d+)", output)
    explorer = namespace["explorer"]
    audit = [line for line in output.splitlines() if "free cells reach the treasure" in line]
    import keras
    import numpy
    import tensorflow

    result = {
        "code": "original/Soto_Michael_ProjectTwo.ipynb, executed cell by cell",
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "tensorflow": tensorflow.__version__,
            "keras": keras.__version__,
        },
        "seed": namespace.get("SEED"),
        "solved_at_epoch": int(reached.group(1)) if reached else None,
        "games": len(explorer.tallies),
        "moves": int(sum(sum(t) for t in explorer.tallies)),
        "train_seconds": round(train_seconds, 1) if train_seconds else None,
        "audit": audit[0] if audit else None,
    }
    if result["moves"] and result["train_seconds"]:
        result["ms_per_move"] = round(1000 * result["train_seconds"] / result["moves"], 2)
    out = ROOT / "docs" / "evidence" / "original.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
