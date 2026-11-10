"""Lets the package run as `python -m pirate_agent`.

Programmer: Michael Soto
Date: November 10, 2026
Version: 2.0.0
Course: CS 499 Computer Science Capstone, Enhancement One (software design and engineering)

Intent:
    Run the command-line interface without installing the package's console script.

Decision:
    The exit code from cli.main() is passed straight to the shell, so scripts and CI can
    tell a bad input (3) from a refused model (4) or a failed audit (5).
"""

from .cli import main

raise SystemExit(main())
