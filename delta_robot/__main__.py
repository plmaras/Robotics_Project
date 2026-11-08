"""Unified command-line entry point for robot simulations and tools."""

from __future__ import annotations

import argparse
import importlib
import sys


COMMANDS = {
    "benchmark": "scripts.run_demo",
    "cad-scene": "cad.run_scene",
    "pick": "scripts.run_pick_place",
    "verify": "scripts.verify_mujoco",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="python -m delta_robot",
        description="Delta robot benchmarks, CAD scene tools, and pick-and-place simulation.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command = argv[0]
    if command not in COMMANDS:
        parser.error(f"invalid choice: {command!r} (choose from {', '.join(sorted(COMMANDS))})")
    module = importlib.import_module(COMMANDS[command])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
