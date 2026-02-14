#!/usr/bin/env python3
"""Example: run the full improvement loop.

Usage:
    python examples/run_loop.py /path/to/repo "Improve test coverage" --rounds 5 --mode dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codex_manager.codex_cli import CodexRunner
from codex_manager.eval_tools import RepoEvaluator
from codex_manager.loop import ImprovementLoop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Codex improvement loop.")
    parser.add_argument("repo", help="Path to the target repo")
    parser.add_argument("goal", help="Improvement goal in natural language")
    parser.add_argument("--rounds", type=int, default=5, help="Max rounds (default 5)")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run | apply (default: dry-run)",
    )
    parser.add_argument("--test-cmd", default=None, help="Custom test command")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    test_cmd = args.test_cmd.split() if args.test_cmd else None

    loop = ImprovementLoop(
        repo_path=args.repo,
        goal=args.goal,
        mode=args.mode,
        max_rounds=args.rounds,
        runner=CodexRunner(),
        evaluator=RepoEvaluator(test_cmd=test_cmd),
    )

    state = loop.run()

    print(f"\nDone! {len(state.rounds)} rounds, stop reason: {state.stop_reason}")
    print(f"State saved to: {state.state_path()}")


if __name__ == "__main__":
    main()
