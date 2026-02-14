#!/usr/bin/env python3
"""Example: run a single Codex invocation and print the result.

Usage:
    python examples/run_once.py /path/to/repo "Add type hints to utils.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codex_manager.codex_cli import CodexRunner


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: run_once.py <repo_path> <prompt>")
        sys.exit(1)

    repo_path = sys.argv[1]
    prompt = sys.argv[2]

    runner = CodexRunner(timeout=300)
    print(f"Running Codex on {repo_path!r} …")
    result = runner.run(repo_path, prompt, full_auto=False)

    print(f"\nSuccess:       {result.success}")
    print(f"Exit code:     {result.exit_code}")
    print(f"Duration:      {result.duration_seconds:.1f}s")
    print(f"Events:        {len(result.events)}")
    print(f"File changes:  {len(result.file_changes)}")
    print(f"Errors:        {result.errors}")
    print(f"\nFinal message:\n{result.final_message[:500]}")
    print(f"\nUsage: {result.usage.model_dump_json(indent=2)}")


if __name__ == "__main__":
    main()
