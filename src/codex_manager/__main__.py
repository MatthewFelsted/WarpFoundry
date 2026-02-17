"""CLI entrypoint for WarpFoundry."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from codex_manager.preflight import PreflightReport, build_preflight_report, parse_agents


def _load_dotenv() -> None:
    """Load .env from cwd, its parent, or package root so it's found regardless of cwd."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Package root = directory containing pyproject.toml / .env (parent of src/)
    _this_file = Path(__file__).resolve()
    _package_root = (
        _this_file.parent.parent.parent
    )  # src/codex_manager/__main__.py -> codex_manager/
    for dir_ in (Path.cwd(), Path.cwd().parent, _package_root, _package_root.parent):
        env_file = dir_ / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            return
    load_dotenv()


_load_dotenv()

STRATEGIC_PRODUCT_MAXIMIZATION_GOAL = (
    "Operate in STRATEGIC PRODUCT MAXIMIZATION MODE for this repository. "
    "Rank 3-5 opportunities by impact, effort, and risk; pick the single "
    "highest-leverage change; then implement it end-to-end with production-"
    "quality code, tests, and documentation updates."
)

DISCOVER_CHAIN_FOCUS = (
    "If this is Discover Chain, prioritize improvements that raise discovery "
    "quality, user trust, and repeat usage."
)

STRATEGIC_FOCUS_GUIDANCE: dict[str, str] = {
    "docs": "Improve onboarding clarity and reduce ambiguity in day-one usage.",
    "dx": "Reduce contributor friction and make changes easier to ship safely.",
    "growth": "Increase activation, retention, and repeat usage momentum.",
    "performance": "Speed up critical user workflows and cut avoidable latency.",
    "reliability": "Eliminate failure paths, regressions, and fragile behavior first.",
    "security": "Harden trust boundaries and reduce data or execution risk.",
    "testing": "Increase confidence with targeted regression and edge-case coverage.",
    "ux": "Make key journeys clearer, faster, and easier to complete correctly.",
}


def _extract_requested_followups(repo: Path, *, limit: int = 3) -> list[str]:
    """Return unresolved requested follow-up items when present."""
    todo_path = repo / "docs" / "REQUESTED_FEATURES_TODO.md"
    if not todo_path.is_file():
        return []
    try:
        lines = todo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    in_section = False
    items: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not in_section:
            if line.lower().startswith("## remaining follow-ups"):
                in_section = True
            continue

        if line.startswith("## "):
            break
        if not line.startswith("- "):
            continue

        item = line[2:].strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered.startswith("none at the moment") or lowered == "none":
            return []

        items.append(item)
        if len(items) >= limit:
            break

    return items


def _normalize_strategic_focus_values(focus_values: list[str] | None) -> list[str]:
    """Deduplicate and normalize strategic focus options while preserving order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for value in focus_values or []:
        key = str(value or "").strip().lower()
        if not key or key in seen:
            continue
        if key not in STRATEGIC_FOCUS_GUIDANCE:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _build_strategic_goal(
    *,
    repo: Path,
    goal_extra: str = "",
    focus_values: list[str] | None = None,
) -> str:
    """Build a repository-aware strategic goal prompt."""
    sections: list[str] = [STRATEGIC_PRODUCT_MAXIMIZATION_GOAL]
    repo_name = repo.name.lower()
    if "discover" in repo_name and "chain" in repo_name:
        sections.append(DISCOVER_CHAIN_FOCUS)

    normalized_focus = _normalize_strategic_focus_values(focus_values)
    if normalized_focus:
        focus_lines = [
            f"- {focus}: {STRATEGIC_FOCUS_GUIDANCE[focus]}" for focus in normalized_focus
        ]
        sections.append("Strategic focus priorities:\n" + "\n".join(focus_lines))

    context_lines: list[str] = []
    todo_wishlist = repo / ".codex_manager" / "owner" / "TODO_WISHLIST.md"
    if todo_wishlist.is_file():
        context_lines.append(
            "Use .codex_manager/owner/TODO_WISHLIST.md as the primary execution backlog. "
            "If items are stale or vague, refine before implementing."
        )

    feature_dreams = repo / ".codex_manager" / "owner" / "FEATURE_DREAMS.md"
    if feature_dreams.is_file():
        context_lines.append(
            "Use .codex_manager/owner/FEATURE_DREAMS.md as secondary input when TODO "
            "backlog quality is insufficient."
        )

    requested_followups = _extract_requested_followups(repo)
    if requested_followups:
        context_lines.append(
            "Review unresolved follow-ups from docs/REQUESTED_FEATURES_TODO.md when they "
            "align with the top-ranked opportunity."
        )

    if context_lines:
        sections.append("Repository-specific context:\n" + "\n".join(f"- {line}" for line in context_lines))
    if requested_followups:
        sections.append(
            "Requested follow-ups to consider:\n"
            + "\n".join(f"- {item}" for item in requested_followups)
        )

    extra = goal_extra.strip()
    if extra:
        sections.append(f"Additional priority context:\n{extra}")

    return "\n\n".join(sections)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser for all supported modes."""
    p = argparse.ArgumentParser(
        prog="warpfoundry",
        description="WarpFoundry - orchestrate AI coding agents for iterative repo improvement.",
    )
    # -- Sub-commands ---------------------------------------------------------
    sub = p.add_subparsers(dest="command")

    # GUI sub-command
    gui_p = sub.add_parser("gui", help="Launch the web GUI.")
    gui_p.add_argument("--port", type=int, default=5088, help="Port (default 5088)")
    gui_p.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    gui_p.add_argument(
        "--pipeline-resume-checkpoint",
        type=str,
        default="",
        help="Optional pipeline checkpoint path to auto-resume after server restart.",
    )

    # Pipeline sub-command
    pipe_p = sub.add_parser(
        "pipeline",
        help="Run the autonomous improvement pipeline.",
    )
    pipe_p.add_argument(
        "--repo",
        type=str,
        default="",
        help="Path to the target git repository.",
    )
    pipe_p.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help=(
            "Resume from a pipeline checkpoint JSON file produced by self-improvement "
            "restart handoff."
        ),
    )
    pipe_p.add_argument(
        "--resume-state",
        action="store_true",
        help=(
            "Resume from <repo>/.codex_manager/state/pipeline_resume.json "
            "(or current directory when --repo is omitted)."
        ),
    )
    pipe_p.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run: read-only; apply: agents may edit files (default: dry-run).",
    )
    pipe_p.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Maximum pipeline cycles (default: 3).",
    )
    pipe_p.add_argument(
        "--science",
        action="store_true",
        help="Enable scientist mode (theorize -> experiment -> analyze).",
    )
    pipe_p.add_argument(
        "--brain",
        action="store_true",
        help="Enable the AI brain layer for prompt refinement.",
    )
    pipe_p.add_argument(
        "--brain-model",
        type=str,
        default="gpt-5.2",
        help="Model for the brain layer (default: gpt-5.2).",
    )
    pipe_p.add_argument(
        "--agent",
        choices=["codex", "claude_code"],
        default="codex",
        help="Agent to use for execution (default: codex).",
    )
    pipe_p.add_argument(
        "--codex-bin",
        type=str,
        default="codex",
        help="Path to the codex CLI binary (default: codex).",
    )
    pipe_p.add_argument(
        "--claude-bin",
        type=str,
        default="claude",
        help="Path to the Claude Code CLI binary (default: claude).",
    )
    pipe_p.add_argument(
        "--test-cmd",
        type=str,
        default=None,
        help='Test command (default: "python -m pytest -q").',
    )
    pipe_p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Inactivity timeout per phase in seconds; 0 disables (default: 600).",
    )
    pipe_p.add_argument(
        "--max-tokens",
        type=int,
        default=5_000_000,
        help="Maximum total token budget (default: 5,000,000).",
    )
    pipe_p.add_argument(
        "--max-time",
        type=int,
        default=240,
        help="Maximum time in minutes (default: 240).",
    )
    pipe_p.add_argument(
        "--webhook-url",
        action="append",
        default=None,
        help=(
            "Run-completion webhook URL (Slack/Discord/generic HTTP). "
            "Repeat for multiple destinations."
        ),
    )
    pipe_p.add_argument(
        "--webhook-timeout",
        type=int,
        default=None,
        help="Webhook request timeout in seconds (default: 10).",
    )
    pipe_p.add_argument(
        "--local-only",
        action="store_true",
        help="Force all AI calls through local Ollama (no cloud APIs).",
    )
    pipe_p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip setup diagnostics before running (not recommended).",
    )

    # GitHub Actions workflow generator sub-command
    gha_p = sub.add_parser(
        "github-actions",
        help="Generate a GitHub Actions workflow for automated pipeline runs.",
    )
    gha_p.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the target git repository.",
    )
    gha_p.add_argument(
        "--workflow-file",
        type=str,
        default="warpfoundry-pipeline.yml",
        help="Workflow filename under .github/workflows (default: warpfoundry-pipeline.yml).",
    )
    gha_p.add_argument(
        "--branch",
        action="append",
        default=[],
        help=(
            "Branch to trigger on; repeat for multiple branches. "
            "Defaults to current repo branch (or main)."
        ),
    )
    gha_p.add_argument(
        "--python-version",
        type=str,
        default="3.11",
        help="Python version for actions/setup-python (default: 3.11).",
    )
    gha_p.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="Pipeline mode inside CI workflow (default: dry-run).",
    )
    gha_p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Pipeline cycles in CI workflow (default: 1).",
    )
    gha_p.add_argument(
        "--max-time",
        type=int,
        default=120,
        help="Pipeline max-time in minutes for CI workflow (default: 120).",
    )
    gha_p.add_argument(
        "--agent",
        choices=["codex", "claude_code"],
        default="codex",
        help="Pipeline agent in CI workflow (default: codex).",
    )
    gha_p.add_argument(
        "--artifact-prefix",
        type=str,
        default="warpfoundry-pipeline",
        help="Prefix for uploaded artifact names (default: warpfoundry-pipeline).",
    )
    gha_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the workflow file when it already exists.",
    )

    # Strategic loop sub-command
    strategic_p = sub.add_parser(
        "strategic",
        help="Run the CLI loop with a built-in Strategic Product Maximization goal.",
    )
    strategic_p.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to the target git repository.",
    )
    strategic_p.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run: read-only; apply: Codex may edit files (default: dry-run).",
    )
    strategic_p.add_argument(
        "--rounds",
        type=int,
        default=6,
        help="Maximum strategic rounds (default: 6).",
    )
    strategic_p.add_argument(
        "--test-cmd",
        type=str,
        default=None,
        help="Test command (default: evaluator default).",
    )
    strategic_p.add_argument(
        "--codex-bin",
        type=str,
        default="codex",
        help="Path to the codex CLI binary (default: codex).",
    )
    strategic_p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Inactivity timeout in seconds per Codex invocation; 0 disables (default: 600).",
    )
    strategic_p.add_argument(
        "--goal-extra",
        type=str,
        default="",
        help="Optional extra context appended to the strategic goal.",
    )
    strategic_p.add_argument(
        "--focus",
        action="append",
        choices=sorted(STRATEGIC_FOCUS_GUIDANCE),
        default=[],
        help=(
            "Optional product focus area. Repeat for multiple priorities "
            "(reliability, ux, performance, growth, security, testing, dx, docs)."
        ),
    )
    strategic_p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip setup diagnostics before running (not recommended).",
    )

    # Optimize-prompts sub-command
    opt_p = sub.add_parser(
        "optimize-prompts",
        help="AI-optimize all prompts in the catalog.",
    )
    opt_p.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="Model to use for optimization (default: gpt-5.2).",
    )
    opt_p.add_argument(
        "--threshold",
        type=float,
        default=7.5,
        help="Score threshold - prompts above this are kept as-is (default: 7.5).",
    )
    opt_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate prompts but don't save changes.",
    )
    opt_p.add_argument(
        "--local-only",
        action="store_true",
        help="Use local Ollama model instead of cloud APIs.",
    )

    # Visual-test sub-command (CUA)
    vt_p = sub.add_parser(
        "visual-test",
        help="Run a CUA (Computer-Using Agent) visual test session.",
    )
    vt_p.add_argument(
        "--url",
        type=str,
        default="",
        help="URL to test (e.g. http://localhost:5088).",
    )
    vt_p.add_argument(
        "--task",
        type=str,
        default="",
        help="Task / goal for the CUA (default: inspect and report issues).",
    )
    vt_p.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="CUA provider (default: openai).",
    )
    vt_p.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Max CUA steps (default: 30).",
    )
    vt_p.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in visible (headed) mode instead of headless.",
    )
    vt_p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300).",
    )

    # List-prompts sub-command
    sub.add_parser(
        "list-prompts",
        help="List all prompts in the catalog.",
    )

    # List-recipes sub-command
    recipes_p = sub.add_parser(
        "list-recipes",
        help="List built-in GUI recipes and optional step prompts.",
    )
    recipes_p.add_argument(
        "--recipe",
        type=str,
        default="",
        help="Optional recipe id for full step-by-step details.",
    )

    # Doctor sub-command
    doctor_p = sub.add_parser(
        "doctor",
        help="Run setup diagnostics (repo access, binaries, authentication).",
    )
    doctor_p.add_argument(
        "--repo",
        type=str,
        default="",
        help="Optional path to target git repository.",
    )
    doctor_p.add_argument(
        "--agents",
        type=str,
        default="codex",
        help="Comma-separated agents to validate (codex, claude_code, auto).",
    )
    doctor_p.add_argument(
        "--codex-bin",
        type=str,
        default="codex",
        help="Codex CLI binary name/path (default: codex).",
    )
    doctor_p.add_argument(
        "--claude-bin",
        type=str,
        default="claude",
        help="Claude Code CLI binary name/path (default: claude).",
    )
    doctor_p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report instead of human-readable output.",
    )
    # -- Top-level (loop) arguments ------------------------------------------
    p.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Path to the target git repository.",
    )
    p.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Natural-language improvement goal.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Maximum improvement rounds (default: 10).",
    )
    p.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run: read-only; apply: Codex may edit files (default: dry-run).",
    )
    p.add_argument(
        "--test-cmd",
        type=str,
        default=None,
        help='Test command (default: "python -m pytest -q").',
    )
    p.add_argument(
        "--codex-bin",
        type=str,
        default="codex",
        help="Path to the codex CLI binary (default: codex).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Inactivity timeout in seconds per Codex invocation; 0 disables (default: 600).",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip setup diagnostics before running (not recommended).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate mode."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    # -- Logging setup (early, for all modes) --------------------------------
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # -- GUI mode -------------------------------------------------------------
    if args.command == "gui":
        from codex_manager.gui import main as gui_main

        checkpoint = str(getattr(args, "pipeline_resume_checkpoint", "") or "").strip()
        if checkpoint:
            gui_main(
                port=args.port,
                open_browser=not args.no_browser,
                pipeline_resume_checkpoint=checkpoint,
            )
        else:
            gui_main(port=args.port, open_browser=not args.no_browser)
        return 0
    # -- Pipeline mode --------------------------------------------------------
    if args.command == "pipeline":
        return _run_pipeline(args)
    # -- GitHub Actions workflow generation -----------------------------------
    if args.command == "github-actions":
        return _run_github_actions(args)
    # -- Strategic loop mode --------------------------------------------------
    if args.command == "strategic":
        return _run_strategic(args)
    # -- Optimize prompts -----------------------------------------------------
    if args.command == "optimize-prompts":
        return _optimize_prompts(args)
    # -- Visual test (CUA) ---------------------------------------------------
    if args.command == "visual-test":
        return _visual_test(args)
    # -- List prompts ---------------------------------------------------------
    if args.command == "list-prompts":
        return _list_prompts()
    # -- List recipes ---------------------------------------------------------
    if args.command == "list-recipes":
        return _list_recipes(args)

    if args.command == "doctor":
        return _run_doctor(args)
    # -- CLI loop mode (requires --repo and --goal) --------------------------
    if not args.repo or not args.goal:
        parser.print_help()
        print(
            "\nTip: run 'warpfoundry gui' to launch the web GUI,\n"
            "     'warpfoundry doctor --repo <path>' to validate setup,\n"
            "     'warpfoundry pipeline --repo <path>' for the autonomous pipeline,\n"
            "     'warpfoundry github-actions --repo <path>' to generate CI workflow,\n"
            "     'warpfoundry strategic --repo <path>' for strategic product maximization,\n"
            "     'warpfoundry visual-test --url <url>' for CUA visual testing,\n"
            "     (or use 'python -m codex_manager <command>' in source checkouts),\n"
            "     or provide --repo and --goal for CLI loop mode.",
            file=sys.stderr,
        )
        return 1

    return _run_goal_loop(
        repo_path=args.repo,
        goal=args.goal,
        mode=args.mode,
        rounds=args.rounds,
        test_cmd=args.test_cmd,
        codex_bin=args.codex_bin,
        timeout=args.timeout,
        skip_preflight=args.skip_preflight,
    )


def _print_doctor_report(report: PreflightReport) -> None:
    """Print a human-readable diagnostics report."""

    print("\n  WarpFoundry - Setup Diagnostics")
    print("  " + "=" * 58)
    print(f"  Repository: {report.resolved_repo_path or '(not provided)'}")
    print(f"  Agents:     {', '.join(report.requested_agents)}")

    for check in report.checks:
        status = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(check.status, "INFO")
        print(f"\n  [{status}] {check.label}")
        print(f"    {check.detail}")
        if check.hint and check.status != "pass":
            print(f"    Fix: {check.hint}")

    summary = report.summary
    print("\n  " + "-" * 58)
    print(f"  Summary: {summary['pass']} pass, {summary['warn']} warn, {summary['fail']} fail")
    print(f"  Ready:   {'yes' if report.ready else 'no'}")

    actions = list(getattr(report, "next_actions", []) or [])
    if actions:
        print("\n  Next Actions")
        print("  " + "-" * 58)
        for idx, action in enumerate(actions, start=1):
            title = getattr(action, "title", str(action))
            detail = getattr(action, "detail", "")
            command = getattr(action, "command", "")
            severity = getattr(action, "severity", "required")
            badge = "Required" if severity == "required" else "Recommended"

            print(f"  {idx}. [{badge}] {title}")
            if detail:
                print(f"     {detail}")
            if command:
                print(f"     Command: {command}")
    print()


def _print_cli_preflight_guard(
    *,
    repo: Path,
    agents: list[str],
    codex_bin: str,
    claude_bin: str,
) -> bool:
    """Validate required setup and print actionable failures."""
    report = build_preflight_report(
        repo_path=repo,
        agents=agents,
        codex_binary=codex_bin,
        claude_binary=claude_bin,
    )
    if report.ready:
        return True

    print("\nError: preflight checks failed before execution.", file=sys.stderr)
    for message in report.failure_messages():
        print(f"  - {message}", file=sys.stderr)
    print(
        "\nRun diagnostics for full details:\n"
        f'  warpfoundry doctor --repo "{repo}" --agents {",".join(agents)}\n'
        f'  python -m codex_manager doctor --repo "{repo}" --agents {",".join(agents)}',
        file=sys.stderr,
    )
    return False


def _run_doctor(args: argparse.Namespace) -> int:
    """Run setup diagnostics and print the report."""
    report = build_preflight_report(
        repo_path=args.repo,
        agents=parse_agents(args.agents),
        codex_binary=args.codex_bin,
        claude_binary=args.claude_bin,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_doctor_report(report)
    return 0 if report.ready else 1


def _run_goal_loop(
    *,
    repo_path: str,
    goal: str,
    mode: str,
    rounds: int,
    test_cmd: str | None,
    codex_bin: str,
    timeout: int,
    skip_preflight: bool = False,
) -> int:
    """Run the classic CLI goal loop with explicit configuration values."""
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        print(f"Error: repo path does not exist: {repo}", file=sys.stderr)
        return 1
    if not skip_preflight and not _print_cli_preflight_guard(
        repo=repo,
        agents=["codex"],
        codex_bin=codex_bin,
        claude_bin="claude",
    ):
        return 1

    if mode == "dry-run":
        print("\n  WARNING: SAFE MODE ACTIVE (dry-run). Any file edits will be reverted.\n")

    # Build components
    from codex_manager.codex_cli import CodexRunner
    from codex_manager.eval_tools import RepoEvaluator, parse_test_command
    from codex_manager.loop import ImprovementLoop

    runner = CodexRunner(codex_binary=codex_bin, timeout=timeout)
    test_cmd_parts = parse_test_command(test_cmd)
    evaluator = RepoEvaluator(test_cmd=test_cmd_parts)

    loop = ImprovementLoop(
        repo_path=repo,
        goal=goal,
        mode=mode,
        max_rounds=rounds,
        runner=runner,
        evaluator=evaluator,
    )

    state = loop.run()

    # Summary
    print("\n" + "=" * 60)
    print("  WarpFoundry - Run Summary")
    print("=" * 60)
    print(f"  Goal:        {state.goal}")
    print(f"  Mode:        {state.mode}")
    print(f"  Rounds:      {len(state.rounds)} / {state.max_rounds}")
    print(f"  Stop reason: {state.stop_reason}")
    if state.branch_name:
        print(f"  Branch:      {state.branch_name}")
    print(f"  Tokens:      {state.total_input_tokens + state.total_output_tokens:,} total")
    print(f"  State file:  {state.state_path()}")
    print("=" * 60)

    # Per-round table
    if state.rounds:
        print(f"\n  {'#':>3}  {'Tests':<8}  {'Files':>5}  {'Net d':>6}  {'Commit':<10}")
        print(f"  {'-' * 3}  {'-' * 8}  {'-' * 5}  {'-' * 6}  {'-' * 10}")
        for r in state.rounds:
            sha = r.commit_sha or "-"
            print(
                f"  {r.round_number:>3}  "
                f"{r.eval_result.test_outcome.value:<8}  "
                f"{r.eval_result.files_changed:>5}  "
                f"{r.eval_result.net_lines_changed:>+6}  "
                f"{sha:<10}"
            )
    print()

    return 0


def _run_strategic(args: argparse.Namespace) -> int:
    """Run the classic loop with a built-in strategic goal template."""
    repo = Path(args.repo).resolve()
    goal = _build_strategic_goal(
        repo=repo,
        goal_extra=str(getattr(args, "goal_extra", "") or ""),
        focus_values=list(getattr(args, "focus", []) or []),
    )

    return _run_goal_loop(
        repo_path=str(repo),
        goal=goal,
        mode=args.mode,
        rounds=args.rounds,
        test_cmd=args.test_cmd,
        codex_bin=args.codex_bin,
        timeout=args.timeout,
        skip_preflight=getattr(args, "skip_preflight", False),
    )


def _run_github_actions(args: argparse.Namespace) -> int:
    """Generate a GitHub Actions workflow that runs the WarpFoundry pipeline."""
    from codex_manager.github_actions import (
        PipelineWorkflowConfig,
        detect_default_branch,
        generate_pipeline_workflow,
        normalize_branches,
    )

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"Error: repo path does not exist: {repo}", file=sys.stderr)
        return 1
    if int(args.cycles) < 1:
        print("Error: --cycles must be at least 1.", file=sys.stderr)
        return 1
    if int(args.max_time) < 1:
        print("Error: --max-time must be at least 1 minute.", file=sys.stderr)
        return 1

    branches = normalize_branches(list(getattr(args, "branch", []) or []))
    if not branches:
        branches = (detect_default_branch(repo),)

    config = PipelineWorkflowConfig(
        branches=branches,
        python_version=str(args.python_version),
        mode=str(args.mode),
        cycles=int(args.cycles),
        max_time_minutes=int(args.max_time),
        agent=str(args.agent),
        artifact_prefix=str(args.artifact_prefix),
    )

    try:
        workflow_path = generate_pipeline_workflow(
            repo,
            config=config,
            workflow_filename=str(args.workflow_file),
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    except (FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if isinstance(exc, FileExistsError):
            print("Tip: rerun with --overwrite to replace it.", file=sys.stderr)
        return 1

    print("\n  WarpFoundry - GitHub Actions Workflow")
    print("  " + "=" * 58)
    print(f"  Repository: {repo}")
    print(f"  Workflow:   {workflow_path}")
    print(f"  Branches:   {', '.join(branches)}")
    print(
        "  Pipeline:   "
        f"warpfoundry pipeline --repo . --mode {config.mode} --cycles {config.cycles} "
        f"--max-time {config.max_time_minutes} --agent {config.agent}"
    )
    print("  Artifacts:  .codex_manager/logs/** and .codex_manager/outputs/**/*.md")
    print("  Secrets:    Configure CODEX_API_KEY and/or provider API keys in repository secrets.")
    print("  " + "=" * 58 + "\n")
    return 0


def _pipeline_resume_checkpoint_path(repo: Path) -> Path:
    """Return the default pipeline resume checkpoint path for a repository."""
    return repo / ".codex_manager" / "state" / "pipeline_resume.json"


def _load_pipeline_resume_checkpoint(
    checkpoint_path: Path,
) -> tuple[Path, dict[str, object], int, int]:
    """Load and validate a pipeline resume checkpoint payload."""
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read checkpoint: {checkpoint_path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse checkpoint JSON: {checkpoint_path} ({exc})") from exc

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a JSON object.")

    repo_raw = str(payload.get("repo_path") or "").strip()
    if not repo_raw:
        raise ValueError("Checkpoint missing repo_path.")
    repo_path = Path(repo_raw).resolve()
    if not repo_path.is_dir():
        raise ValueError(f"Checkpoint repo_path does not exist: {repo_path}")

    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Checkpoint missing pipeline config object.")

    try:
        resume_cycle = int(payload.get("resume_cycle") or 1)
        resume_phase_index = int(payload.get("resume_phase_index") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint has invalid resume_cycle/resume_phase_index values.") from exc

    return repo_path, config_payload, max(1, resume_cycle), max(0, resume_phase_index)


# Pipeline command
def _run_pipeline(args: argparse.Namespace) -> int:
    """Run the autonomous improvement pipeline."""
    from codex_manager.pipeline import PipelineConfig, PipelineOrchestrator

    repo_raw = str(getattr(args, "repo", "") or "").strip()
    resume_checkpoint_raw = str(getattr(args, "resume_checkpoint", "") or "").strip()
    resume_state = bool(getattr(args, "resume_state", False))
    webhook_urls_arg = getattr(args, "webhook_url", None)
    webhook_timeout_arg = getattr(args, "webhook_timeout", None)
    if resume_checkpoint_raw and resume_state:
        print("Error: --resume-checkpoint and --resume-state cannot be used together.", file=sys.stderr)
        return 1

    checkpoint_to_consume: Path | None = None
    if resume_checkpoint_raw:
        checkpoint_to_consume = Path(resume_checkpoint_raw).expanduser().resolve()
    elif resume_state:
        repo_hint = Path(repo_raw).resolve() if repo_raw else Path.cwd().resolve()
        checkpoint_to_consume = _pipeline_resume_checkpoint_path(repo_hint)

    resume_cycle = 1
    resume_phase_index = 0
    if checkpoint_to_consume is not None:
        if not checkpoint_to_consume.is_file():
            print(f"Error: resume checkpoint not found: {checkpoint_to_consume}", file=sys.stderr)
            if resume_state and not repo_raw:
                print(
                    "Tip: pass --repo <path> when resuming state from a different repository.",
                    file=sys.stderr,
                )
            return 1
        try:
            repo, config_payload, resume_cycle, resume_phase_index = (
                _load_pipeline_resume_checkpoint(checkpoint_to_consume)
            )
            config = PipelineConfig(**config_payload)
        except Exception as exc:
            print(f"Error: could not resume from checkpoint: {exc}", file=sys.stderr)
            return 1
        if repo_raw:
            requested_repo = Path(repo_raw).resolve()
            if requested_repo != repo:
                print(
                    "Error: --repo does not match checkpoint repo_path.",
                    file=sys.stderr,
                )
                print(f"  --repo: {requested_repo}", file=sys.stderr)
                print(f"  checkpoint repo_path: {repo}", file=sys.stderr)
                return 1
        print(
            "\n  Resuming from checkpoint: "
            f"{checkpoint_to_consume} (cycle={resume_cycle}, phase_index={resume_phase_index})"
        )
    else:
        if not repo_raw:
            print(
                "Error: --repo is required unless --resume-checkpoint or --resume-state is provided.",
                file=sys.stderr,
            )
            return 1
        repo = Path(repo_raw).resolve()
        if not repo.is_dir():
            print(f"Error: repo path does not exist: {repo}", file=sys.stderr)
            return 1

        config = PipelineConfig(
            mode=args.mode,
            max_cycles=args.cycles,
            agent=args.agent,
            science_enabled=args.science,
            brain_enabled=args.brain,
            brain_model=args.brain_model,
            local_only=args.local_only,
            codex_binary=getattr(args, "codex_bin", "codex"),
            claude_binary=getattr(args, "claude_bin", "claude"),
            timeout_per_phase=args.timeout,
            max_total_tokens=args.max_tokens,
            max_time_minutes=args.max_time,
            run_completion_webhooks=[
                str(url).strip() for url in list(webhook_urls_arg or []) if str(url).strip()
            ],
            run_completion_webhook_timeout_seconds=(
                int(webhook_timeout_arg)
                if webhook_timeout_arg is not None
                else 10
            ),
        )
        if args.test_cmd:
            config.test_cmd = args.test_cmd

    if webhook_urls_arg is not None:
        config.run_completion_webhooks = [
            str(url).strip() for url in list(webhook_urls_arg or []) if str(url).strip()
        ]
    if webhook_timeout_arg is not None:
        config.run_completion_webhook_timeout_seconds = max(2, min(60, int(webhook_timeout_arg)))

    if not getattr(args, "skip_preflight", False) and not _print_cli_preflight_guard(
        repo=repo,
        agents=[str(getattr(config, "agent", "codex"))],
        codex_bin=str(getattr(config, "codex_binary", "codex")),
        claude_bin=str(getattr(config, "claude_binary", "claude")),
    ):
        return 1

    if config.mode == "dry-run":
        print("\n  WARNING: SAFE MODE ACTIVE (dry-run). Any file edits will be reverted.\n")

    pipeline = PipelineOrchestrator(
        repo_path=repo,
        config=config,
        resume_cycle=resume_cycle,
        resume_phase_index=resume_phase_index,
    )
    state = pipeline.run()
    if checkpoint_to_consume is not None:
        try:
            checkpoint_to_consume.unlink(missing_ok=True)
        except OSError as exc:
            print(
                f"Warning: could not remove consumed checkpoint {checkpoint_to_consume}: {exc}",
                file=sys.stderr,
            )

    # Summary
    print("\n" + "=" * 60)
    print("  WarpFoundry - Pipeline Summary")
    print("=" * 60)
    print(f"  Mode:        {config.mode}")
    print(f"  Cycles:      {state.total_cycles_completed} / {config.max_cycles}")
    print(f"  Phases:      {state.total_phases_completed}")
    print(f"  Stop reason: {state.stop_reason}")
    print(f"  Tokens:      {state.total_tokens:,}")
    print(f"  Elapsed:     {state.elapsed_seconds:.0f}s")
    if config.science_enabled:
        print("  Science:     enabled")
    print("=" * 60)

    # Per-phase results
    if state.results:
        print(
            f"\n  {'Phase':<20}  {'Iter':>4}  {'Status':<8}  {'Tests':<8}  {'Files':>5}  {'Net d':>6}"
        )
        print(f"  {'-' * 20}  {'-' * 4}  {'-' * 8}  {'-' * 8}  {'-' * 5}  {'-' * 6}")
        for r in state.results:
            status = "OK" if r.success else "FAIL"
            print(
                f"  {r.phase:<20}  "
                f"{r.iteration:>4}  "
                f"{status:<8}  "
                f"{r.test_outcome:<8}  "
                f"{r.files_changed:>5}  "
                f"{r.net_lines_changed:>+6}"
            )
    print()
    return 0


# -- Optimize prompts command -------------------------------------------------


def _optimize_prompts(args: argparse.Namespace) -> int:
    """AI-optimize all prompts in the catalog."""
    from codex_manager.prompts import PromptOptimizer

    # When --local-only is set, override model to default Ollama model
    model = args.model
    if args.local_only:
        from codex_manager.brain.connector import get_default_ollama_model

        model = get_default_ollama_model()
        print(f"\n  Local-only mode: using {model}")

    print(f"\n  Optimizing prompts (model={model}, threshold={args.threshold})")
    if args.dry_run:
        print("  DRY RUN - no changes will be saved\n")
    else:
        print()

    optimizer = PromptOptimizer(model=model, threshold=args.threshold)
    optimizer.optimize_all(dry_run=args.dry_run)
    print(f"\n{optimizer.summary()}")
    return 0


# -- Visual test command ------------------------------------------------------


def _visual_test(args: argparse.Namespace) -> int:
    """Run a CUA visual test session."""
    try:
        from codex_manager.cua.actions import CUAProvider, CUASessionConfig
        from codex_manager.cua.session import run_cua_session_sync
    except ImportError as exc:
        print(
            f"\nError: CUA dependencies not installed: {exc}\n"
            "Install with:\n"
            "  pip install warpfoundry[cua]\n"
            "  python -m playwright install\n",
            file=sys.stderr,
        )
        return 1

    provider = CUAProvider.ANTHROPIC if args.provider == "anthropic" else CUAProvider.OPENAI
    task = args.task or (
        "Visually inspect the application UI. Navigate through the main views, "
        "test interactive elements (buttons, forms, dropdowns), and report any "
        "visual bugs, broken layouts, or usability issues you find."
    )

    config = CUASessionConfig(
        provider=provider,
        target_url=args.url,
        task=task,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout,
        headless=not args.headed,
        save_screenshots=True,
    )

    print("\n  CUA Visual Test")
    print(f"  {'=' * 50}")
    print(f"  Provider:   {provider.value}")
    if provider == CUAProvider.OPENAI:
        print(f"  Model:      {config.openai_model}")
    elif provider == CUAProvider.ANTHROPIC:
        print(f"  Model:      {config.anthropic_model}")
    print(f"  URL:        {args.url or '(blank page)'}")
    print(f"  Task:       {task[:80]}...")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Headless:   {config.headless}")
    print(f"  {'=' * 50}\n")

    result = run_cua_session_sync(config)

    # Summary
    print(f"\n  {'=' * 50}")
    print("  CUA Session Complete")
    print(f"  {'=' * 50}")
    print(f"  Success:    {result.success}")
    print(f"  Steps:      {result.total_steps}")
    print(f"  Duration:   {result.duration_seconds}s")
    if result.error:
        print(f"  Error:      {result.error}")
        if (
            "CONNECTION_REFUSED" in result.error.upper()
            or "connection refused" in result.error.lower()
        ):
            print(
                "\n  Hint: Start the app at that URL first. For the default WarpFoundry GUI:\n"
                "        warpfoundry gui\n"
            )
    if result.observations:
        print(f"\n  Observations ({len(result.observations)}):")
        for obs in result.observations:
            icon = {
                "critical": "!!",
                "major": "! ",
                "minor": "- ",
                "cosmetic": "  ",
                "positive": "+ ",
            }.get(obs.severity, "  ")
            print(f"    {icon}[{obs.severity.upper()}] {obs.element}")
            if obs.actual:
                print(f"      Actual: {obs.actual}")
            if obs.expected:
                print(f"      Expected: {obs.expected}")
            if obs.recommendation:
                print(f"      Fix: {obs.recommendation}")
    if result.summary:
        # Print summary without raw OBSERVATION lines
        clean_lines = [
            summary_line
            for summary_line in result.summary.split("\n")
            if not summary_line.strip().upper().startswith("OBSERVATION|")
        ]
        if any(summary_line.strip() for summary_line in clean_lines):
            print("\n  Summary:")
            for line in clean_lines:
                if line.strip():
                    print(f"    {line}")
    if result.screenshots_saved:
        print(f"\n  Screenshots ({len(result.screenshots_saved)}):")
        for s in result.screenshots_saved[:10]:
            print(f"    {s}")
        if len(result.screenshots_saved) > 10:
            print(f"    ... and {len(result.screenshots_saved) - 10} more")
    print()
    return 0 if result.success else 1


# -- List prompts command -----------------------------------------------------


def _list_recipes(args: argparse.Namespace) -> int:
    """List built-in GUI recipes and optional detailed steps."""
    from codex_manager.gui.recipes import DEFAULT_RECIPE_ID, get_recipe, list_recipe_summaries

    requested = (getattr(args, "recipe", "") or "").strip()
    summaries = list_recipe_summaries()
    summary_map = {entry["id"]: entry for entry in summaries}

    if requested:
        recipe = get_recipe(requested)
        if recipe is None:
            available = ", ".join(sorted(summary_map))
            print(f"\nError: unknown recipe id '{requested}'.", file=sys.stderr)
            print(f"Available recipe ids: {available}", file=sys.stderr)
            return 1

        print(f"\n  Recipe: {recipe['name']} ({recipe['id']})")
        description = str(recipe.get("description", "")).strip()
        if description:
            print(f"  Description: {description}")
        sequence = str(recipe.get("sequence", "")).strip()
        if sequence:
            print(f"  Sequence: {sequence}")

        steps = recipe.get("steps", [])
        if isinstance(steps, list):
            print(f"  Steps: {len(steps)}")
            for idx, step in enumerate(steps, start=1):
                step_name = str(step.get("name") or step.get("job_type") or f"step-{idx}")
                job_type = str(step.get("job_type", "custom"))
                try:
                    loop_count = int(step.get("loop_count", 1) or 1)
                except (TypeError, ValueError):
                    loop_count = 1
                prompt_mode = str(step.get("prompt_mode", "preset"))

                print(
                    f"\n  {idx}. {step_name}"
                    f"\n     job_type={job_type}, loop_count={loop_count}, prompt_mode={prompt_mode}"
                )

                custom_prompt = str(step.get("custom_prompt", "") or "").strip()
                if custom_prompt:
                    print(f"     custom_prompt: {custom_prompt}")
        print()
        return 0

    print(f"\n  Built-in Recipes - {len(summaries)}")
    print("  " + "=" * 58)
    for entry in summaries:
        marker = " (default)" if entry["id"] == DEFAULT_RECIPE_ID else ""
        print(f"\n  [{entry['id']}] {entry['name']}{marker}")
        print(f"    {entry['description']}")
        print(f"    Steps: {entry['step_count']}")
        sequence = str(entry.get("sequence", "")).strip()
        if sequence:
            print(f"    Sequence: {sequence}")

    print("\n  Detail: warpfoundry list-recipes --recipe <id>")
    print()
    return 0


# -- List prompts command -----------------------------------------------------


def _list_prompts() -> int:
    """List all prompts in the catalog."""
    from codex_manager.prompts.catalog import get_catalog

    catalog = get_catalog()
    all_prompts = catalog.all_prompts()

    print(f"\n  Prompt Catalog - {len(all_prompts)} prompts")
    print("  " + "=" * 58)

    for entry in all_prompts:
        content = entry["content"]
        preview = content[:80].replace("\n", " ").strip()
        print(f"\n  [{entry['path']}]")
        print(f"    {entry['name']}")
        print(f"    {preview}...")
        print(f"    ({len(content)} chars)")

    print(f"\n  Edit: {Path(__file__).resolve().parent / 'prompts' / 'templates.yaml'}")
    print(f"  Override: {Path.home() / '.codex_manager' / 'prompt_overrides.yaml'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
