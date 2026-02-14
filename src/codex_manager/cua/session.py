"""CUA session manager — orchestrates the computer-use agent loop.

Brings together:
- A browser environment (Playwright)
- A CUA provider (OpenAI or Anthropic)
- The action execution + screenshot loop

Usage::

    config = CUASessionConfig(
        provider=CUAProvider.OPENAI,
        target_url="http://localhost:5088",
        task="Test the Chain Builder UI: add a step and verify it appears",
    )
    result = await run_cua_session(config)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codex_manager.cua.actions import (
    CUA_OBSERVATION_SUFFIX,
    ActionType,
    CUAObservation,
    CUAProvider,
    CUASessionConfig,
    CUASessionResult,
    CUAStepResult,
)
from codex_manager.cua.browser import BrowserEnvironment

if TYPE_CHECKING:
    from codex_manager.ledger import KnowledgeLedger

logger = logging.getLogger(__name__)


def _parse_observations(text: str, screenshots: list[str] | None = None) -> list[CUAObservation]:
    """Parse OBSERVATION lines from the CUA's final summary text.

    Each line has the format:
        OBSERVATION|severity|category|element|expected|actual|recommendation
    """
    observations: list[CUAObservation] = []
    if not text:
        return observations
    for line in text.splitlines():
        line = line.strip()
        if not line.upper().startswith("OBSERVATION|"):
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        obs = CUAObservation(
            severity=parts[1].strip().lower() or "minor",
            category=parts[2].strip().lower(),
            element=parts[3].strip(),
            expected=parts[4].strip(),
            actual=parts[5].strip(),
            recommendation=parts[6].strip() if len(parts) > 6 else "",
        )
        observations.append(obs)
    # Attach latest screenshot to each observation
    if screenshots and observations:
        last_ss = screenshots[-1] if screenshots else ""
        for obs in observations:
            obs.screenshot = last_ss
    return observations


async def run_cua_session(
    config: CUASessionConfig,
    ledger: KnowledgeLedger | None = None,
    step_ref: str = "",
) -> CUASessionResult:
    """Run a complete CUA session and return the results.

    This is the main entry point. It:
    1. Launches a browser
    2. Navigates to the target URL
    3. Takes an initial screenshot
    4. Sends it to the CUA model with the task
    5. Loops: execute action -> screenshot -> send back -> repeat
    6. Stops when the model is done or limits are reached

    If ``ledger`` is provided, parsed observations are written to the project
    knowledge ledger so other phases (e.g. debugging) can see them.
    """
    # Enhance the task prompt with structured observation instructions
    enhanced_task = config.task + CUA_OBSERVATION_SUFFIX

    result = CUASessionResult(
        task=config.task,
        provider=config.provider.value,
        started_at=dt.datetime.now().isoformat(),
    )

    # Set up screenshot directory
    screenshots_dir: Path | None = None
    if config.save_screenshots:
        base = config.screenshots_dir or str(Path.home() / ".codex_manager" / "cua_screenshots")
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshots_dir = Path(base) / ts
        screenshots_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()

    try:
        async with BrowserEnvironment(
            width=config.viewport_width,
            height=config.viewport_height,
            headless=config.headless,
        ) as browser:
            # Navigate to target
            if config.target_url:
                await browser.navigate(config.target_url)
                await asyncio.sleep(1)  # Let the page settle

            # Take initial screenshot
            initial_screenshot = await browser.screenshot_b64()
            if screenshots_dir:
                await browser.save_screenshot(screenshots_dir / "step_000_initial.png")
                result.screenshots_saved.append(str(screenshots_dir / "step_000_initial.png"))

            # Dispatch to the appropriate provider loop
            if config.provider == CUAProvider.OPENAI:
                await _openai_loop(
                    config,
                    browser,
                    initial_screenshot,
                    result,
                    screenshots_dir,
                    start_time,
                    enhanced_task,
                )
            elif config.provider == CUAProvider.ANTHROPIC:
                await _anthropic_loop(
                    config,
                    browser,
                    initial_screenshot,
                    result,
                    screenshots_dir,
                    start_time,
                    enhanced_task,
                )
            else:
                result.error = f"Unsupported CUA provider: {config.provider}"

            # Parse structured observations from the summary
            if result.summary:
                result.observations = _parse_observations(result.summary, result.screenshots_saved)
                if result.observations:
                    logger.info(
                        "Parsed %d structured observations from CUA summary",
                        len(result.observations),
                    )
                    # Write to project knowledge ledger when provided (pipeline/chain)
                    if ledger is not None:
                        source = f"cua:{config.provider.value}"
                        for obs in result.observations:
                            detail = f"Expected: {obs.expected}\nActual: {obs.actual}"
                            if obs.recommendation:
                                detail += f"\nRecommendation: {obs.recommendation}"
                            ledger.add(
                                category="observation",
                                title=obs.element or "Unknown element",
                                detail=detail,
                                severity=obs.severity,
                                source=source,
                                file_path=obs.screenshot or "",
                                step_ref=step_ref,
                            )

            result.success = not result.error

    except Exception as exc:
        logger.error("CUA session failed: %s", exc)
        err_msg = str(exc)
        # Clarify OpenAI CUA 404: usually means account doesn't have access (Tier 3+ required)
        if "404" in err_msg and (
            "does not exist or you do not have access" in err_msg or "model_not_found" in err_msg
        ):
            err_msg += (
                " The computer-use model often requires OpenAI API Tier 3+ or allowlist access. "
                "See https://platform.openai.com/docs/models/computer-use-preview. "
                "You can try a snapshot via CUA_OPENAI_MODEL (e.g. computer-use-preview-2025-03-11)."
            )
        result.error = err_msg

    result.duration_seconds = round(time.monotonic() - start_time, 1)
    result.total_steps = len(result.steps)
    result.finished_at = dt.datetime.now().isoformat()

    logger.info(
        "CUA session complete: %d steps, %.1fs, success=%s",
        result.total_steps,
        result.duration_seconds,
        result.success,
    )
    return result


# ══════════════════════════════════════════════════════════════════
# OpenAI CUA loop
# ══════════════════════════════════════════════════════════════════


async def _openai_loop(
    config: CUASessionConfig,
    browser: BrowserEnvironment,
    initial_screenshot: str,
    result: CUASessionResult,
    screenshots_dir: Path | None,
    start_time: float = 0.0,
    task_text: str = "",
) -> None:
    """Run the OpenAI CUA loop."""
    from codex_manager.cua.openai_cua import OpenAICUA, parse_openai_action

    task = task_text or config.task

    provider = OpenAICUA(
        model=config.openai_model,
        display_width=config.viewport_width,
        display_height=config.viewport_height,
        environment="browser",
    )

    logger.info("Starting OpenAI CUA loop: %s", config.task)

    # Initial request
    response = await asyncio.to_thread(
        provider.create_initial_request,
        task,
        initial_screenshot,
    )

    for step_num in range(1, config.max_steps + 1):
        # Check timeout
        elapsed = time.monotonic() - start_time
        if config.timeout_seconds > 0 and elapsed > config.timeout_seconds:
            result.error = f"Timeout after {config.timeout_seconds}s"
            break

        # Extract computer call
        call_id, action_data, reasoning = provider.extract_computer_call(response)
        if call_id is None or action_data is None:
            # Model is done — extract final text
            final_text = provider.extract_text_output(response)
            result.summary = final_text or reasoning or "Task completed"
            logger.info("CUA finished: %s", result.summary[:200])
            break

        # Parse and execute the action
        action = parse_openai_action(action_data)
        step_result = CUAStepResult(action=action, reasoning=reasoning)

        logger.info(
            "Step %d: %s at (%d,%d) — %s",
            step_num,
            action.action_type.value,
            action.x,
            action.y,
            reasoning[:100] if reasoning else "no reasoning",
        )

        try:
            await browser.execute_action(action)
            await asyncio.sleep(0.8)  # Let page settle
            step_result.success = True
        except Exception as exc:
            logger.warning("Action failed: %s", exc)
            step_result.success = False
            step_result.error = str(exc)

        # Screenshot after action
        screenshot = await browser.screenshot_b64()
        step_result.screenshot_b64 = screenshot[:100] + "..."  # Don't store full b64 in result

        if screenshots_dir:
            fname = f"step_{step_num:03d}_{action.action_type.value}.png"
            await browser.save_screenshot(screenshots_dir / fname)
            result.screenshots_saved.append(str(screenshots_dir / fname))

        result.steps.append(step_result)

        # Send screenshot back to model
        response = await asyncio.to_thread(
            provider.send_screenshot,
            response.id,
            call_id,
            screenshot,
        )

    else:
        result.summary = f"Reached max steps ({config.max_steps})"


# ══════════════════════════════════════════════════════════════════
# Anthropic CUA loop
# ══════════════════════════════════════════════════════════════════


async def _anthropic_loop(
    config: CUASessionConfig,
    browser: BrowserEnvironment,
    initial_screenshot: str,
    result: CUASessionResult,
    screenshots_dir: Path | None,
    start_time: float = 0.0,
    task_text: str = "",
) -> None:
    """Run the Anthropic Claude CUA loop."""
    from codex_manager.cua.anthropic_cua import AnthropicCUA, parse_anthropic_action

    task = task_text or config.task

    provider = AnthropicCUA(
        model=config.anthropic_model,
        tool_version=config.anthropic_tool_version,
        beta_flag=config.anthropic_beta,
        display_width=config.viewport_width,
        display_height=config.viewport_height,
    )

    logger.info("Starting Anthropic CUA loop: %s", config.task)

    # Build initial messages
    messages: list[dict[str, Any]] = []

    # Initial request
    response = await asyncio.to_thread(
        provider.create_initial_request,
        task,
        initial_screenshot,
    )

    # Add assistant response to history
    messages.append({"role": "user", "content": task})
    messages.append({"role": "assistant", "content": response.content})

    for step_num in range(1, config.max_steps + 1):
        # Check timeout
        if config.timeout_seconds > 0:
            elapsed = time.monotonic() - start_time
            if elapsed > config.timeout_seconds:
                result.error = f"Timeout after {config.timeout_seconds}s"
                break

        # Check stop reason
        if response.stop_reason == "end_turn":
            final_text = provider.extract_text_output(response)
            result.summary = final_text or "Task completed"
            logger.info("CUA finished (end_turn): %s", result.summary[:200])
            break

        # Extract computer call
        tool_use_id, tool_input, reasoning = provider.extract_computer_call(response)
        if tool_use_id is None or tool_input is None:
            final_text = provider.extract_text_output(response)
            result.summary = final_text or reasoning or "No more actions"
            break

        # Parse and execute the action
        action = parse_anthropic_action(tool_input)
        step_result = CUAStepResult(action=action, reasoning=reasoning)

        logger.info(
            "Step %d: %s at (%d,%d) — %s",
            step_num,
            action.action_type.value,
            action.x,
            action.y,
            reasoning[:100] if reasoning else "no reasoning",
        )

        # Handle screenshot-only actions (no browser action needed)
        if action.action_type == ActionType.SCREENSHOT:
            step_result.success = True
        else:
            try:
                await browser.execute_action(action)
                await asyncio.sleep(0.8)
                step_result.success = True
            except Exception as exc:
                logger.warning("Action failed: %s", exc)
                step_result.success = False
                step_result.error = str(exc)

        # Screenshot after action
        screenshot = await browser.screenshot_b64()
        step_result.screenshot_b64 = screenshot[:100] + "..."

        if screenshots_dir:
            fname = f"step_{step_num:03d}_{action.action_type.value}.png"
            await browser.save_screenshot(screenshots_dir / fname)
            result.screenshots_saved.append(str(screenshots_dir / fname))

        result.steps.append(step_result)

        # Send screenshot back to Claude
        response = await asyncio.to_thread(
            provider.send_tool_result,
            messages,
            tool_use_id,
            screenshot,
        )

        # Add new assistant response to history
        messages.append({"role": "assistant", "content": response.content})

    else:
        result.summary = f"Reached max steps ({config.max_steps})"


# ══════════════════════════════════════════════════════════════════
# Synchronous wrapper (for CLI / non-async contexts)
# ══════════════════════════════════════════════════════════════════


def run_cua_session_sync(
    config: CUASessionConfig,
    ledger: KnowledgeLedger | None = None,
    step_ref: str = "",
) -> CUASessionResult:
    """Synchronous wrapper for :func:`run_cua_session`."""
    return asyncio.run(run_cua_session(config, ledger=ledger, step_ref=step_ref))
