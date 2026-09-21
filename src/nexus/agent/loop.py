"""Bounded observe -> plan -> gate -> execute -> observe loop."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from nexus.computer.controller import ComputerController, ControllerResult
from nexus.computer.schemas import ComputerAction
from nexus.safety.policy import check_action
from nexus.semif.adapter import SemanticDecisionEngine
from nexus.semif.decisions import ACTION_GATING, TASK_PROGRESS, TOOL_ROUTING
from nexus.vision.schemas import ScreenObservation, TaskContext


class VisionProvider(Protocol):
    def analyze(self, screenshot: str | Path, context: TaskContext) -> ScreenObservation: ...


class Planner(Protocol):
    def propose(self, context: TaskContext, observation: ScreenObservation): ...


class AutonomousLoop:
    def __init__(self, controller: ComputerController, vision: VisionProvider | None, planner: Planner,
                 max_steps: int = 10, semantic_engine: SemanticDecisionEngine | None = None):
        self.controller, self.vision, self.planner = controller, vision, planner
        self.semantic_engine = semantic_engine
        self.max_steps = max_steps

    def run(self, task: str) -> list[ControllerResult]:
        results: list[ControllerResult] = []
        previous = ""
        if self.semantic_engine is not None:
            route = self.semantic_engine.decide(
                {"task": task}, "Which tool should handle this task?", TOOL_ROUTING, "route-task"
            )
            if route.winner_id != "computer":
                raise PermissionError(f"SemIf routed task to {route.winner_id}")
        for step_number in range(self.max_steps):
            context = TaskContext(task=task, step=step_number, previous_result=previous)
            screenshot = self.controller.capture_screenshot()
            if not screenshot.ok or not screenshot.screenshot_path:
                raise RuntimeError(screenshot.detail)
            observation = (
                self.vision.analyze(screenshot.screenshot_path, context)
                if self.vision is not None
                else ScreenObservation(visible_app="windows-uia")
            )
            try:
                authoritative = self.controller.accessibility_elements()
            except (AttributeError, RuntimeError):
                authoritative = []
            if authoritative:
                observation = observation.model_copy(update={"elements": authoritative})
            plan = self.planner.propose(context, observation)
            proposal = dict(plan.action)
            # Some local models use JSON Schema's keyword `type` for the
            # action discriminator. Normalize that one harmless alias at the
            # boundary; all downstream execution still uses `action`.
            if "action" not in proposal and isinstance(proposal.get("type"), str):
                proposal["action"] = proposal.pop("type")
            action = ComputerAction.model_validate(proposal)
            verdict = check_action(plan.tool, action.model_dump(), plan.requires_confirmation)
            if verdict != "allow":
                raise PermissionError(f"action requires confirmation or was denied: {verdict}")
            if self.semantic_engine is not None:
                gate = self.semantic_engine.decide(
                    {"task": task, "step": step_number, "action": action.model_dump()},
                    "Should this computer action execute now?", ACTION_GATING, f"gate-{step_number}"
                )
                if gate.winner_id != "allow":
                    raise PermissionError(f"SemIf action gate: {gate.winner_id}")
            result = self.controller.execute(action)
            results.append(result)
            if not result.ok:
                break
            # Verify the desktop state after every successful input before
            # accepting a completed plan step or continuing the loop.
            after = self.controller.capture_screenshot()
            if not after.ok or not after.screenshot_path:
                raise RuntimeError(f"post-action screenshot failed: {after.detail}")
            previous = result.detail
            if plan.expected_result.lower() in {"done", "complete", "completed"}:
                if self.semantic_engine is not None:
                    progress = self.semantic_engine.decide(
                        {"task": task, "step": step_number, "result": result.detail},
                        "What is the task state after this verified action?", TASK_PROGRESS,
                        f"progress-{step_number}"
                    )
                    if progress.winner_id == "failed":
                        raise RuntimeError("SemIf classified the task as failed")
                break
        return results
