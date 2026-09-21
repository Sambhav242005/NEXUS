"""Deterministic computer-action candidates selected by SemIf.

This module never asks a generative model to emit executable action JSON.
"""
from __future__ import annotations

import re

from nexus.agent.schemas import PlanStep
from nexus.agent.intent import IntentStep
from nexus.semif.adapter import SemanticDecisionEngine
from nexus.semif.schemas import DecisionOption
from nexus.vision.schemas import ScreenObservation, TaskContext


class SemIfCandidatePlanner:
    def __init__(self, semantic_engine: SemanticDecisionEngine,
                 intent_steps: list[IntentStep] | None = None):
        self.semantic_engine = semantic_engine
        self.intent_steps = intent_steps or []

    @staticmethod
    def _text(task: str) -> str:
        match = re.search(r"\b(?:write|type)\s+['\"]?(.+?)['\"]?\s*[.!]?\s*$", task, re.I)
        return match.group(1).strip().rstrip(".!\"") if match else "Hello world"

    def _commands(self, task: str) -> list[str]:
        """Extract command words while discarding conversational glue."""
        planned = [step.value for step in self.intent_steps if step.kind == "run_command"]
        if planned:
            return planned
        matches = re.findall(r"\brun\s+([\w./:-]+)(?:\s+command)?", task, re.I)
        return [match.strip() for match in matches]

    def _intent_actions(self, task: str) -> list[PlanStep]:
        """Expand deterministic intent steps into one-action computer steps."""
        actions: list[PlanStep] = []
        for step in self.intent_steps:
            if step.kind == "computer_task" and "tab" in step.value.lower():
                actions.append(PlanStep(
                    goal="open a new terminal tab", tool="computer",
                    action={"action": "hotkey", "keys": ["ctrl", "shift", "t"]},
                    expected_result="needs_more_observation",
                ))
            elif step.kind == "run_command":
                actions.extend((
                    PlanStep(
                        goal=f"enter terminal command {step.value}", tool="computer",
                        action={"action": "type", "text": step.value},
                        expected_result="needs_more_observation",
                    ),
                    PlanStep(
                        goal="submit the terminal command", tool="computer",
                        action={"action": "keypress", "key": "enter"},
                        expected_result="needs_more_observation",
                    ),
                ))
        if actions:
            actions[-1] = actions[-1].model_copy(update={"expected_result": "completed"})
        return actions

    def _candidates(self, context: TaskContext, observation: ScreenObservation) -> list[PlanStep]:
        task = context.task.lower()
        candidates: list[PlanStep] = []
        if context.step == 0 and re.search(r"\b(new file|new document|create a new)\b", task):
            candidates.append(PlanStep(
                goal="create a new blank document",
                tool="computer",
                action={"action": "hotkey", "keys": ["ctrl", "n"]},
                expected_result="needs_more_observation",
            ))
        elif self._intent_actions(context.task):
            actions = self._intent_actions(context.task)
            if context.step < len(actions):
                candidates.append(actions[context.step])
        elif re.search(r"\brun\s+.+", task) and "terminal" in task:
            commands = self._commands(context.task)
            command_index = context.step // 2
            is_typing = context.step % 2 == 0
            if is_typing and command_index < len(commands):
                final_command = command_index == len(commands) - 1
                candidates.append(PlanStep(
                    goal=f"enter terminal command {commands[command_index]}",
                    tool="computer", action={"action": "type", "text": commands[command_index]},
                    expected_result="needs_more_observation",
                ))
            elif not is_typing and command_index < len(commands):
                final_command = command_index == len(commands) - 1
                candidates.append(PlanStep(
                    goal="submit the terminal command",
                    tool="computer", action={"action": "keypress", "key": "enter"},
                    expected_result="completed" if final_command else "needs_more_observation",
                ))
        elif re.search(r"\b(write|type)\b", task):
            candidates.append(PlanStep(
                goal="enter the requested text into the focused application",
                tool="computer",
                action={"action": "type", "text": self._text(context.task)},
                expected_result="completed",
            ))
        for element in observation.elements[:10]:
            if element.source != "uia" or element.x is None or element.y is None:
                continue
            if element.width is None or element.height is None or element.width <= 0 or element.height <= 0:
                continue
            labels = [value.lower() for value in (element.label, element.role) if value]
            if any(label and label in task for label in labels):
                candidates.append(PlanStep(
                    goal=f"click visible target {element.label or element.role}",
                    tool="computer",
                    action={"action": "click", "x": element.x + element.width // 2,
                            "y": element.y + element.height // 2},
                    expected_result="completed",
                ))
        if not candidates:
            candidates.append(PlanStep(
                goal="take another observation because no safe candidate was found",
                tool="computer", action={"action": "screenshot"},
                expected_result="needs_more_observation",
            ))
        if len(candidates) == 1:
            candidates.append(PlanStep(
                goal="take another observation instead",
                tool="computer", action={"action": "screenshot"},
                expected_result="needs_more_observation",
            ))
        return candidates[:16]

    def propose(self, context: TaskContext, observation: ScreenObservation) -> PlanStep:
        candidates = self._candidates(context, observation)
        options = [DecisionOption(id=f"candidate-{i}", description=c.goal)
                   for i, c in enumerate(candidates)]
        decision = self.semantic_engine.decide(
            {"task": context.task, "screen": observation.model_dump(), "step": context.step},
            "Which single candidate computer action should execute next?",
            options,
            f"candidate-{context.step}",
        )
        index = int(decision.winner_id.rsplit("-", 1)[-1])
        if not 0 <= index < len(candidates):
            raise ValueError(f"SemIf selected unknown candidate: {decision.winner_id}")
        return candidates[index]
