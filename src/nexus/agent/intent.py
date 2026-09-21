"""Small local intent router. Output is one fixed label, never executable JSON."""
from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pydantic import BaseModel, Field


ALLOWED_INTENTS = {"open_app", "computer_task", "shell_task", "clarify"}
ALLOWED_STEPS = {"open_app", "run_command", "computer_task", "clarify"}


class IntentStep(BaseModel):
    kind: str
    value: str = Field(min_length=1)


def parse_intent_plan(text: str) -> list[IntentStep]:
    """Parse Gemma's line/dot DSL; never parse arbitrary generated JSON."""
    labels = r"open_app|run_command|computer_task|clarify"
    pattern = re.compile(
        rf"(?P<kind>{labels})\s*:\s*(?P<value>.*?)(?=(?:\.|\r?\n)\s*(?:{labels})\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    steps = []
    for match in pattern.finditer(text):
        value = re.sub(r"\s+", " ", match.group("value")).strip(" .\t\r\n")
        kind = match.group("kind").lower()
        terminal_open = re.fullmatch(
            r"(?:open|launch|start)\s+(?:a|the)\s+(terminal|powershell|cmd)",
            value, re.IGNORECASE,
        )
        if kind == "run_command" and terminal_open:
            kind, value = "open_app", terminal_open.group(1).lower()
        elif kind == "computer_task":
            command = re.fullmatch(r"run\s+(.+?)(?:\s+command)?", value, re.IGNORECASE)
            if command:
                kind, value = "run_command", command.group(1).strip()
        if kind == "run_command" and os.name == "nt":
            value = re.sub(r"^(?:/bin/)?(?:bash|sh)(?:\s+-c)?\s+", "", value, flags=re.IGNORECASE)
            value = value.strip(" '\"")
        # Small models sometimes repeat DSL examples instead of filling them.
        # Treat those values as missing rather than executable intent.
        if value.lower() in {
            "application name", "app name", "your application",
            "exact command text", "short gui goal", "question",
        }:
            continue
        if value:
            steps.append(IntentStep(kind=kind, value=value))
    return steps


def repair_intent_plan(request: str, steps: list[IntentStep]) -> list[IntentStep]:
    """Repair common small-model mislabels using the original request.

    The planner is allowed to propose labels, but it is not allowed to turn
    UI operations such as "new tab" into shell commands.  Keep this repair
    deterministic and narrow; executable commands still come from the request.
    """
    repaired: list[IntentStep] = []
    for step in steps:
        value = step.value.strip()
        if step.kind == "run_command" and re.fullmatch(
                r"(?:open\s+)?(?:a\s+)?new\s+tab", value, re.IGNORECASE):
            repaired.append(IntentStep(kind="computer_task", value="open a new terminal tab"))
            continue
        if step.kind == "clarify" and value.lower().startswith("replace every placeholder"):
            continue
        repaired.append(step)

    lowered = request.lower()
    if re.search(r"\bnew\s+tab\b", lowered) and not any(
            s.kind == "computer_task" and "tab" in s.value.lower() for s in repaired):
        insert_at = next((i for i, s in enumerate(repaired) if s.kind == "run_command"), len(repaired))
        repaired.insert(insert_at, IntentStep(kind="computer_task", value="open a new terminal tab"))

    # Preserve an explicit WSL request even when a small model emits it as
    # part of a malformed sentence or omits it after inventing "new tab".
    if re.search(r"\brun\s+wsl\b", lowered) and not any(
            s.kind == "run_command" and s.value.lower() == "wsl" for s in repaired):
        repaired.append(IntentStep(kind="run_command", value="wsl"))
    return repaired


class OllamaIntentRouter:
    def __init__(self, model: str = "gemma3:1b", host: str = "http://127.0.0.1:11434",
                 timeout: float = 30.0, think: bool = False):
        self.model, self.host, self.timeout, self.think = model, host.rstrip("/"), timeout, think

    def classify(self, task: str) -> str:
        steps = self.plan(task)
        if steps:
            if steps[0].kind == "open_app":
                return "open_app"
            if any(step.kind == "run_command" for step in steps):
                return "shell_task"
            if steps[0].kind == "computer_task":
                return "computer_task"
        return "clarify"

    def plan(self, task: str) -> list[IntentStep]:
        prompt = (
            "Convert the request into a short sequence of intent lines. Output only lines using these forms:\n"
            "open_app: application name\nrun_command: exact command text\n"
            "computer_task: short GUI goal\nclarify: question\n"
            "Replace every placeholder with the actual value from the request. "
            "Do not output JSON, bullets, explanations, coordinates, or alternate commands. "
            "Preserve command text after run_command exactly.\n"
            f"Request: {task}"
        )
        payload = {"model": self.model, "stream": False,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": {"temperature": 0}, "think": self.think}
        request = Request(f"{self.host}/api/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(f"intent model {self.model} timed out") from exc
        except (URLError, HTTPError) as exc:
            raise RuntimeError(f"cannot reach Ollama for intent routing: {exc}") from exc
        answer = body.get("message", {}).get("content", "").strip()
        return repair_intent_plan(task, parse_intent_plan(answer))
