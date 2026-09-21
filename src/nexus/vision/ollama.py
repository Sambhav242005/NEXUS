"""Ollama-compatible local vision provider using the standard library."""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .schemas import ScreenObservation, TaskContext


_OBSERVATION_FORMAT = {
    "type": "object",
    "properties": {
        "visible_app": {"type": "string"},
        "text": {"type": "array", "items": {"type": "string"}},
        "elements": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "role": {"type": "string"},
            "confidence": {"type": "number"},
        }, "required": ["label", "role", "confidence"]}},
        "uncertainty": {"type": "string"},
    },
    "required": ["visible_app", "text", "elements", "uncertainty"],
}


class OllamaVisionProvider:
    def __init__(self, model: str = "minicpm-v4.6", host: str = "http://127.0.0.1:11434",
                 timeout: float = 180.0):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def analyze(self, screenshot: str | Path, context: TaskContext) -> ScreenObservation:
        path = Path(screenshot)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "Analyze this desktop screenshot for the requested task. Return JSON only. "
            "Identify visible targets by label and role. Do not invent coordinates or sizes; "
            "Windows UI Automation supplies authoritative rectangles separately. "
            "Use conservative confidence.\n"
            f"Task: {context.task}\nStep: {context.step}"
        )
        payload = {"model": self.model, "stream": False, "format": _OBSERVATION_FORMAT,
                   "messages": [{"role": "user", "content": prompt, "images": [encoded]}]}
        request = Request(f"{self.host}/api/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama vision timed out after {self.timeout:.0f}s for {self.model}. "
                "Warm the model with `ollama run minicpm-v4.6`, then retry; "
                "increase the provider timeout if the first vision load is slower."
            ) from exc
        except (URLError, HTTPError) as exc:
            raise RuntimeError(
                f"Cannot reach Ollama vision at {self.host}: {exc}. "
                "Start Ollama with `ollama serve` and verify `ollama list`."
            ) from exc
        content = body.get("message", {}).get("content", "")
        observation = ScreenObservation.model_validate(json.loads(content))
        return observation.model_copy(update={"screenshot_path": str(path)})
