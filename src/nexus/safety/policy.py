"""Safety policy — independent of LLM and SemIf. Deny-by-default."""
from __future__ import annotations
from typing import Literal

DESTRUCTIVE_SHELL_HINTS = ("rm -rf", "del /", "format ", "mkfs", ":(){:|:&};:", "shutdown", "rd /s")

PolicyVerdict = Literal["allow", "confirm", "deny"]


def check_action(tool: str, action: dict, requires_confirmation: bool = False) -> PolicyVerdict:
    text = str(action).lower()
    destructive = (
        tool in ("shell", "filesystem")
        and any(h in text for h in ("delete", "remove", "rm ", "del ", "format", "overwrite"))
    ) or any(h in text for h in DESTRUCTIVE_SHELL_HINTS)
    external_irreversible = tool in ("browser",) and any(
        h in text for h in ("send", "purchase", "pay", "upload", "post", "email")
    )
    window_state_change = tool == "computer" and action.get("action") in {"close_window", "save"}
    if destructive or external_irreversible or window_state_change:
        return "confirm"
    if requires_confirmation:
        return "confirm"
    if tool not in ("browser", "computer", "filesystem", "shell", "response", "clarification"):
        return "deny"  # unknown tool -> deny-by-default
    return "allow"
