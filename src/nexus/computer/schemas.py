"""Computer action schemas. Coordinates: physical pixels, origin top-left.
Validate before execution. Reject malformed / out-of-bounds."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, model_validator


class ComputerAction(BaseModel):
    action: Literal[
        "click", "double_click", "right_click", "middle_click", "drag", "type",
        "keypress", "hotkey", "scroll", "move", "resize", "close_window",
        "minimize_window", "maximize_window", "launch_app", "open_url",
        "open_file", "save", "select_all", "copy", "paste", "screenshot",
    ]
    x: int | None = None
    y: int | None = None
    text: str | None = None  # text for type; signed wheel delta for scroll
    key: str | None = None
    keys: list[str] | None = None
    x2: int | None = None
    y2: int | None = None
    width: int | None = None
    height: int | None = None

    @model_validator(mode="after")
    def check_fields(self):
        a = self.action
        if a in ("click", "double_click", "right_click", "middle_click", "move", "scroll", "drag", "resize"):
            if self.x is None or self.y is None:
                raise ValueError(f"{a} requires x and y")
            if self.x < 0 or self.y < 0 or self.x > 16384 or self.y > 16384:
                raise ValueError("coordinates out of bounds (0..16384)")
        if a == "drag":
            if self.x2 is None or self.y2 is None:
                raise ValueError("drag requires x2 and y2")
            if min(self.x2, self.y2) < 0 or max(self.x2, self.y2) > 16384:
                raise ValueError("drag destination out of bounds (0..16384)")
        if a == "resize" and (self.width is None or self.height is None or self.width <= 0 or self.height <= 0):
            raise ValueError("resize requires positive width and height")
        if a == "type" and not self.text:
            raise ValueError("type requires text")
        if a == "keypress" and not self.key:
            raise ValueError("keypress requires key")
        if a == "hotkey" and not self.keys:
            raise ValueError("hotkey requires keys")
        if a in ("launch_app", "open_url", "open_file") and not self.text:
            raise ValueError(f"{a} requires text")
        if a == "scroll":
            if self.x is None or self.y is None or self.text is None:
                raise ValueError("scroll requires x, y and text wheel delta")
            try:
                int(self.text)
            except ValueError as exc:
                raise ValueError("scroll text must be an integer wheel delta") from exc
        return self
