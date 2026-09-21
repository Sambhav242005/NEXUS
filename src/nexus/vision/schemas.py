"""Validated screen observations shared by vision and planning."""
from __future__ import annotations

from pydantic import BaseModel, Field


class UIElement(BaseModel):
    label: str = ""
    role: str = "unknown"
    x: int | None = Field(default=None, ge=0)
    y: int | None = Field(default=None, ge=0)
    # Coordinates are authoritative only when source == "uia". Vision models
    # may describe a target without inventing a rectangle.
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    source: str = "vision"


class ScreenObservation(BaseModel):
    visible_app: str = "unknown"
    text: list[str] = Field(default_factory=list)
    elements: list[UIElement] = Field(default_factory=list)
    uncertainty: str = ""
    screenshot_path: str | None = None


class TaskContext(BaseModel):
    task: str = Field(min_length=1)
    step: int = Field(ge=0, default=0)
    previous_result: str = ""
