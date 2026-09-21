"""Agent schemas."""
from __future__ import annotations
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    goal: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    action: dict = Field(default_factory=dict)
    expected_result: str = Field(min_length=1)
    requires_confirmation: bool = False
