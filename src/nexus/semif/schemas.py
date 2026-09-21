"""NEXUS SemIf decision schemas (Pydantic boundaries)."""
from __future__ import annotations
from pydantic import BaseModel, Field


class DecisionOption(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class DecisionResult(BaseModel):
    decision_id: str
    winner_id: str
    probabilities: list[float]
    option_ids: list[str]
    option_logits: list[float] | None = None
    prompt_sha256: str | None = None
    prompt_version: str | None = None
    model_source: str | None = None
    model_revision: str | None = None
    forward_seconds: float | None = None
    total_seconds: float | None = None
    probability_status: str = "conditional option score; uncalibrated as decision confidence"
