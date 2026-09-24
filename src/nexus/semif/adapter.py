"""NEXUS SemIf adapter — Real SemIf validation path + pluggable scorer.

Real path: imports `semif_phase1.core` (validate_row, direct_messages,
softmax, digest) from the real SemIf repo. No fork. Model forward pass
is injected as a scorer callable so:
- dev/tests: deterministic mock scorer (no GPU needed)
- prod CUDA: pass a scorer wrapping `semif_phase1.direct.score`
  with a loaded model/tokenizer (requires exactly 1 visible CUDA GPU,
  pinned model revision — see SemIf core.load_causal_model).

Probabilities are conditional option scores, never safety proof.
Safety enforced separately in safety/policy.py.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Callable, Protocol
from pydantic import BaseModel

from .schemas import DecisionOption, DecisionResult

# Allow real SemIf repo checkout without install (dev convenience).
# Prod: `pip install -e <SemIf>` then this path insert is harmless.
_SEMIF_SRC = Path(r"C:\Users\NPC\AppData\Local\Temp\opencode\SemIf\src")
if _SEMIF_SRC.exists() and str(_SEMIF_SRC) not in sys.path:
    sys.path.insert(0, str(_SEMIF_SRC))

try:
    from semif_phase1 import core as _semif_core
    _SEMIF_AVAILABLE = True
    _SEMIF_IMPORT_ERROR: str | None = None
except Exception as e:  # pragma: no cover - env dependent
    _semif_core = None  # type: ignore
    _SEMIF_AVAILABLE = False
    _SEMIF_IMPORT_ERROR = str(e)

SEMIF_AVAILABLE = _SEMIF_AVAILABLE
SEMIF_IMPORT_ERROR = _SEMIF_IMPORT_ERROR


class SemanticDecisionEngine(Protocol):
    def decide(self, state: dict | str | list, question: str,
               options: list[DecisionOption], decision_id: str = "d1") -> DecisionResult: ...


class SemIfConfig(BaseModel):
    model: str = "Qwen/Qwen3.5-4B"
    revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    backend: str = "mock"  # mock | torch-direct | subprocess-cli | laya
    timeout_seconds: float = 30.0
    max_tokens: int = 4096


ScorerFn = Callable[[dict], dict]
"""Takes SemIf row {id,state,question,options} -> dict with
option_ids, probabilities, option_logits?, prompt_sha256?, etc."""


def mock_scorer_first_wins(row: dict) -> dict:
    """Deterministic mock: first option wins with 0.7, rest split 0.3."""
    n = len(row["options"])
    rest = 0.3 / max(1, n - 1) if n > 1 else 0.0
    probs = [0.7] + [rest] * (n - 1) if n > 1 else [1.0]
    return {
        "option_ids": [o["id"] for o in row["options"]],
        "probabilities": probs,
        "option_logits": [2.0] + [0.0] * (n - 1),
        "prompt_version": "mock-v1",
    }


def mock_scorer_nexus(row: dict) -> dict:
    """Deterministic NEXUS fixture with sensible winners for each decision set."""
    question = row["question"].lower()
    preferred = "computer" if "which tool" in question else (
        "allow" if "execute" in question else "completed"
    )
    ids = [option["id"] for option in row["options"]]
    winner_index = ids.index(preferred) if preferred in ids else 0
    probabilities = [0.1] * len(ids)
    probabilities[winner_index] = 0.7
    remainder = 0.3 / max(1, len(ids) - 1)
    for index in range(len(ids)):
        if index != winner_index:
            probabilities[index] = remainder
    return {
        "option_ids": ids,
        "probabilities": probabilities,
        "option_logits": [2.0 if i == winner_index else 0.0 for i in range(len(ids))],
        "prompt_version": "nexus-mock-v1",
    }


class RealSemIfAdapter:
    """Uses real semif_phase1.core for validation + prompt hashing.

    Args:
        config: model/revision/backend/timeout settings.
        scorer: callable producing model scores. Default mock.
            For real CUDA: `functools.partial(real_direct_scorer, model, tokenizer, metadata)`.
    """

    def __init__(self, config: SemIfConfig | None = None, scorer: ScorerFn | None = None):
        self.config = config or SemIfConfig()
        self.scorer = scorer or mock_scorer_first_wins
        self._use_core = self.config.backend != "mock" and self.config.backend != "laya"
        if self._use_core and _semif_core is None:
            raise RuntimeError(f"Real SemIf core unavailable: {SEMIF_IMPORT_ERROR}")

    def decide(self, state: dict | str | list, question: str,
               options: list[DecisionOption], decision_id: str = "d1") -> DecisionResult:
        row = {
            "id": decision_id,
            "state": state if isinstance(state, (str, dict, list)) else str(state),
            "question": question,
            "options": [{"id": o.id, "description": o.description} for o in options],
        }
        # Real validation — raises ValueError on malformed rows.
        if self._use_core:
            _semif_core.validate_row(row)
        started = time.perf_counter()
        out = self.scorer(row)
        total = time.perf_counter() - started
        if total > self.config.timeout_seconds:
            raise TimeoutError(f"SemIf decision exceeded {self.config.timeout_seconds}s ({total:.2f}s)")
        option_ids = out["option_ids"]
        probs = out["probabilities"]
        if len(option_ids) != len(options) or len(probs) != len(options):
            raise ValueError("Scorer returned mismatched option count")
        winner = option_ids[int(max(range(len(probs)), key=lambda i: probs[i]))]
        # Real prompt hash when available (auditable).
        prompt_hash = out.get("prompt_sha256")
        if prompt_hash is None and self._use_core:
            try:
                msgs = _semif_core.direct_messages(row)
                prompt_hash = _semif_core.digest(msgs[1]["content"] + msgs[0]["content"])
            except Exception:
                prompt_hash = None
        return DecisionResult(
            decision_id=decision_id,
            winner_id=winner,
            probabilities=list(probs),
            option_ids=list(option_ids),
            option_logits=out.get("option_logits"),
            prompt_sha256=prompt_hash,
            prompt_version=out.get("prompt_version", "direct-options-v1"),
            model_source=self.config.model,
            model_revision=self.config.revision,
            forward_seconds=out.get("forward_seconds"),
            total_seconds=out.get("total_seconds", total),
        )


def real_direct_scorer(model, tokenizer, metadata: dict, max_tokens: int = 4096):
    """Build a ScorerFn wrapping real semif_phase1.direct.score (needs CUDA model)."""
    from semif_phase1 import direct as _direct

    def _score(row: dict) -> dict:
        return _direct.score(model, tokenizer, row, metadata, max_tokens)
    return _score


def laya_scorer(agent) -> ScorerFn:
    """Build a ScorerFn wrapping a loaded Laya agent.

    Converts SemIf row {id,state,question,options} into a Laya choice question,
    runs a single forward pass, and maps the result back to SemIf format.
    """
    def _score(row: dict) -> dict:
        state_text = row["state"]
        if isinstance(state_text, dict):
            state_text = str(state_text)
        elif isinstance(state_text, list):
            state_text = " ".join(str(x) for x in state_text)

        criteria = {o["id"]: o.get("description", o["id"]) for o in row["options"]}
        questions = {
            "decision": {
                "type": "choice",
                "instructions": row["question"],
                "criteria": criteria,
            }
        }
        result = agent.predict(state_text, questions)
        answers = result.get("answers", {})
        decision_answer = answers.get("decision", {})

        choice_id = decision_answer.get("choice", row["options"][0]["id"])
        choice_confidence = float(decision_answer.get("confidence", 0.5))

        option_ids = [o["id"] for o in row["options"]]
        probabilities = []
        for oid in option_ids:
            if oid == choice_id:
                probabilities.append(choice_confidence)
            else:
                probabilities.append((1.0 - choice_confidence) / max(1, len(option_ids) - 1))

        total = sum(probabilities) or 1.0
        probabilities = [p / total for p in probabilities]

        return {
            "option_ids": option_ids,
            "probabilities": probabilities,
            "option_logits": [2.0 if oid == choice_id else 0.0 for oid in option_ids],
            "prompt_version": "laya-v1",
            "forward_seconds": result.get("routing", {}).get("latency_ms", 0) / 1000.0,
        }
    return _score
