# AGENTS.md — NEXUS contributor rules

For AI coding agents (Codex, Claude, Copilot, etc.) working in this repo. Humans: see `README.md`, `SPEC.md`.

## 1. Read first

1. `README.md` → status + run
2. `SPEC.md` → behavior source of truth
3. `PROJECT_STRUCTURE.md` → modules + interfaces
4. `configs/` + `src/nexus/` relevant file only (one at a time)
5. Tests for any module touched

Empty repo at audit — do not assume SemIf/voice/vision code exists. Verify with `pip show`, file read before import.

## 2. Architecture laws

- Planner reasons. SemIf gates (routing, allow/ask/reject, retry, task-state). Vision observes. Safety decides independently.
- Never use SemIf raw probability as safety proof.
- Voice and text share backend. TTS independent from planner.
- One action per loop step. Screenshot-before/after. Verify before claiming success.

## 3. Workflow (staged, no skipping)

Stage order: schemas → SemIf adapter → computer (mock) → planner → voice → vision → safety/observability → integration.
One logical module per change. Tests with each stage. `pytest tests/unit tests/safety -q` before next module.

## 4. Code rules

- Type hints + Pydantic validation on all boundaries (actions, decisions, plans).
- Protocols + dependency injection for providers (planner, ASR, TTS, vision, SemIf). No hardcoded Ollama/paths/secrets. Config via YAML/env.
- No global mutable state. Focused functions. Async only with concrete benefit. Cleanup streams/playback/handles. No silent `except: pass` — log + re-raise or typed error.
- Platform code isolated (`computer/`). Win+Linux where practical.
- VRAM-aware: configurable placement, no multi-large-model GPU load without check. CPU fallback.

## 5. Safety rules

- Destructive / irreversible / external / private-data actions → `requires_confirmation=True` + policy check + audit. Deny-by-default when ambiguous.
- Reject malformed actions, out-of-bounds coords. Document coordinate system in `computer/schemas.py`.
- Planner cannot bypass `safety/policy.py`. No false success.

## 6. Verification before done

- Run relevant tests, show command + result honestly.
- No real mouse/mic/model in unit tests — mocks + fixtures only.
- After edits: changed files, how to install/configure/run text + voice + computer-use, safety limits, test results, remaining work. Never claim unimplemented = done.

## 7. Response style in this repo

- Chat: terse (caveman mode per root AGENTS.md) — substance only.
- Files/commits/PRs: normal prose. No emojis unless asked.

## 8. Stop + ask

Missing SemIf source, model choice, real-input permission, credential/API request → stop, ask user. Never invent secrets, never hardcode paths.
