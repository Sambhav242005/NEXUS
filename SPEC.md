# NEXUS SPEC

Source: project objective brief (2026-09-20). Greenfield build. This file is source of truth for behavior. `README.md` = overview, `PROJECT_STRUCTURE.md` = modules, `AGENTS.md` = agent workflow.

## 1. Objective

Local AI agent. Voice + text share same backend.

Input → transcribe → task state → plan → gate → safety → one action → observe → verify → respond → TTS → continue conversation.

## 2. Role separation (hard rule)

- **Planner LLM**: planning, reasoning, decomposition, tool-action generation, complex intent, result summary.
- **SemIf**: fast semantic decisions only — tool routing, action gating, confirmation triage, retry triage, task-state classification, action validation. Never primary reasoner / computer-use model.
- **Vision**: screen understanding, UI detection, screenshot analysis, visual verification.
- **Safety layer**: independent policy checks. SemIf scores ≠ safety proof.

## 3. Voice pipeline

```text
Mic → VAD → capture → local ASR → agent → response → local TTS → playback
```

### ASR interface

```python
class SpeechRecognizer(Protocol):
    def transcribe(self, audio: AudioInput) -> Transcription: ...
```

Local Whisper-compatible. Configurable size, CPU/GPU. Handle mic error, silence, empty segments (never send empty to ASR). Timestamps where supported. Avoid blocking agent loop.

### VAD

Configurable threshold, min speech duration, silence timeout, start/stop detection, stream cleanup.

### TTS interface

```python
class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str) -> AudioOutput: ...
```

Local provider. Configurable voice/model. Playback queue. Interruptible. Graceful errors. No mandatory external API. Independent from planner.

### Barge-in

Reply/ack playback is async with a mic + Enter monitor. Sustained mic
level above `barge_in.threshold` for `barge_in.min_speech_ms` (or Enter)
purges playback; the loop records fresh immediately (cut speech is not
captured). Monitor failure never truncates a reply — it plays through.

### Voice server transports

REST (file): `POST /v1/stt` multipart 16 kHz mono WAV → `{"text"}`;
`POST /v1/tts` JSON `{text, voice?, speed?}` → `audio/wav`; `GET /health`.

WebSocket (streaming): `/ws/stt` binary int16 16 kHz mono frames in →
`speech_started` event + `final` transcript (server VAD auto-endpoints
after trailing silence; `stop`/`abort` controls; multi-utterance per
connection); `/ws/tts` JSON `{text, voice?, speed?}` in → `meta` +
binary chunks + `done`. Same local backends; separately hostable.

## 4. Computer-use

Platform abstraction. Capabilities: screenshot, move, click, double-click, type, keypress, hotkey, scroll, screen dims, active window (where supported).

```python
class ComputerAction(BaseModel):
    action: Literal["click","double_click","type","keypress","hotkey","scroll","move","screenshot"]
    x: int | None = None
    y: int | None = None
    text: str | None = None
    key: str | None = None
    keys: list[str] | None = None
```

Validate before execute. Reject malformed / out-of-bounds coords. One coordinate system — document physical vs logical vs normalized. Screenshot-before + screenshot-after workflow.

## 5. Vision

```python
class ScreenUnderstandingProvider(Protocol):
    def analyze(self, screenshot: ImageInput, task_context: TaskContext) -> ScreenObservation: ...
```

Observation: visible app, UI elements + approx locations, text, interaction targets, confidence, uncertainty. Coords never assumed exact. Prefer a11y/semantic targeting over pixels. Verify after each action (before/after description).

## 6. SemIf integration

Source: `https://github.com/TheoLeeCJ/SemIf` (MIT, `semif-phase1` 0.1.0, Python >=3.10).
Clean adapter, no fork of SemIf internals unless justified. NEXUS depends on protocol only; real backend optional.

```python
class SemanticDecisionEngine(Protocol):
    def decide(self, state: dict, question: str, options: list[DecisionOption]) -> DecisionResult: ...
```

SemIf native mapping (from `src/semif_phase1/`):

- Row format: `{id, state: str|dict|list (nonempty, JSON-finite), question: str, options: [{id, description}]}` — 2–16 options, unique ids. See `core.validate_row`.
- Scoring: `direct.score(model, tokenizer, row, metadata)` — one forward pass, native full-vocab last-position logits restricted to answer slots A–P, softmax → `probabilities`. Returns `option_ids, option_logits, probabilities, input_tokens, forward_seconds, total_seconds, prompt_sha256, model{source, revision, dtype, torch_version, transformers_version}, probability_status="conditional option score; uncalibrated as decision confidence"`.
- Modes: `direct` (one row), `serial` (prefix reuse, same state), `shared` (prefill once, parallel criteria), `reranker` (retrieval only, not for general decisions). CLI: `semif-score --mode direct --model Qwen/Qwen3.5-4B --revision <40-char-pin> --input decisions.jsonl --output results.jsonl`. Requires exactly one visible CUDA GPU (`CUDA_VISIBLE_DEVICES=0`).
- NEXUS adapter rules: pin model revision for remote models; hash prompts; log `prompt_sha256`, timings, model metadata; enforce decision timeout; treat `probabilities` as conditional scores, never safety confidence.

VRAM conflict (RTX 4070 Ti 12GB): SemIf 4B BF16 (~8GB) + Ollama planner `qwen3:4b` (~2.5GB) + vision `minicpm-v4.6` (~1.6GB) exceeds budget if concurrent. Adapter must support sequential placement: score → release, or subprocess with exclusive GPU. Default dev backend = mock `SemanticDecisionEngine` with deterministic fixtures; real SemIf backend behind `semif.backend: mock|torch-direct|subprocess-cli` config.

Decision sets:

- **Tool routing**: `browser | computer | filesystem | shell | response | clarification`
- **Action gating**: `allow | ask_confirmation | reject`
- **Task progress**: `completed | failed | needs_more_observation | retry | replan`
- **Clarification**: `continue | ask_user`

## 7. Planner

Local LLM. Receives intent + state → structured plan → proposes one action at a time → consumes results → updates plan → stops on done / unsafe.

```python
class PlanStep(BaseModel):
    goal: str
    tool: str
    action: dict
    expected_result: str
    requires_confirmation: bool
```

Explicit iteration limit. Structured outputs. Cannot bypass safety.

## 8. Safety

Independent from LLM/SemIf. Deny-by-default on unsupported/ambiguous high-impact.

Confirm: send message/email, purchase, delete, destructive shell, security change, private upload, irreversible external.
Provide preview, confirm, cancel, audit log, configurable policy, session permissions.
Never report success without executor confirmation.

## 9. Execution loop

```text
INPUT → TRANSCRIBE → TASK STATE → PLANNER → SEMIF → SAFETY → CONFIRM?
→ EXECUTE ONE → OBSERVE → VERIFY → CONTINUE/REPLAN/STOP → RESPOND → TTS
```

Max steps/task, per-tool timeout, cancellation, manual interrupt, failure states, transient-error recovery, no infinite retry, persist useful state.

## 10. Providers / config

Interfaces for planner, ASR, TTS, vision, SemIf. Adapters, no hardcoded Ollama/llama.cpp/Transformers. Switch via `configs/models.yaml` + `configs/default.yaml` + env. See README example.

Resource rule: configurable placement, VRAM-aware (12GB RTX 4070 Ti target), CPU fallback where practical, Win+Linux where practical.

## 11. Observability

Structured logs: session/task ID, input metadata, planner + SemIf + safety decisions, tool proposal/status, latency, error type, verification. No sensitive data. No permanent audio/screenshot store unless configured. `--debug` mode.

## 12. Testing

- Unit: action validation, decision schemas, SemIf adapter, policy, state, planner output, voice/computer interfaces (mocked, no real HW/model).
- Integration: text→planner→SemIf, mock computer exec, screenshot→obs→planner, confirm/retry/cancel flows.
- Safety: destructive needs confirm, malformed/coords rejected, no bypass, truthful failure reporting.

## 13. Stages

1. Audit (done 2026-09-20: empty dir, see README) → architecture report
2. Core schemas + config + tests
3. SemIf adapter (direct scoring, errors)
4. Computer controller (mock first)
5. Planner (structured, bounded)
6. Voice (ASR/VAD/TTS)
7. Vision (analysis + verification)
8. Safety + observability
9. Integration + docs

No skipping tests to go faster.

## 14. Open questions (need user)

1. SemIf source? Resolved: `https://github.com/TheoLeeCJ/SemIf` (MIT). Remaining: confirm model pin (`Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`?) and install path (pip `-e` vs separate venv, torch 2.10.0 on Python 3.14 needs wheel check).
2. Planner model? `qwen3:4b-instruct` OK or other? Vision = `minicpm-v4.6`?
3. ASR/TTS impl? faster-whisper + Piper/Coqui? Constraints?
4. Real input execution allowed in dev, or mock-only until safety passes?
