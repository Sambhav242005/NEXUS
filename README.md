# NEXUS — Neural Execution & eXpressive User System

Local-first, voice-controlled computer-use AI agent.

Voice + text share same agent backend.

## What it does

User speaks or types intent, e.g.:

> "Open Chrome, search for latest NVIDIA driver, tell me what you find."

Pipeline:

```text
Mic → VAD → ASR → Planner → SemIf gate → Safety → Execute one action
→ Screenshot → Verify → Respond → TTS
```

Bounded loop. Max steps. No infinite retry.

## Architecture principle

Three separated brains:

| Component | Role |
|---|---|
| SemIf + candidate planner | Typed action selection, gating, and task-state decisions |
| SemIf (semantic decision engine) | Fast routing / gating: tool routing, allow/ask/reject, task-state, clarification. Binary / multi-class only. |
| Vision model / CV pipeline | Screen understanding, UI detection, visual verification |

SemIf option probabilities are conditional scores, not calibrated safety confidence.
Safety policy enforced independently of LLM and SemIf. Deny-by-default.

## Repo status

Greenfield. `C:\Users\NPC\Desktop\NEXUS` empty at audit (2026-09-20).
Python 3.14.6. Windows 11 Pro. RTX 4070 Ti 12GB.
Ollama present: `minicpm-v4.6`, `quaternion/minicpm5:2b`, `qwen3-embedding:4b`.
No `semif` package installed. No existing voice/vision/computer code.

See `SPEC.md` for full spec, `PROJECT_STRUCTURE.md` for layout, `AGENTS.md` for agent workflow.

## Target layout

```text
nexus/
├── pyproject.toml
├── README.md / SPEC.md / AGENTS.md / PROJECT_STRUCTURE.md
├── .env.example
├── configs/default.yaml
├── configs/models.yaml
├── src/nexus/
│   ├── main.py
│   ├── agent/ (loop, planner, state, task_manager, schemas)
│   ├── semif/ (adapter, decisions, gates, schemas)
│   ├── voice/ (audio_loop, vad, asr, tts, schemas)
│   ├── computer/ (controller, screenshot, mouse, keyboard, schemas)
│   ├── vision/ (screen_parser, element_locator, schemas)
│   ├── safety/ (policy, confirmation, audit)
│   ├── tools/ (registry, browser, filesystem, shell)
│   ├── config/loader.py
│   └── observability/ (logging, metrics, tracing)
└── tests/unit + integration + safety
```

## Model / provider config

Switch providers via config, no hardcoding. Example `configs/models.yaml`:

```yaml
models:
  semif: {provider: semif, model: configurable-model}
  asr: {provider: whisper, model: base}
  tts: {provider: local, model: configurable-model}
  vision: {provider: configurable, model: minicpm-v4.6}
```

Paths / settings via env or YAML. No hardcoded secrets or model paths.
Configurable GPU/CPU placement. Never load multiple large models to GPU without VRAM check (12GB budget).

## Run (target, not yet implemented)

```powershell
# text mode (no mic needed, for tests)
python -m nexus.main --mode text --config configs/default.yaml

# voice mode (press Enter to stop recording)
python -m nexus.main --mode voice --autonomous --enable-computer-use --config configs/default.yaml

# debug
python -m nexus.main --mode text --debug
```

Computer use requires explicit enable + confirmation policy. Mock controller for unit tests (no real mouse/mic/model).

Voice models: STT uses `openai/whisper-large-v3-turbo` with CUDA and CPU fallback. TTS uses the local
`KittenML/kitten-tts-mini-0.8` model with the `Jasper` voice on CPU. Install the optional voice dependencies with
`python -m pip install -e ".[voice]"`. KittenTTS 0.8.1 currently requires Python below 3.13 because one
of its dependencies does not publish Python 3.13/3.14 wheels. On Python 3.11, install it with
`python -m pip install -e ".[voice,voice-kitten]"`. The KittenTTS model is downloaded from Hugging Face on first use.
The model, voice, speed, and device are configurable in `configs/models.yaml`.

Voice endpointing checks 100 ms microphone chunks. After speech is detected, one second of silence finalizes
the utterance. NEXUS transcribes only the finalized WAV; empty transcription is discarded and cannot reach
Gemma or the computer-use loop. Voice mode continues listening after each response; say `exit`, `quit`, or
`stop listening` to end the session.

## Real Windows mouse and keyboard

The Task Recorder's Execute button is the explicit opt-in for real desktop input. It now uses the Windows `user32` input API for mouse movement, clicks, double-clicks, scrolling, typing, keypresses, and hotkeys. Screenshots use Pillow's `ImageGrab`.

The controller stays mock-only when constructed without `enabled=True`, so unit tests never move the real pointer. SemIf selects from deterministic typed action candidates; the vision model only describes the current screen.

Autonomous local-model mode:

```powershell
python -m nexus.main --mode text --task "open notepad and type hello" --autonomous --enable-computer-use
```

The default autonomous path uses Windows UI Automation and does not require a vision model. SemIf selects typed action candidates, gates them, and classifies progress. Set `vision.provider: ollama` only for apps that lack usable UI Automation support.

Install the optional Windows UI Automation backend for authoritative control rectangles:

```powershell
python -m pip install -e ".[computer]"
```

If UI Automation cannot find a target rectangle, NEXUS refuses to click it rather than using coordinates invented by the vision model.

## Virtual mouse and UI gates

```powershell
python launch_ui.py --virtual-mouse
python launch_ui.py --workdir C:\projects --virtual-mouse
```

`--virtual-mouse` opens a screen-map panel (primary screen only) alongside the recorder. Clicking the map drives the real pointer through the validated controller, so schema + screen-bounds checks and the safety policy apply unchanged.

| Input | Action |
|---|---|
| Single click on map | `move` pointer to that screen point |
| Double-click on map | `click` |
| Right-click on map | `right_click` (context menu) |
| Wheel up / down on map | `scroll` at that point |
| `Save (Ctrl+S)` button | `save` (Ctrl+S) to focused window — confirm dialog |
| `Close window` button | `close_window` on focused window — confirm dialog |
| `Type` field + button | `type` text into focused window |
| `Enter` button | `keypress enter` |

Coordinates are physical screen pixels, origin top-left; the map scales 1:1 to the primary monitor. Gated buttons (`Save`, `Close window`) require an explicit confirm dialog before dispatch — no autonomous emission.

## Safety

Confirmation required: send message/email, purchase, delete, destructive shell, security change, private-data upload, irreversible external action.
Unsupported / ambiguous high-impact = deny.
Preview + confirm + cancel + audit log. Session permissions.
Never claim success without executor confirmation.

## Tests

```powershell
pytest tests/unit -q
pytest tests/integration -q
pytest tests/safety -q
```

Unit: action validation, decision schemas, SemIf adapter, policy, state, planner output, provider interfaces (mocked).
Safety: destructive needs confirm, malformed rejected, bad coords rejected, planner cannot bypass, failures reported truthfully.

## Docs for agents

- `SPEC.md` — build spec (voice, computer, vision, SemIf, planner, safety, loop, observability, stages)
- `PROJECT_STRUCTURE.md` — modules, interfaces, integration points
- `AGENTS.md` — workflow rules for coding agents
