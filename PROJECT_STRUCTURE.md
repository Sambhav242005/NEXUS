# PROJECT_STRUCTURE.md — NEXUS modules

Target layout. Adapt to existing code, never duplicate modules. Status at audit: nothing implemented, empty dir.

## Tree

```text
nexus/
├── pyproject.toml
├── README.md / SPEC.md / AGENTS.md / PROJECT_STRUCTURE.md
├── .env.example
├── configs/
│   ├── default.yaml      # loop limits, timeouts, safety flags, debug
│   └── models.yaml       # planner/semif/asr/tts/vision provider+model+device
├── src/nexus/
│   ├── __init__.py
│   ├── main.py           # CLI: --mode text|voice --config --debug
│   ├── agent/
│   │   ├── loop.py       # bounded loop: plan→gate→safety→exec→observe→verify
│   │   ├── planner.py    # LLM adapter, PlanStep generation, replan
│   │   ├── state.py      # TaskState, history, cancellation
│   │   ├── task_manager.py # timeouts, retries, persistence
│   │   └── schemas.py    # PlanStep, TaskContext, ActionProposal
│   ├── semif/
│   │   ├── adapter.py    # SemanticDecisionEngine impl, timeouts, logging
│   │   ├── decisions.py  # routing/gating/progress/clarification option sets
│   │   ├── gates.py      # threshold + policy helpers (not safety proof)
│   │   └── schemas.py    # DecisionOption, DecisionResult
│   ├── voice/
│   │   ├── audio_loop.py # mic capture orchestration
│   │   ├── vad.py        # threshold, min-duration, silence-timeout
│   │   ├── asr.py        # SpeechRecognizer impl (Whisper-compatible)
│   │   ├── tts.py        # SpeechSynthesizer impl, queue, interrupt
│   │   └── schemas.py    # AudioInput, Transcription, AudioOutput
│   ├── computer/
│   │   ├── controller.py # validated dispatch, before/after screenshots
│   │   ├── screenshot.py # capture, dims, active window
│   │   ├── mouse.py      # move/click/dclick/scroll
│   │   ├── keyboard.py   # type/keypress/hotkey
│   │   └── schemas.py    # ComputerAction, coordinate-system doc
│   ├── vision/
│   │   ├── screen_parser.py   # screenshot → ScreenObservation
│   │   ├── element_locator.py # semantic/a11y-first, pixel fallback
│   │   └── schemas.py         # ImageInput, ScreenObservation
│   ├── safety/
│   │   ├── policy.py       # allow/confirm/deny, deny-by-default
│   │   ├── confirmation.py # preview, ask, cancel, session perms
│   │   └── audit.py        # append-only action log
│   ├── tools/
│   │   ├── registry.py   # tool names: browser/computer/filesystem/shell/response
│   │   ├── browser.py
│   │   ├── filesystem.py
│   │   └── shell.py      # destructive-cmd detection
│   ├── config/
│   │   └── loader.py     # YAML + env, validation, defaults
│   └── observability/
│       ├── logging.py    # session/task IDs, decisions, latency
│       ├── metrics.py
│       └── tracing.py
└── tests/
    ├── unit/         # schemas, adapter, policy, state, planner output (mocked)
    ├── integration/  # text→plan→gate, mock exec, confirm/retry/cancel
    ├── safety/       # destructive-confirm, malformed/coords reject, no-bypass
    └── fixtures/
```

## Key interfaces (see SPEC.md for full defs)

- `SpeechRecognizer.transcribe(audio) → Transcription`
- `SpeechSynthesizer.synthesize(text) → AudioOutput`
- `SemanticDecisionEngine.decide(state, question, options) → DecisionResult`
- `ScreenUnderstandingProvider.analyze(screenshot, context) → ScreenObservation`
- `ComputerAction{action, x?, y?, text?, key?, keys?}` — validated pre-exec
- `PlanStep{goal, tool, action, expected_result, requires_confirmation}`

## Integration points

```text
main.py → config/loader → agent/loop
loop → agent/planner → semif/adapter → safety/policy → tools/registry
tools/computer → computer/controller → vision/* → loop (verify)
voice/* → loop (input) / loop → voice/tts (output)
all → observability/logging + safety/audit
```

## Build order

schemas → semif adapter → computer mock → planner → voice → vision → safety/obs → integration. Tests each stage.

## Risks

- SemIf source missing → stub `SemanticDecisionEngine` protocol + mock adapter first, swap real impl later.
- VRAM (12GB) → one GPU model at a time default; device selectable per provider in `models.yaml`.
- PyAutoGUI/MSS mic/screenshot perms on Win11 → isolate in `computer/`, mock in tests.
- Python 3.14 compat → pin deps in `pyproject.toml`, verify whisper/TTS wheels before locking.
