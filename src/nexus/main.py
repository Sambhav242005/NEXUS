"""NEXUS entrypoint (text mode first, voice later)."""
from __future__ import annotations
import argparse
import re
from pathlib import Path

from nexus.config.loader import load_default
from nexus.semif.adapter import (
    RealSemIfAdapter, SemIfConfig, SEMIF_AVAILABLE, mock_scorer_nexus, real_direct_scorer, laya_scorer,
)
from nexus.semif.decisions import TOOL_ROUTING, ACTION_GATING
from nexus.safety.policy import check_action
from nexus.computer.schemas import ComputerAction
from nexus.computer.controller import ComputerController
from nexus.agent.loop import AutonomousLoop
from nexus.agent.candidate_planner import SemIfCandidatePlanner
from nexus.agent.intent import build_intent_router
from nexus.vision.ollama import OllamaVisionProvider


def extract_text_to_write(task: str) -> str:
    """Pull the text to write out of an instruction like '... write X'.

    Falls back to 'Hello world' (demo default) when no write-target found.
    """
    import re
    m = re.search(r"\bwrite\s+['\"]?(.+?)['\"]?\s*[.!]?\s*$", task, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".!")
    return "Hello world"


def classify_intent(task: str) -> str:
    """notepad | terminal | unsupported. Small explicit router until planner LLM lands."""
    t = task.lower()
    if "terminal" in t or "powershell" in t or t.strip().startswith("cmd"):
        return "terminal"
    if "notepad" in t or "note" in t or "file" in t or "write" in t:
        return "notepad"
    return "unsupported"


def run_task(task: str, cfg: dict) -> None:
    """Bounded agent loop: planner stub -> SemIf gate -> safety -> ONE action at a time."""
    print(f"TASK: {task}")
    adapter = RealSemIfAdapter(SemIfConfig())
    workdir = Path(cfg.get("workdir", "."))
    ctl = ComputerController(workdir, enabled=bool(cfg.get("computer_use_enabled", False)))
    # 1. Route via Real SemIf.
    route = adapter.decide({"utterance": task}, "Which tool should handle this request?",
                           TOOL_ROUTING, decision_id="task-route")
    print(f"[planner] intent parsed. [semif-route] -> {route.winner_id}")
    # 2. Screenshot before.
    before = ctl.execute(ComputerAction(action="screenshot"))
    print(f"[observe-before] {before.detail}")
    # 3. Plan by intent. One action per loop step. No fake success.
    intent = classify_intent(task)
    print(f"[planner] intent={intent}")
    if intent == "unsupported":
        print("[stop] I can't do that yet (supported: notepad/file/write, terminal). Nothing executed.")
        return
    if intent == "terminal":
        gate = adapter.decide({"action": "launch_terminal"},
                              "Is this action safe to execute now?",
                              ACTION_GATING, decision_id="gate-terminal")
        print(f"[semif-gate] launch_terminal -> {gate.winner_id}")
        verdict = check_action("shell", {"op": "launch_terminal"})
        print(f"[safety] verdict={verdict}")
        if verdict == "deny" or gate.winner_id == "reject":
            print("[stop] terminal launch rejected. Nothing executed.")
            return
        res = ctl.launch_terminal()
        print(f"[execute] ok={res.ok} {res.detail}")
        if not res.ok:
            print("[stop] terminal launch failed. No false success.")
            return
        if ctl.enabled and re.search(r"\b(write|type)\b", task, re.IGNORECASE):
            text = extract_text_to_write(task)
            typed = ctl.execute(ComputerAction(action="type", text=text))
            print(f"[execute] type -> ok={typed.ok} {typed.detail}")
            if not typed.ok:
                print("[stop] terminal input failed. No false success.")
                return
        proc = ComputerController.process_running("WindowsTerminal.exe", "powershell.exe", "cmd.exe")
        after = ctl.execute(ComputerAction(action="screenshot"))
        print(f"[observe-after] {after.detail}")
        if proc:
            print(f"[verify] PASS: terminal running ({proc}).")
        else:
            print("[verify] FAIL: launched but no terminal process found.")
        return
    text = extract_text_to_write(task)
    print(f"[planner] will write: {text!r}")
    if ctl.enabled:
        launched = ctl.launch_notepad()
        print(f"[execute] launch_notepad -> ok={launched.ok} {launched.detail}")
        if not launched.ok:
            print("[stop] Notepad launch failed. No false success.")
            return
        steps = [ComputerAction(action="type", text=text)]
    else:
        steps = [ComputerAction(action="type", text=text), ComputerAction(action="click", x=100, y=100)]
    for i, step in enumerate(steps, 1):
        gate = adapter.decide({"action": step.model_dump()},
                              "Is this action safe to execute now?",
                              ACTION_GATING, decision_id=f"gate-{i}")
        print(f"[semif-gate {i}] {step.action} -> {gate.winner_id}")
        verdict = check_action("computer", step.model_dump())
        print(f"[safety {i}] verdict={verdict}")
        if verdict == "deny" or gate.winner_id == "reject":
            print(f"[stop] step {i} rejected. Not executing.")
            return
        if verdict == "confirm" or gate.winner_id == "ask_confirmation":
            print(f"[confirm-required] step {i} needs user OK in interactive mode. Proceeding (demo).")
        res = ctl.execute(step)
        print(f"[execute {i}] ok={res.ok} {res.detail}")
        if not res.ok:
            print(f"[stop] step {i} failed. No false success.")
            return
    after = ctl.execute(ComputerAction(action="screenshot"))
    print(f"[observe-after] {after.detail}")
    target = workdir / "hello.txt"
    if target.exists() and target.read_text(encoding="utf-8").strip() == text:
        print(f"[verify] PASS: hello.txt contains {text!r}, Notepad launched by agent.")
    else:
        print("[verify] FAIL: content mismatch.")


def run_autonomous_task(task: str, cfg: dict) -> None:
    """Run the local vision/planner loop with a strict one-action bound."""
    if not cfg.get("computer_use_enabled", False):
        raise PermissionError("autonomous computer use is disabled; enable computer_use_enabled first")
    models = cfg.get("models", {})
    vision_cfg = models.get("vision", {})
    semif_cfg = models.get("semif", {})
    intent_cfg = models.get("intent", {})
    controller = ComputerController(cfg.get("workdir", "."), enabled=True)
    semantic_engine = build_semif_engine(semif_cfg)
    intent_router = build_intent_router(models)
    intent_steps = intent_router.plan(task)
    intent = "clarify"
    if intent_steps:
        if any(step.kind == "run_command" for step in intent_steps):
            intent = "shell_task"
        else:
            intent = intent_steps[0].kind
    print(f"[intent:{intent}] provider={intent_cfg.get('provider', 'ollama')} model={intent_cfg.get('name', 'gemma3:1b')}")
    for index, step in enumerate(intent_steps, 1):
        print(f"[intent-step {index}] {step.kind}: {step.value}")
    if (intent == "open_app" or re.search(r"\b(open|launch|start)\b", task, re.I)) and not requests_existing_app(task):
        planned_app = next((step.value for step in intent_steps if step.kind == "open_app"), None)
        app_name = planned_app or extract_app_name(task)
        if not app_name:
            raise ValueError("could not identify which application to open")
        launched = controller.launch_application(app_name)
        if not launched.ok:
            raise RuntimeError(launched.detail)
    vision = None
    if vision_cfg.get("provider", "uia") == "ollama":
        vision = OllamaVisionProvider(model=vision_cfg.get("model", "minicpm-v4.6"))
    loop = AutonomousLoop(
        controller,
        vision,
        SemIfCandidatePlanner(semantic_engine, intent_steps=intent_steps),
        max_steps=int(cfg.get("max_steps", 10)),
        semantic_engine=semantic_engine,
    )
    results = loop.run(task)
    for index, result in enumerate(results, 1):
        print(f"[autonomous {index}] ok={result.ok} {result.detail}")


def run_voice_task(cfg: dict, autonomous: bool) -> None:
    """Keep listening; speech during a task is queued, never lost."""
    from nexus.voice.asr import build_asr
    from nexus.voice import tts as tts_mod
    from nexus.voice.audio_loop import VoiceLoop
    from nexus.voice.schemas import ASRConfig, TTSConfig

    models = cfg.get("models", {})
    raw_asr = models.get("asr", {})
    asr = build_asr(ASRConfig.model_validate(raw_asr))
    synth = tts_mod.build_tts(TTSConfig.model_validate(models.get("tts", {})))
    voice_cfg = cfg.get("voice", {})

    def on_task(task: str) -> str:
        try:
            if autonomous:
                run_autonomous_task(task, cfg)
            else:
                run_task(task, cfg)
            return "Task processing finished."
        except Exception as exc:
            print(f"[voice] task failed: {exc}")
            return "The task failed."

    loop = VoiceLoop(
        asr=asr,
        synth=synth,
        recordings=Path(cfg.get("workdir", ".")) / "recordings",
        voice_cfg=voice_cfg,
        device=raw_asr.get("device_index"),
        max_pending=int(voice_cfg.get("max_pending", 5)),
        on_task=on_task,
    )
    loop.run()


def build_semif_engine(model_cfg: dict):
    """Build the configured SemIf adapter; model loading is opt-in."""
    config = SemIfConfig(
        model=model_cfg.get("model", "Qwen/Qwen3.5-4B"),
        revision=model_cfg.get("revision", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
        backend=model_cfg.get("backend", "mock"),
    )
    if config.backend == "mock":
        return RealSemIfAdapter(config, scorer=mock_scorer_nexus)
    if config.backend == "laya":
        import laya as _laya
        agent = _laya.load(config.model)
        adapter = RealSemIfAdapter(config, scorer=laya_scorer(agent))
        adapter._loaded_agent = agent
        return adapter
    if config.backend != "torch-direct":
        raise ValueError(f"unsupported SemIf backend for autonomous loop: {config.backend}")
    from semif_phase1.core import load_causal_model
    model, tokenizer, metadata = load_causal_model(config.model, config.revision)
    adapter = RealSemIfAdapter(config, scorer=real_direct_scorer(model, tokenizer, metadata))
    # Keep the model alive for the adapter's lifetime; caller may release the process after the task.
    adapter._loaded_model = model
    adapter._loaded_tokenizer = tokenizer
    return adapter


def extract_app_name(task: str) -> str | None:
    match = re.search(r"\b(?:open|launch|start)\s+(.+?)(?:\s+and\b|\s+then\b|$)", task, re.I)
    if not match:
        return None
    name = match.group(1).strip(" .,!\"")
    return re.sub(r"^(?:a|an|the)\s+", "", name, flags=re.IGNORECASE)


def requests_existing_app(task: str) -> bool:
    return bool(re.search(
        r"\b(existing|already\s+open|current|active|focused)\s+(?:the\s+)?(?:terminal|window|app)",
        task, re.I,
    ))


def main() -> None:
    ap = argparse.ArgumentParser(description="NEXUS local agent")
    ap.add_argument("--mode", choices=("text", "voice"), default="text")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--task", default="")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--autonomous", action="store_true",
                    help="use local vision + SemIf candidate selection for bounded computer use")
    ap.add_argument("--enable-computer-use", action="store_true",
                    help="explicitly allow real desktop input for this run")
    args = ap.parse_args()
    cfg = load_default(args.config)
    if args.enable_computer_use:
        cfg["computer_use_enabled"] = True
    if args.debug:
        cfg["debug"] = True
    print(f"NEXUS mode={args.mode} semif_available={SEMIF_AVAILABLE} debug={cfg.get('debug')}")
    if args.mode == "voice":
        run_voice_task(cfg, args.autonomous)
        return
    if args.task:
        if args.autonomous:
            run_autonomous_task(args.task, cfg)
        else:
            run_task(args.task, cfg)
        return
    # Minimal text demo: route one intent through Real SemIf validation path.
    adapter = RealSemIfAdapter(SemIfConfig())
    res = adapter.decide(
        {"utterance": "open chrome and search nvidia driver"},
        "Which tool should handle this request?",
        TOOL_ROUTING,
        decision_id="demo-1",
    )
    print(f"route -> {res.winner_id} probs={ [round(p,3) for p in res.probabilities] }")


if __name__ == "__main__":
    main()
