"""Task recorder: click Record -> capture screenshots -> Stop -> agent executes.

Logic (TaskRecorder) is GUI-free and tested headless. launch_ui() is a thin
tkinter shell: dark ops-console, big REC lamp, task entry, live status log.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class Recording:
    task: str
    dir: Path
    frames: list[str] = field(default_factory=list)
    started_at: str = ""
    stopped_at: str = ""
    audio_path: Path | None = None
    audio_error: str | None = None


class TaskRecorder:
    """Record mic + screenshots at interval, then hand task to agent loop."""

    def __init__(self, controller, workdir: str | Path = "recordings", interval: float = 1.0,
                 mic=None, mic_device: int | None = None):
        self.controller = controller
        self.workdir = Path(workdir)
        self.interval = interval
        self.mic = mic
        self.mic_device = mic_device
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.current: Recording | None = None

    def start(self, task: str) -> Recording:
        if self.current is not None:
            raise RuntimeError("already recording — stop first")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task.strip()[:40]) or "task"
        rdir = self.workdir / f"{stamp}-{safe}"
        rdir.mkdir(parents=True, exist_ok=True)
        self.current = Recording(task=task, dir=rdir, started_at=stamp)
        (rdir / "task.txt").write_text(task, encoding="utf-8")
        if self.mic is not None:
            try:
                self.mic.device = self.mic_device
                self.current.audio_path = self.mic.start(rdir / "audio.wav")
            except Exception as e:
                self.current.audio_path = None
                self.current.audio_error = str(e)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self.current

    def _loop(self):
        i = 0
        while not self._stop.is_set():
            try:
                # Lazy import avoids cycle: controller exposes execute().
                from nexus.computer.schemas import ComputerAction
                out = self.controller.execute(ComputerAction(action="screenshot"))
                i += 1
                line = f"frame-{i:04d} {time.strftime('%H:%M:%S')} {out.detail}\n"
                assert self.current is not None
                with open(self.current.dir / "frames.log", "a", encoding="utf-8") as f:
                    f.write(line)
                self.current.frames.append(line.strip())
            except Exception as e:
                assert self.current is not None
                self.current.frames.append(f"error: {e}")
            self._stop.wait(self.interval)

    def stop(self) -> Recording:
        if self.current is None:
            raise RuntimeError("nothing recording")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.mic is not None:
            try:
                self.current.audio_path = self.mic.stop()
            except Exception as e:
                if self.current.audio_path is None:
                    self.current.audio_error = str(e)
        self.current.stopped_at = datetime.now().strftime("%Y%m%d-%H%M%S")
        rec, self.current = self.current, None
        return rec


def launch_ui(workdir: str = ".", virtual_mouse: bool = False):
    """Dark ops-console: task entry + mic + REC + Play + Stop & Execute.

    virtual_mouse=True also opens the screen-map panel that drives the real
    pointer (primary screen)."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    from nexus.computer.controller import ComputerController
    from nexus.computer.schemas import ComputerAction
    from nexus.main import run_task
    from nexus.voice.audio import MicRecorder, default_mic, list_mics, play_wav

    # Clicking Execute is the explicit user opt-in for real desktop input.
    ctl = ComputerController(workdir, enabled=True)
    _mic_err: str | None = None
    try:
        mics = list_mics()
    except Exception as e:
        mics = []
        _mic_err = str(e)
    mic_names = [f"{i}: {name}" for i, name in mics] or ["(no mic found)"]
    def_mic = default_mic()
    rec = TaskRecorder(ctl, workdir=Path(workdir) / "recordings", interval=1.0,
                       mic=MicRecorder(device=def_mic), mic_device=def_mic)

    root = tk.Tk()
    root.title("NEXUS — Task Recorder")
    root.geometry("560x600")
    root.minsize(480, 520)
    root.configure(bg="#0d1117")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#0d1117")
    style.configure("TLabel", background="#0d1117", foreground="#c9d1d9",
                    font=("Consolas", 10))
    style.configure("Title.TLabel", font=("Consolas", 14, "bold"), foreground="#58a6ff")
    style.configure("Rec.TButton", font=("Consolas", 11, "bold"))
    style.configure("Stop.TButton", font=("Consolas", 11, "bold"))

    frm = ttk.Frame(root, padding=16)
    frm.pack(fill="both", expand=True)
    ttk.Label(frm, text="NEXUS // TASK RECORDER", style="Title.TLabel").pack(anchor="w")
    ttk.Label(frm, text="1. ● Record + speak  2. ■ Stop (shows what you said)  3. ▶ Execute").pack(anchor="w", pady=(4, 10))

    task_var = tk.StringVar(value="open notepad and create new file and write Hello world")
    ttk.Label(frm, text="Task:").pack(anchor="w")
    ttk.Entry(frm, textvariable=task_var, width=60).pack(fill="x", pady=(2, 6))

    ttk.Label(frm, text="Mic (Windows):").pack(anchor="w")
    mic_var = tk.StringVar(value=mic_names[0])
    mic_box = ttk.Combobox(frm, textvariable=mic_var, values=mic_names, state="readonly", width=50)
    mic_box.pack(fill="x", pady=(2, 4))
    if _mic_err:
        ttk.Label(frm, text=f"mic warn: {_mic_err}").pack(anchor="w")
    last_audio: dict = {}

    def _selected_mic() -> int | None:
        try:
            return int(mic_var.get().split(":", 1)[0])
        except Exception:
            return def_mic

    lamp_var = tk.StringVar(value="○ IDLE")
    lamp = ttk.Label(frm, textvariable=lamp_var, font=("Consolas", 12, "bold"))
    lamp.pack(anchor="w", pady=(0, 8))

    log = tk.Text(frm, height=7, bg="#161b22", fg="#c9d1d9",
                  insertbackground="#c9d1d9", font=("Consolas", 9))
    log.pack(fill="both", expand=True)

    def say(msg: str):
        log.insert("end", msg + "\n")
        log.see("end")

    def _run_action(action_name: Literal["save", "close_window"], confirm_title: str, confirm_msg: str):
        """Gated computer action: explicit confirm dialog, then validated dispatch."""
        if not messagebox.askyesno(confirm_title, confirm_msg):
            say(f"…{action_name} cancelled")
            return
        try:
            res = ctl.execute(ComputerAction(action=action_name))
            say(f"{'OK' if res.ok else 'ERR'} {action_name}: {res.detail}")
        except Exception as e:
            say(f"ERR {action_name}: {e}")

    def blink():
        if rec.current is not None:
            lamp_var.set("● REC  " + time.strftime("%H:%M:%S"))
            root.after(500, lambda: lamp_var.set("○ REC  " + time.strftime("%H:%M:%S"))
                       if rec.current is not None else None)
            root.after(1000, blink)

    def on_record():
        try:
            rec.mic_device = _selected_mic()
            if rec.mic is not None:
                rec.mic.device = rec.mic_device
            r = rec.start(task_var.get())
            if r.audio_error:
                say(f"● recording → {r.dir} (mic ERR: {r.audio_error})")
            else:
                say(f"● recording mic+screen → {r.dir}")
            blink()
        except Exception as e:
            say(f"ERR record: {e}")

    def on_play():
        p = last_audio.get("path")
        if p is None or not Path(p).exists():
            say("ERR play: no recording yet — Record then Stop first.")
            return
        try:
            say(f"▶ playing → {p}")
            play_wav(p, blocking=False)
        except Exception as e:
            say(f"ERR play: {e}")

    def on_stop():
        lamp_var.set("■ stopping…")
        say("■ stopping… (UI stays live, transcribing in background)")

        def work():
            try:
                r = rec.stop()
                info = {"frames": len(r.frames), "dir": str(r.dir),
                        "audio": str(r.audio_path) if r.audio_path is not None else None,
                        "audio_error": r.audio_error, "heard": None, "err": None}
                if info["audio"] is not None and Path(info["audio"]).exists():
                    last_audio["path"] = info["audio"]
                    try:
                        from nexus.voice.audio import transcribe_wav, LAST_DEVICE
                        info["heard"] = transcribe_wav(info["audio"])
                        import nexus.voice.audio as _va
                        info["device"] = _va.LAST_DEVICE
                    except Exception as e:
                        info["err"] = f"transcribe ERR: {e}"
            except Exception as e:
                info = {"err": f"ERR stop: {e}"}
            root.after(0, lambda: finish_stop(info))

        def finish_stop(info: dict):
            if "frames" not in info:
                say(info.get("err", "ERR stop: unknown"))
                lamp_var.set("○ IDLE")
                return
            say(f"■ stopped: {info['frames']} frames in {info['dir']}")
            if info["audio"] is not None:
                say(f"  audio saved → {info['audio']}")
            if info.get("heard") is not None:
                task_var.set(info["heard"])
                say(f'  YOU SAID ({info.get("device", "?")}): "{info["heard"]}"')
                say("  task box updated — hit ▶ Execute, or edit first.")
            elif info.get("err"):
                say(f"  {info['err']} (task box unchanged)")
            elif info.get("audio_error"):
                say(f"  audio ERR: {info['audio_error']}")
            lamp_var.set("○ IDLE")

        threading.Thread(target=work, daemon=True).start()

    def on_execute():
        task = task_var.get().strip()
        if not task:
            say("ERR execute: task box empty — Record + Stop first.")
            return
        say(f"▶ executing: {task}")
        try:
            run_task(task, {"workdir": workdir, "computer_use_enabled": True})
            say("✔ task finished — verify the target window and result.")
        except Exception as e:
            say(f"ERR execute: {e}")

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(10, 0))
    ttk.Button(btns, text="● Record", command=on_record, style="Rec.TButton").pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="▶ Play", command=on_play).pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="■ Stop", command=on_stop, style="Stop.TButton").pack(side="left", padx=(0, 8))
    ttk.Button(btns, text="▶ Execute", command=on_execute).pack(side="left")

    gated = ttk.Frame(frm)
    gated.pack(fill="x", pady=(6, 0))
    ttk.Button(gated, text="Save (Ctrl+S)",
               command=lambda: _run_action("save", "Send Ctrl+S?",
                                           "Send Ctrl+S to the active window?")).pack(side="left", padx=(0, 6))
    ttk.Button(gated, text="Close window",
               command=lambda: _run_action("close_window", "Close active window?",
                                           "Close the active window? Unsaved changes are lost.")).pack(side="left", padx=(0, 6))
    ttk.Label(gated, text="(act on focused window)", foreground="#8b949e",
              background="#0d1117", font=("Consolas", 9)).pack(side="right")

    def _preload():
        try:
            from nexus.voice.audio import preload_transcriber
            dev = preload_transcriber()
            root.after(0, lambda: say(f"…transcriber ready ({dev}) — Stop will be fast."))
        except Exception as e:
            root.after(0, lambda: say(f"…transcriber preload ERR: {e}"))

    if virtual_mouse:
        try:
            from nexus.ui.virtual_mouse import build_virtual_mouse_panel
            build_virtual_mouse_panel(ctl, parent=root)
            say("virtual mouse panel open — primary screen only")
        except Exception as e:
            say(f"ERR virtual mouse: {e}")

    threading.Thread(target=_preload, daemon=True).start()
    root.mainloop()
