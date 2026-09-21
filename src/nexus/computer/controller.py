"""Validated computer action dispatcher with injectable real/mock backends."""
from __future__ import annotations
import subprocess
import time
import webbrowser
from pathlib import Path
from pydantic import BaseModel
from .schemas import ComputerAction
from .keyboard import WindowsKeyboard
from .mouse import WindowsMouse
from . import screenshot as screen


class ControllerResult(BaseModel):
    ok: bool
    detail: str
    screenshot_path: str | None = None


class ComputerController:
    def __init__(self, workdir: str | Path = ".", enabled: bool = False,
                 mouse=None, keyboard=None):
        self.workdir = Path(workdir)
        self.enabled = enabled
        self.mouse = mouse
        self.keyboard = keyboard

    def _real_backends(self):
        if self.mouse is None:
            self.mouse = WindowsMouse()
        if self.keyboard is None:
            self.keyboard = WindowsKeyboard()

    def execute(self, action: ComputerAction) -> ControllerResult:
        if action.action == "screenshot":
            return self.screenshot()
        if self.enabled:
            self._real_backends()
            if action.action == "move":
                self.mouse.move(action.x, action.y)
                return ControllerResult(ok=True, detail=f"moved pointer to ({action.x}, {action.y})")
            if action.action in ("click", "double_click"):
                self.mouse.click(action.x, action.y, 2 if action.action == "double_click" else 1)
                return ControllerResult(ok=True, detail=f"{action.action} at ({action.x}, {action.y})")
            if action.action in ("right_click", "middle_click"):
                self.mouse.click(action.x, action.y, button=action.action.removesuffix("_click"))
                return ControllerResult(ok=True, detail=f"{action.action} at ({action.x}, {action.y})")
            if action.action == "drag":
                self.mouse.drag(action.x, action.y, action.x2, action.y2)
                return ControllerResult(ok=True, detail=f"dragged ({action.x}, {action.y}) to ({action.x2}, {action.y2})")
            if action.action == "scroll":
                self.mouse.scroll(action.x or 0, action.y or 0, int(action.text or "0"))
                return ControllerResult(ok=True, detail=f"scrolled at ({action.x}, {action.y})")
            if action.action == "type":
                self.keyboard.type_text(action.text or "")
                return ControllerResult(ok=True, detail=f"typed {len(action.text or '')} chars")
            if action.action == "keypress":
                self.keyboard.keypress(action.key or "")
                return ControllerResult(ok=True, detail=f"pressed {action.key}")
            if action.action == "hotkey":
                self.keyboard.hotkey(action.keys or [])
                return ControllerResult(ok=True, detail=f"pressed {'+'.join(action.keys or [])}")
            if action.action in ("select_all", "copy", "paste", "save", "open_file"):
                keys = {"select_all": ["ctrl", "a"], "copy": ["ctrl", "c"],
                        "paste": ["ctrl", "v"], "save": ["ctrl", "s"],
                        "open_file": ["ctrl", "o"]}[action.action]
                self.keyboard.hotkey(keys)
                return ControllerResult(ok=True, detail=f"pressed {'+'.join(keys)}")
            if action.action == "launch_app":
                subprocess.Popen([action.text])
                return ControllerResult(ok=True, detail=f"launched {action.text}")
            if action.action == "open_url":
                webbrowser.open(action.text or "")
                return ControllerResult(ok=True, detail=f"opened URL {action.text}")
            if action.action in ("close_window", "minimize_window", "maximize_window", "resize"):
                from .accessibility import active_window_action
                active_window_action(action.action, action.x, action.y, action.width, action.height)
                return ControllerResult(ok=True, detail=f"window action {action.action}")
        if action.action == "type":
            # Real: agent writes its target file itself.
            target = self.workdir / "hello.txt"
            target.write_text(action.text or "", encoding="utf-8")
            return ControllerResult(ok=True, detail=f"typed {len(action.text or '')} chars -> {target}")
        if action.action == "click":
            # Demo mapping: click = launch notepad with hello.txt (one validated action).
            target = self.workdir / "hello.txt"
            if not target.exists():
                return ControllerResult(ok=False, detail=f"missing {target}, type first")
            try:
                subprocess.Popen(["notepad.exe", str(target)])
                return ControllerResult(ok=True, detail=f"launched notepad.exe {target}")
            except Exception as e:
                return ControllerResult(ok=False, detail=f"launch failed: {e}")
        return ControllerResult(ok=True, detail=f"mock-executed {action.action}")

    def launch_terminal(self) -> ControllerResult:
        """Open a real terminal window. Tries Windows Terminal, else PowerShell, else cmd."""
        for cmd in (["wt.exe"], ["powershell.exe"], ["cmd.exe"]):
            try:
                subprocess.Popen(cmd)
                return ControllerResult(ok=True, detail=f"launched {' '.join(cmd)}")
            except FileNotFoundError:
                continue
            except Exception as e:
                return ControllerResult(ok=False, detail=f"launch failed: {e}")
        return ControllerResult(ok=False, detail="no terminal found (wt/powershell/cmd)")

    def launch_notepad(self) -> ControllerResult:
        try:
            subprocess.Popen(["notepad.exe"])
            time.sleep(0.4)
            return ControllerResult(ok=True, detail="launched notepad.exe")
        except Exception as e:
            return ControllerResult(ok=False, detail=f"launch failed: {e}")

    def launch_application(self, name: str) -> ControllerResult:
        from .apps import launch_app
        ok, detail = launch_app(name)
        if ok:
            time.sleep(0.4)
        return ControllerResult(ok=ok, detail=detail)

    @staticmethod
    def process_running(*names: str) -> str | None:
        """Return first matching running process name from tasklist, else None."""
        try:
            out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True,
                                 text=True, timeout=10).stdout.lower()
        except Exception:
            return None
        for n in names:
            if n.lower() in out:
                return n
        return None

    def capture_screenshot(self, path: str | Path | None = None) -> ControllerResult:
        try:
            width, height, saved = screen.capture(path or (self.workdir / ".nexus-last-screen.png"))
            return ControllerResult(ok=True, detail=f"screenshot {width}x{height}", screenshot_path=saved)
        except Exception as e:
            return ControllerResult(ok=False, detail=f"screenshot failed: {e}")

    def accessibility_elements(self):
        from .accessibility import active_elements
        return active_elements()

    def screenshot(self) -> ControllerResult:
        try:
            width, height, path = screen.capture()
            return ControllerResult(ok=True, detail=f"screenshot {width}x{height}", screenshot_path=path)
        except Exception as e:
            return ControllerResult(ok=False, detail=f"screenshot failed: {e}")
