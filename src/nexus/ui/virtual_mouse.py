"""Virtual mouse: a small screen-map panel that drives the real pointer.

Controls the PRIMARY monitor only. Coordinates are physical screen pixels
(origin top-left); the panel scales 1:1 to the screen. Every dispatch goes
through ComputerController.execute(), so schema + screen-bounds validation and
the safety policy apply unchanged. Single click on the map = move; double
click = click; right click = context menu; wheel = scroll.
"""
from __future__ import annotations

from dataclasses import dataclass

from nexus.computer.controller import ControllerResult
from nexus.computer.schemas import ComputerAction


@dataclass
class ScreenMap:
    """Pure panel<->screen pixel mapping with clamping to screen bounds."""

    screen_w: int
    screen_h: int
    panel_w: int
    panel_h: int

    def to_screen(self, px: int, py: int) -> tuple[int, int]:
        sx = int(round(px * self.screen_w / self.panel_w))
        sy = int(round(py * self.screen_h / self.panel_h))
        sx = max(0, min(self.screen_w - 1, sx))
        sy = max(0, min(self.screen_h - 1, sy))
        return sx, sy

    def to_panel(self, sx: int, sy: int) -> tuple[int, int]:
        px = int(round(sx * self.panel_w / self.screen_w))
        py = int(round(sy * self.panel_h / self.screen_h))
        px = max(0, min(self.panel_w - 1, px))
        py = max(0, min(self.panel_h - 1, py))
        return px, py


class VirtualMouse:
    """GUI-free control surface. Maps panel pixels to screen pixels and
    dispatches validated ComputerAction through the controller."""

    def __init__(self, controller, screen_w: int, screen_h: int,
                 panel_w: int = 480, panel_h: int = 320):
        self.controller = controller
        self.map = ScreenMap(screen_w, screen_h, panel_w, panel_h)

    def _dispatch(self, action: ComputerAction) -> ControllerResult:
        try:
            return self.controller.execute(action)
        except Exception as e:
            return ControllerResult(ok=False, detail=f"virtual-mouse {action.action} rejected: {e}")

    def move(self, px: int, py: int) -> ControllerResult:
        x, y = self.map.to_screen(px, py)
        return self._dispatch(ComputerAction(action="move", x=x, y=y))

    def click(self, px: int, py: int) -> ControllerResult:
        x, y = self.map.to_screen(px, py)
        return self._dispatch(ComputerAction(action="click", x=x, y=y))

    def double_click(self, px: int, py: int) -> ControllerResult:
        x, y = self.map.to_screen(px, py)
        return self._dispatch(ComputerAction(action="double_click", x=x, y=y))

    def right_click(self, px: int, py: int) -> ControllerResult:
        x, y = self.map.to_screen(px, py)
        return self._dispatch(ComputerAction(action="right_click", x=x, y=y))

    def scroll(self, px: int, py: int, amount: int) -> ControllerResult:
        x, y = self.map.to_screen(px, py)
        return self._dispatch(ComputerAction(action="scroll", x=x, y=y, text=str(amount)))

    def type_text(self, text: str) -> ControllerResult:
        return self._dispatch(ComputerAction(action="type", text=text))

    def keypress(self, key: str) -> ControllerResult:
        return self._dispatch(ComputerAction(action="keypress", key=key))


def build_virtual_mouse_panel(controller, parent=None, screen_w: int | None = None,
                              screen_h: int | None = None):
    """Build the Tk screen-map panel. Returns the Toplevel window."""
    import tkinter as tk
    from tkinter import ttk
    from nexus.computer.mouse import WindowsMouse

    if screen_w is None or screen_h is None:
        screen_w, screen_h = WindowsMouse().screen_size()
    panel_w, panel_h = 480, 320
    vm = VirtualMouse(controller, screen_w, screen_h, panel_w, panel_h)
    last = [0, 0]  # screen pixels of last map point
    bg, border, accent, line = "#161b22", "#30363d", "#58a6ff", "#c9d1d9"

    win = tk.Toplevel(parent if parent is not None else None)
    win.title("NEXUS // VIRTUAL MOUSE (primary screen)")
    win.geometry(f"{panel_w + 40}x{panel_h + 200}")
    win.minsize(panel_w + 40, panel_h + 200)
    win.configure(bg="#0d1117")
    style = ttk.Style(win)
    style.theme_use("clam")
    style.configure("TFrame", background="#0d1117")
    style.configure("TLabel", background="#0d1117", foreground="#c9d1d9",
                    font=("Consolas", 10))
    style.configure("Title.TLabel", font=("Consolas", 12, "bold"), foreground="#58a6ff")
    style.configure("TButton", font=("Consolas", 10), background="#21262d",
                    foreground="#c9d1d9", bordercolor="#30363d")

    frm = ttk.Frame(win, padding=16)
    frm.pack(fill="both", expand=True)
    ttk.Label(frm, text="VIRTUAL MOUSE — primary screen only", style="Title.TLabel").pack(anchor="w")
    ttk.Label(frm, text="click=move  double-click=click  right-click=context  wheel=scroll").pack(anchor="w", pady=(2, 10))

    canvas = tk.Canvas(frm, width=panel_w, height=panel_h, bg=bg,
                       highlightthickness=0, cursor="crosshair")
    canvas.pack(anchor="w")
    canvas.create_rectangle(0, 0, panel_w, panel_h, fill=bg, outline=border)
    cross = [canvas.create_line(0, 0, 0, 0, fill=accent, width=2),
             canvas.create_line(0, 0, 0, 0, fill=accent, width=2)]

    status_var = tk.StringVar(value="ready — click the map to move the pointer")
    ttk.Label(frm, textvariable=status_var, background="#0d1117", foreground="#8b949e",
              font=("Consolas", 9)).pack(anchor="w", pady=(8, 0))

    def _status(res: ControllerResult):
        txt = f"OK  {res.detail}" if res.ok else f"ERR {res.detail}"
        status_var.set(txt)
        # Keep the window on top of what the pointer just jumped to, briefly.
        win.lift()
        win.focus_force()

    def _redraw():
        x, y = last
        canvas.coords(cross[0], x - 12, y, x + 12, y)
        canvas.coords(cross[1], x, y - 12, x, y + 12)

    def on_map_move(e):
        x, y = vm.map.to_screen(e.x, e.y)
        last[0], last[1] = x, y
        _redraw()
        _status(vm.move(e.x, e.y))

    def on_map_click(e):
        x, y = vm.map.to_screen(e.x, e.y)
        last[0], last[1] = x, y
        _redraw()
        _status(vm.click(e.x, e.y))

    def on_map_right(e):
        x, y = vm.map.to_screen(e.x, e.y)
        last[0], last[1] = x, y
        _redraw()
        _status(vm.right_click(e.x, e.y))

    def on_map_wheel(e):
        # Windows: <MouseWheel> with e.delta (+up/-down, 120 per notch).
        # X11/Linux: <Button-4> (num 4) = up, <Button-5> (num 5) = down.
        delta = getattr(e, "delta", None)
        if delta is not None and delta != 0:
            amount = delta
        else:
            amount = -120 if getattr(e, "num", 4) == 4 else 120
        _status(vm.scroll(e.x, e.y, amount))

    canvas.bind("<Button-1>", on_map_move)
    canvas.bind("<Double-1>", on_map_click)
    canvas.bind("<Button-3>", on_map_right)
    canvas.bind("<MouseWheel>", on_map_wheel)  # Windows wheel
    canvas.bind("<Button-4>", on_map_wheel)  # X11 wheel up
    canvas.bind("<Button-5>", on_map_wheel)  # X11 wheel down

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(12, 0))
    ttk.Button(btns, text="Click", command=lambda: _status(vm.click(last[0], last[1]))).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Right-click", command=lambda: _status(vm.right_click(last[0], last[1]))).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Scroll up", command=lambda: _status(vm.scroll(last[0], last[1], -120))).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Scroll down", command=lambda: _status(vm.scroll(last[0], last[1], 120))).pack(side="left", padx=(0, 6))
    ttk.Button(btns, text="Enter", command=lambda: _status(vm.keypress("enter"))).pack(side="left", padx=(0, 6))

    type_var = tk.StringVar()
    tk.Entry(frm, textvariable=type_var, width=30, font=("Consolas", 10),
             bg="#161b22", fg="#c9d1d9").pack(anchor="w", pady=(8, 0), fill="x")
    type_btn = ttk.Frame(frm)
    type_btn.pack(fill="x", pady=(0, 0))
    ttk.Button(type_btn, text="Type", command=lambda: _status(vm.type_text(type_var.get()))).pack(side="left", padx=(0, 6))
    ttk.Button(type_btn, text="Clear", command=lambda: type_var.set("")).pack(side="left")

    return win
