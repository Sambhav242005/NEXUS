"""Small Windows mouse backend using user32 SendInput.

The backend is deliberately dependency-free. Coordinates are physical screen
pixels, origin at the top-left corner.
"""
from __future__ import annotations

import ctypes
import platform
import time


class WindowsMouse:
    def __init__(self, user32=None):
        if platform.system() != "Windows":
            raise OSError("WindowsMouse is Windows-only")
        self.user32 = user32 or ctypes.windll.user32

    def screen_size(self) -> tuple[int, int]:
        return int(self.user32.GetSystemMetrics(0)), int(self.user32.GetSystemMetrics(1))

    def _check(self, x: int, y: int) -> None:
        width, height = self.screen_size()
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"coordinates out of screen bounds: ({x}, {y}) for {width}x{height}")

    def move(self, x: int, y: int) -> None:
        self._check(x, y)
        if not self.user32.SetCursorPos(x, y):
            raise OSError("SetCursorPos failed")

    def click(self, x: int, y: int, count: int = 1, button: str = "left") -> None:
        self.move(x, y)
        flags = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010),
                 "middle": (0x0020, 0x0040)}
        down, up = flags[button]
        for _ in range(count):
            self.user32.mouse_event(down, 0, 0, 0, 0)
            self.user32.mouse_event(up, 0, 0, 0, 0)
            if count > 1:
                time.sleep(0.05)

    def drag(self, x: int, y: int, x2: int, y2: int) -> None:
        self._check(x, y)
        self._check(x2, y2)
        self.move(x, y)
        self.user32.mouse_event(0x0002, 0, 0, 0, 0)
        if not self.user32.SetCursorPos(x2, y2):
            raise OSError("SetCursorPos failed during drag")
        self.user32.mouse_event(0x0004, 0, 0, 0, 0)

    def scroll(self, x: int, y: int, amount: int) -> None:
        self.move(x, y)
        self.user32.mouse_event(0x0800, 0, 0, int(amount), 0)
