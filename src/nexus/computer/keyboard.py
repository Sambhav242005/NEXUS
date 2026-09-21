"""Dependency-free Windows keyboard input backend."""
from __future__ import annotations

import ctypes
import platform
import time


_VK = {
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "space": 0x20,
    "backspace": 0x08, "esc": 0x1B, "escape": 0x1B, "delete": 0x2E,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "ctrl": 0x11, "control": 0x11,
    "alt": 0x12, "shift": 0x10, "win": 0x5B,
}


class WindowsKeyboard:
    # Desktop text controls can drop back-to-back injected Unicode events.
    # A small gap is materially more reliable than reporting success early.
    TEXT_EVENT_DELAY = 0.015

    def __init__(self, user32=None):
        if platform.system() != "Windows":
            raise OSError("WindowsKeyboard is Windows-only")
        self.user32 = user32 or ctypes.windll.user32

    def _vk(self, key: str) -> int:
        name = key.strip().lower()
        if len(name) == 1:
            return ord(name.upper())
        if name.startswith("f") and name[1:].isdigit():
            n = int(name[1:])
            if 1 <= n <= 24:
                return 0x70 + n - 1
        if name not in _VK:
            raise ValueError(f"unsupported key: {key}")
        return _VK[name]

    def keypress(self, key: str) -> None:
        vk = self._vk(key)
        self.user32.keybd_event(vk, 0, 0, 0)
        self.user32.keybd_event(vk, 0, 0x0002, 0)

    def hotkey(self, keys: list[str]) -> None:
        vks = [self._vk(k) for k in keys]
        for vk in vks:
            self.user32.keybd_event(vk, 0, 0, 0)
        for vk in reversed(vks):
            self.user32.keybd_event(vk, 0, 0x0002, 0)
            time.sleep(0.01)

    def type_text(self, text: str) -> None:
        # KEYEVENTF_UNICODE handles characters that have no US keyboard mapping.
        for char in text:
            code = ord(char)
            self.user32.keybd_event(0, code & 0xFF, 0x0004, 0)
            self.user32.keybd_event(0, code & 0xFF, 0x0004 | 0x0002, 0)
            time.sleep(self.TEXT_EVENT_DELAY)
