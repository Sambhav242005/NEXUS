"""Windows UI Automation observation provider.

Coordinates come from actual UIA element rectangles, never from an LLM.
"""
from __future__ import annotations

from nexus.vision.schemas import UIElement


def active_elements(limit: int = 200) -> list[UIElement]:
    try:
        from pywinauto import Desktop
    except ImportError as exc:
        raise RuntimeError("UI Automation requires pywinauto; install with pip install -e '.[computer]'") from exc

    desktop = Desktop(backend="uia")
    windows = desktop.windows()
    active = next((window for window in reversed(windows) if window.is_active()), None)
    if active is None and windows:
        active = windows[-1]
    if active is None:
        return []

    result: list[UIElement] = []
    for control in active.descendants()[:limit]:
        try:
            if not control.is_visible():
                continue
            rect = control.rectangle()
            width, height = rect.right - rect.left, rect.bottom - rect.top
            if width <= 0 or height <= 0:
                continue
            label = (control.window_text() or "").strip()
            role = str(getattr(control.element_info, "control_type", "unknown") or "unknown")
            if not label and role.lower() in {"pane", "group", "custom"}:
                continue
            result.append(UIElement(
                label=label, role=role, x=rect.left, y=rect.top,
                width=width, height=height, confidence=1.0, source="uia",
            ))
        except Exception:
            continue
    return result


def active_window_action(action: str, x: int | None = None, y: int | None = None,
                         width: int | None = None, height: int | None = None) -> None:
    """Apply a window-state action to the active window (close/minimize/maximize/resize).

    Coordinates are unused for window actions; they are accepted for schema parity
    with ComputerAction. Close sends WM_CLOSE and waits briefly; it raises if the
    window does not close (e.g. a save-changes prompt is pending)."""
    from pywinauto import Desktop
    desktop = Desktop(backend="uia")
    windows = desktop.windows()
    active = next((w for w in reversed(windows) if w.is_active()), None)
    if active is None and windows:
        active = windows[-1]
    if active is None:
        raise RuntimeError("no active window to act on")

    if action == "close_window":
        active.close()
    elif action == "minimize_window":
        active.minimize()
    elif action == "maximize_window":
        active.maximize()
    elif action == "resize":
        if width is None or height is None or width <= 0 or height <= 0:
            raise ValueError("resize requires positive width and height")
        active.resize(width, height)
    else:
        raise ValueError(f"unsupported window action: {action}")
