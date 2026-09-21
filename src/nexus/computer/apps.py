"""Deterministic Windows application resolver; model output is never a path."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ALIASES = {
    "notepad": "notepad.exe", "terminal": "wt.exe", "windows terminal": "wt.exe",
    "powershell": "powershell.exe", "paint": "mspaint.exe", "calculator": "calc.exe",
    "calc": "calc.exe", "explorer": "explorer.exe",
}


def resolve_app(name: str) -> str | Path | None:
    clean = name.strip().lower().strip('"\'')
    candidate = ALIASES.get(clean, clean)
    found = shutil.which(candidate)
    if found:
        return found
    if not candidate.endswith(".exe"):
        found = shutil.which(candidate + ".exe")
        if found:
            return found
    roots = [
        Path(os.environ.get("PROGRAMDATA", "C:\\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
        Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    ]
    needle = clean.replace(".exe", "")
    for root in roots:
        if not root.exists():
            continue
        matches = [path for path in root.rglob("*.lnk") if needle in path.stem.lower()]
        if matches:
            return matches[0]
    return None


def launch_app(name: str) -> tuple[bool, str]:
    resolved = resolve_app(name)
    if resolved is None:
        return False, f"application not found in PATH or Start Menu: {name}"
    try:
        if str(resolved).lower().endswith(".lnk"):
            os.startfile(str(resolved))
        else:
            subprocess.Popen([str(resolved)])
        return True, f"launched {name} ({resolved})"
    except Exception as exc:
        return False, f"launch failed for {name}: {exc}"
