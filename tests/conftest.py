import sys

collect_ignore = []
if sys.platform != "win32":
    # integration/ drives a real desktop (Windows-only for now).
    collect_ignore.append("integration/")
    # WindowsKeyboard requires user32.
    collect_ignore.append("unit/test_keyboard_backend.py")
    # Voice capture/transcribe tests need the voice extra (numpy + torch).
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        collect_ignore.append("unit/test_audio.py")
