"""Screen capture adapter. PIL is optional and only loaded when enabled."""
from __future__ import annotations

from pathlib import Path


def capture(path: str | Path | None = None) -> tuple[int, int, str | None]:
    from PIL import ImageGrab

    image = ImageGrab.grab(all_screens=True)
    output = None
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        output = str(output_path)
    return image.width, image.height, output
