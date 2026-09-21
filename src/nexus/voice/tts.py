"""Local KittenTTS text-to-speech adapter."""
from __future__ import annotations

from pathlib import Path


class KittenSynthesizer:
    def __init__(self, model: str = "KittenML/kitten-tts-mini-0.8",
                 voice: str = "Jasper", speed: float = 1.0):
        self.model_name = model
        self.voice = voice
        self.speed = speed
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from kittentts import KittenTTS
            except ImportError as exc:
                raise RuntimeError(
                    "KittenTTS is not installed; run python -m pip install -e \".[voice]\""
                ) from exc
            self._model = KittenTTS(self.model_name)
        return self._model

    def preload(self) -> None:
        """Load the model without generating audio."""
        self._load()

    def synthesize(self, text: str, output: str | Path) -> Path:
        clean = text.strip()
        if not clean:
            raise ValueError("cannot synthesize empty text")
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._load().generate_to_file(
                clean, str(target), voice=self.voice, speed=self.speed,
                sample_rate=24000,
            )
        except Exception as exc:
            raise RuntimeError(f"KittenTTS synthesis failed: {exc}") from exc
        if not target.exists() or target.stat().st_size == 0:
            raise RuntimeError("KittenTTS returned no audio")
        return target

    def speak(self, text: str, output: str | Path) -> Path:
        from .audio import play_wav
        path = self.synthesize(text, output)
        play_wav(path)
        return path
