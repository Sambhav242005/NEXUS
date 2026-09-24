"""Local KittenTTS text-to-speech adapter + online/server providers.

Providers: local-kitten (KittenTTS, unchanged) | openai
(OpenAI-compatible /audio/speech) | elevenlabs (native
/v1/text-to-speech/{voice_id}/stream, xi-api-key) | server (standalone
NEXUS voice server). All share synthesize(text, output) -> Path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .http import (
    HttpTransport,
    UrllibTransport,
    VoiceProviderError,
    check_status,
    resolve_key,
)
from .schemas import TTSConfig

LOCAL_DEFAULT_MODEL = "KittenML/kitten-tts-mini-0.8"
LOCAL_DEFAULT_VOICE = "Jasper"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini-tts"
OPENAI_DEFAULT_VOICE = "coral"
ELEVENLABS_DEFAULT_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_DEFAULT_MODEL = "eleven_multilingual_v2"


class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str, output: str | Path) -> Path: ...


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


def _clean_text(text: str) -> str:
    clean = text.strip()
    if not clean:
        raise ValueError("cannot synthesize empty text")
    return clean


def _write_audio(body: bytes, target: Path, url: str, backend: str) -> Path:
    if not body:
        raise VoiceProviderError(f"{backend} TTS returned no audio ({url})")
    if body[:1] == b"{":
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("error"):
            detail = str(payload["error"])[:200]
            raise VoiceProviderError(f"{backend} TTS error: {detail}")
        raise VoiceProviderError(
            f"{backend} TTS returned JSON instead of audio ({url})"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    return target


class OpenAICompatTTS:
    """Any OpenAI-compatible /audio/speech endpoint."""

    def __init__(self, cfg: TTSConfig, transport: HttpTransport | None = None) -> None:
        self.cfg = cfg
        self.transport = transport or UrllibTransport()

    def synthesize(self, text: str, output: str | Path) -> Path:
        clean = _clean_text(text)
        key = resolve_key(self.cfg.api_key, self.cfg.api_key_env, "openai")
        model = (
            OPENAI_DEFAULT_MODEL
            if self.cfg.model == LOCAL_DEFAULT_MODEL
            else self.cfg.model
        )
        voice = (
            OPENAI_DEFAULT_VOICE
            if self.cfg.voice == LOCAL_DEFAULT_VOICE
            else self.cfg.voice
        )
        url = self.cfg.base_url.rstrip("/") + "/audio/speech"
        payload = json.dumps(
            {"model": model, "input": clean, "voice": voice,
             "response_format": "wav"}
        ).encode("utf-8")
        response = self.transport.post_json(
            url,
            payload=payload,
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.cfg.timeout,
        )
        check_status(url, response)
        return _write_audio(response.body, Path(output), url, "openai")


class ElevenLabsTTS:
    """Native ElevenLabs TTS (voice ID + xi-api-key, not OpenAI-shaped)."""

    def __init__(self, cfg: TTSConfig, transport: HttpTransport | None = None) -> None:
        self.cfg = cfg
        self.transport = transport or UrllibTransport()

    def synthesize(self, text: str, output: str | Path) -> Path:
        clean = _clean_text(text)
        if not self.cfg.voice_id:
            raise VoiceProviderError(
                "elevenlabs TTS needs voice_id (the voice ID from your "
                "ElevenLabs account, not the display name)"
            )
        env_name = (
            self.cfg.api_key_env
            if self.cfg.api_key_env != "OPENAI_API_KEY"
            else "ELEVENLABS_API_KEY"
        )
        key = resolve_key(self.cfg.api_key, env_name, "elevenlabs")
        base = self.cfg.base_url
        if "openai.com" in base:
            base = ELEVENLABS_DEFAULT_BASE
        model = (
            ELEVENLABS_DEFAULT_MODEL
            if self.cfg.model == LOCAL_DEFAULT_MODEL
            else self.cfg.model
        )
        url = (
            f"{base.rstrip()}/text-to-speech/{self.cfg.voice_id}/stream"
            f"?output_format={self.cfg.output_format}"
        )
        payload = json.dumps({"text": clean, "model_id": model}).encode("utf-8")
        response = self.transport.post_json(
            url,
            payload=payload,
            headers={"xi-api-key": key},
            timeout=self.cfg.timeout,
        )
        check_status(url, response)
        return _write_audio(response.body, Path(output), url, "elevenlabs")


class VoiceServerTTS:
    """Standalone NEXUS voice server: POST {server_url}/v1/tts."""

    def __init__(self, cfg: TTSConfig, transport: HttpTransport | None = None) -> None:
        self.cfg = cfg
        self.transport = transport or UrllibTransport()

    def synthesize(self, text: str, output: str | Path) -> Path:
        clean = _clean_text(text)
        url = self.cfg.server_url.rstrip("/") + "/v1/tts"
        payload = json.dumps(
            {"text": clean, "voice": self.cfg.voice, "speed": self.cfg.speed}
        ).encode("utf-8")
        response = self.transport.post_json(
            url, payload=payload, headers={}, timeout=self.cfg.timeout
        )
        check_status(url, response)
        return _write_audio(response.body, Path(output), url, "server")


def build_tts(
    cfg: TTSConfig, transport: HttpTransport | None = None
) -> SpeechSynthesizer:
    if cfg.provider == "local-kitten":
        return KittenSynthesizer(model=cfg.model, voice=cfg.voice, speed=cfg.speed)
    if cfg.provider == "openai":
        return OpenAICompatTTS(cfg, transport)
    if cfg.provider == "elevenlabs":
        return ElevenLabsTTS(cfg, transport)
    if cfg.provider == "server":
        return VoiceServerTTS(cfg, transport)
    raise ValueError(f"unknown TTS provider: {cfg.provider}")


def speak(synth: SpeechSynthesizer, text: str, output: str | Path) -> Path:
    """Synthesize with any provider, then play (Windows-only playback)."""
    path = synth.synthesize(text, output)
    from .audio import play_wav

    play_wav(path)
    return path
