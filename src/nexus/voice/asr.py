"""Speech-to-text providers behind one Protocol.

Providers: local-whisper (existing audio.transcribe_wav) | openai
(OpenAI-compatible /audio/transcriptions) | elevenlabs (native Scribe
/v1/speech-to-text, xi-api-key) | server (standalone NEXUS voice server).

Transport is stdlib urllib with an injectable HttpTransport so unit tests
never touch the network. API keys resolve from env at call time.
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
from .schemas import ASRConfig, Transcription

LOCAL_DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
ELEVENLABS_DEFAULT_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_SCRIBE_MODEL = "scribe_v1"


class SpeechRecognizer(Protocol):
    def transcribe(self, audio_path: str | Path) -> Transcription: ...


def _text_from_json(body: bytes, url: str) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise VoiceProviderError(f"POST {url} returned non-JSON audio reply") from exc
    if isinstance(payload, dict):
        for key in ("text", "transcript"):
            value = payload.get(key)
            if isinstance(value, str):
                # Empty string = silence / no speech, a valid result the
                # voice loop discards. Only a missing key is an error.
                return value.strip()
    raise VoiceProviderError(f"POST {url} reply had no text field")


def _read_wav_bytes(path: Path) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VoiceProviderError(f"cannot read capture {path}: {exc}") from exc
    if not data:
        raise VoiceProviderError(f"capture {path} is empty; nothing to transcribe")
    return data


class LocalWhisperASR:
    """Wraps the existing Windows-local Whisper pipeline (lazy import)."""

    def __init__(self, model: str = LOCAL_DEFAULT_MODEL) -> None:
        self.model = model

    def preload(self) -> str:
        from .audio import preload_transcriber

        return preload_transcriber(model=self.model)

    def transcribe(self, audio_path: str | Path) -> Transcription:
        from .audio import LAST_DEVICE, transcribe_wav

        text = transcribe_wav(audio_path, model=self.model)
        try:
            backend = f"local-whisper:{LAST_DEVICE}"
        except Exception:
            backend = "local-whisper"
        return Transcription(text=text, backend=backend, language="en")


class OpenAICompatASR:
    """Any OpenAI-compatible /audio/transcriptions endpoint."""

    def __init__(self, cfg: ASRConfig, transport: HttpTransport | None = None) -> None:
        self.cfg = cfg
        self.transport = transport or UrllibTransport()

    def transcribe(self, audio_path: str | Path) -> Transcription:
        key = resolve_key(self.cfg.api_key, self.cfg.api_key_env, "openai")
        path = Path(audio_path)
        url = self.cfg.base_url.rstrip("/") + "/audio/transcriptions"
        fields = {"model": self.cfg.model, "language": self.cfg.language}
        files = {"file": (path.name, _read_wav_bytes(path), "audio/wav")}
        response = self.transport.post_multipart(
            url,
            fields=fields,
            files=files,
            headers={"Authorization": f"Bearer {key}"},
            timeout=self.cfg.timeout,
        )
        check_status(url, response)
        return Transcription(
            text=_text_from_json(response.body, url),
            backend="openai",
            language=self.cfg.language,
        )


class ElevenLabsASR:
    """Native Scribe endpoint (xi-api-key, not OpenAI-shaped)."""

    def __init__(self, cfg: ASRConfig, transport: HttpTransport | None = None) -> None:
        self.cfg = cfg
        self.transport = transport or UrllibTransport()

    def transcribe(self, audio_path: str | Path) -> Transcription:
        env_name = (
            self.cfg.api_key_env
            if self.cfg.api_key_env != "OPENAI_API_KEY"
            else "ELEVENLABS_API_KEY"
        )
        key = resolve_key(self.cfg.api_key, env_name, "elevenlabs")
        path = Path(audio_path)
        base = self.cfg.base_url
        if "openai.com" in base:
            base = ELEVENLABS_DEFAULT_BASE
        url = base.rstrip("/") + "/speech-to-text"
        model = (
            ELEVENLABS_SCRIBE_MODEL
            if self.cfg.model == LOCAL_DEFAULT_MODEL
            else self.cfg.model
        )
        fields = {"model_id": model}
        if self.cfg.language not in {"", "auto"}:
            fields["language_code"] = self.cfg.language
        files = {"file": (path.name, _read_wav_bytes(path), "audio/wav")}
        response = self.transport.post_multipart(
            url,
            fields=fields,
            files=files,
            headers={"xi-api-key": key},
            timeout=self.cfg.timeout,
        )
        check_status(url, response)
        return Transcription(
            text=_text_from_json(response.body, url),
            backend="elevenlabs",
            language=self.cfg.language,
        )


class VoiceServerASR:
    """Standalone NEXUS voice server: POST {server_url}/v1/stt."""

    def __init__(self, cfg: ASRConfig, transport: HttpTransport | None = None) -> None:
        self.cfg = cfg
        self.transport = transport or UrllibTransport()

    def transcribe(self, audio_path: str | Path) -> Transcription:
        path = Path(audio_path)
        url = self.cfg.server_url.rstrip("/") + "/v1/stt"
        files = {"file": (path.name, _read_wav_bytes(path), "audio/wav")}
        response = self.transport.post_multipart(
            url, fields={}, files=files, headers={}, timeout=self.cfg.timeout
        )
        check_status(url, response)
        return Transcription(
            text=_text_from_json(response.body, url), backend="server"
        )


def build_asr(
    cfg: ASRConfig, transport: HttpTransport | None = None
) -> SpeechRecognizer:
    if cfg.provider == "local-whisper":
        return LocalWhisperASR(model=cfg.model)
    if cfg.provider == "openai":
        return OpenAICompatASR(cfg, transport)
    if cfg.provider == "elevenlabs":
        return ElevenLabsASR(cfg, transport)
    if cfg.provider == "server":
        return VoiceServerASR(cfg, transport)
    raise ValueError(f"unknown ASR provider: {cfg.provider}")
