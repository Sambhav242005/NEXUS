"""Voice provider schemas (ASR/TTS config + results).

One config shape drives local, OpenAI-compatible, ElevenLabs, and
standalone voice-server backends. Secrets never live here -- only the
name of the env var holding them (resolved at call time).
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _normalize_asr_provider(value: object) -> object:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"whisper", "local", "local-whisper"}:
            return "local-whisper"
        return v
    return value


def _normalize_tts_provider(value: object) -> object:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"kittentts", "kitten", "local", "local-kitten"}:
            return "local-kitten"
        return v
    return value


ASRProvider = Annotated[
    Literal["local-whisper", "openai", "elevenlabs", "server"],
    BeforeValidator(_normalize_asr_provider),
]
TTSProvider = Annotated[
    Literal["local-kitten", "openai", "elevenlabs", "server"],
    BeforeValidator(_normalize_tts_provider),
]


class _Lenient(BaseModel):
    """Ignore unknown keys so flat YAML (device, device_index, ...) keeps working."""

    model_config = ConfigDict(extra="ignore")


class ASRConfig(_Lenient):
    provider: ASRProvider = "local-whisper"
    # Local default; build_asr() substitutes scribe_v1 for elevenlabs when untouched.
    model: str = "openai/whisper-large-v3-turbo"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None  # explicit override (tests/DI); env otherwise
    language: str = "en"
    server_url: str = "http://127.0.0.1:8001"
    timeout: float = 30.0


class TTSConfig(_Lenient):
    provider: TTSProvider = "local-kitten"
    # Local default; build_tts() substitutes per-provider defaults when untouched.
    model: str = "KittenML/kitten-tts-mini-0.8"
    voice: str = "Jasper"
    voice_id: str | None = None  # required by elevenlabs (voice ID, not display name)
    speed: float = 1.0
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    server_url: str = "http://127.0.0.1:8001"
    output_format: str = "pcm_24000"  # elevenlabs output_format query value
    timeout: float = 30.0


class Transcription(BaseModel):
    text: str
    backend: str = ""  # which provider produced it (observability)
    language: str | None = None


class AudioResult(BaseModel):
    path: str
    backend: str = ""
    content_type: str = Field(default="audio/wav")
