"""Voice provider tests — fully mocked, no mic/model/network."""
import json

import pytest

from nexus.voice import tts as tts_mod
from nexus.voice.asr import (
    ElevenLabsASR,
    LocalWhisperASR,
    OpenAICompatASR,
    VoiceServerASR,
    build_asr,
)
from nexus.voice.http import HttpResponse, VoiceProviderError
from nexus.voice.schemas import ASRConfig, TTSConfig
from nexus.voice.tts import (
    ElevenLabsTTS,
    OpenAICompatTTS,
    VoiceServerTTS,
    build_tts,
)


class FakeTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post_multipart(self, url, *, fields, files, headers, timeout):
        self.calls.append(
            {"url": url, "fields": fields, "files": files,
             "headers": headers, "timeout": timeout}
        )
        return self.response

    def post_json(self, url, *, payload, headers, timeout):
        self.calls.append(
            {"url": url, "payload": payload, "headers": headers,
             "timeout": timeout}
        )
        return self.response


def ok_json(payload: dict) -> HttpResponse:
    return HttpResponse(200, json.dumps(payload).encode())


def wav_file(tmp_path, name="cap.wav"):
    p = tmp_path / name
    p.write_bytes(b"RIFF" + b"\x00" * 100)
    return p


# --- configs ---


def test_asr_aliases_and_rejection():
    assert ASRConfig.model_validate({"provider": "whisper"}).provider == "local-whisper"
    assert ASRConfig.model_validate({"provider": "OPENAI"}).provider == "openai"
    assert TTSConfig.model_validate({"provider": "kittentts"}).provider == "local-kitten"
    with pytest.raises(Exception):
        ASRConfig.model_validate({"provider": "nope"})


def test_factories_return_expected_types():
    kw = {"api_key": "k"}
    assert isinstance(build_asr(ASRConfig(provider="local-whisper")), LocalWhisperASR)
    assert isinstance(build_asr(ASRConfig(provider="openai", **kw)), OpenAICompatASR)
    assert isinstance(
        build_asr(ASRConfig(provider="elevenlabs", **kw)), ElevenLabsASR
    )
    assert isinstance(build_asr(ASRConfig(provider="server")), VoiceServerASR)
    assert isinstance(
        build_tts(TTSConfig(provider="openai", **kw)), OpenAICompatTTS
    )
    assert isinstance(
        build_tts(TTSConfig(provider="elevenlabs", voice_id="v", **kw)),
        ElevenLabsTTS,
    )
    assert isinstance(build_tts(TTSConfig(provider="server")), VoiceServerTTS)


# --- OpenAI ASR ---


def test_openai_asr_request_shape(tmp_path):
    transport = FakeTransport(ok_json({"text": "  hello there "}))
    asr = OpenAICompatASR(
        ASRConfig(provider="openai", model="whisper-1",
                  base_url="https://x.test/v1", api_key="sk-abc"),
        transport,
    )
    out = asr.transcribe(wav_file(tmp_path))
    assert out.text == "hello there" and out.backend == "openai"
    call = transport.calls[0]
    assert call["url"] == "https://x.test/v1/audio/transcriptions"
    assert call["fields"]["model"] == "whisper-1"
    assert call["headers"] == {"Authorization": "Bearer sk-abc"}
    name, _data, ctype = call["files"]["file"]
    assert name == "cap.wav" and ctype == "audio/wav"


def test_openai_asr_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    asr = OpenAICompatASR(ASRConfig(provider="openai"), FakeTransport(ok_json({})))
    with pytest.raises(VoiceProviderError, match="OPENAI_API_KEY"):
        asr.transcribe(wav_file(tmp_path))


def test_openai_asr_http_error_hides_key(tmp_path):
    transport = FakeTransport(HttpResponse(401, b'{"error": "bad key"}'))
    asr = OpenAICompatASR(
        ASRConfig(provider="openai", api_key="sk-secret"), transport
    )
    with pytest.raises(VoiceProviderError, match="HTTP 401") as exc:
        asr.transcribe(wav_file(tmp_path))
    assert "sk-secret" not in str(exc.value)


def test_openai_asr_non_json_reply(tmp_path):
    transport = FakeTransport(HttpResponse(200, b"RIFFnoise"))
    asr = OpenAICompatASR(
        ASRConfig(provider="openai", api_key="k"), transport
    )
    with pytest.raises(VoiceProviderError, match="non-JSON"):
        asr.transcribe(wav_file(tmp_path))


def test_empty_capture_rejected(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    asr = OpenAICompatASR(
        ASRConfig(provider="openai", api_key="k"),
        FakeTransport(ok_json({"text": "x"})),
    )
    with pytest.raises(VoiceProviderError, match="empty"):
        asr.transcribe(empty)


# --- ElevenLabs ASR ---


def test_elevenlabs_asr_request_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "xi-abc")
    transport = FakeTransport(ok_json({"text": "hi", "language_code": "en"}))
    asr = ElevenLabsASR(
        ASRConfig(provider="elevenlabs",
                  base_url="https://api.elevenlabs.io/v1"),
        transport,
    )
    out = asr.transcribe(wav_file(tmp_path))
    assert out.text == "hi" and out.backend == "elevenlabs"
    call = transport.calls[0]
    assert call["url"] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert call["fields"]["model_id"] == "scribe_v1"
    assert call["headers"] == {"xi-api-key": "xi-abc"}


def test_elevenlabs_asr_missing_key_names_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    asr = ElevenLabsASR(
        ASRConfig(provider="elevenlabs"), FakeTransport(ok_json({}))
    )
    with pytest.raises(VoiceProviderError, match="ELEVENLABS_API_KEY"):
        asr.transcribe(wav_file(tmp_path))


# --- server ASR ---


def test_server_asr_url(tmp_path):
    transport = FakeTransport(ok_json({"text": "yo"}))
    asr = VoiceServerASR(
        ASRConfig(provider="server", server_url="http://voice:8001/"), transport
    )
    assert asr.transcribe(wav_file(tmp_path)).backend == "server"
    assert transport.calls[0]["url"] == "http://voice:8001/v1/stt"


# --- TTS ---


def test_openai_tts_writes_audio(tmp_path):
    transport = FakeTransport(HttpResponse(200, b"RIFF-audio"))
    synth = OpenAICompatTTS(
        TTSConfig(provider="openai", api_key="sk-abc"), transport
    )
    out = synth.synthesize("hello", tmp_path / "r.wav")
    assert out.read_bytes() == b"RIFF-audio"
    call = transport.calls[0]
    assert call["url"] == "https://api.openai.com/v1/audio/speech"
    payload = json.loads(call["payload"].decode())
    # kitten defaults substituted with openai-capable ones
    assert payload == {"model": "gpt-4o-mini-tts", "input": "hello",
                       "voice": "coral", "response_format": "wav"}
    assert call["headers"] == {"Authorization": "Bearer sk-abc"}


def test_elevenlabs_tts_url_shape(tmp_path):
    transport = FakeTransport(HttpResponse(200, b"\x00\x01audio"))
    synth = ElevenLabsTTS(
        TTSConfig(provider="elevenlabs", voice_id="v123",
                  api_key="xi-k"),
        transport,
    )
    out = synth.synthesize("hi", tmp_path / "r.wav")
    assert out.read_bytes() == b"\x00\x01audio"
    call = transport.calls[0]
    assert call["url"] == (
        "https://api.elevenlabs.io/v1/text-to-speech/v123/stream"
        "?output_format=pcm_24000"
    )
    assert call["headers"] == {"xi-api-key": "xi-k"}
    assert json.loads(call["payload"].decode())["model_id"] == (
        "eleven_multilingual_v2"
    )


def test_elevenlabs_tts_requires_voice_id(tmp_path):
    synth = ElevenLabsTTS(
        TTSConfig(provider="elevenlabs", api_key="k"),
        FakeTransport(HttpResponse(200, b"x")),
    )
    with pytest.raises(VoiceProviderError, match="voice_id"):
        synth.synthesize("hi", tmp_path / "r.wav")


def test_server_tts_payload(tmp_path):
    transport = FakeTransport(HttpResponse(200, b"RIFF-s"))
    synth = VoiceServerTTS(
        TTSConfig(provider="server", server_url="http://voice:8001",
                  voice="Jasper", speed=1.0),
        transport,
    )
    synth.synthesize("hey", tmp_path / "r.wav")
    call = transport.calls[0]
    assert call["url"] == "http://voice:8001/v1/tts"
    assert json.loads(call["payload"].decode()) == {
        "text": "hey", "voice": "Jasper", "speed": 1.0}


def test_online_tts_rejects_empty_text(tmp_path):
    for synth in (
        OpenAICompatTTS(TTSConfig(provider="openai", api_key="k"),
                        FakeTransport(HttpResponse(200, b"x"))),
        ElevenLabsTTS(TTSConfig(provider="elevenlabs", voice_id="v",
                                api_key="k"),
                      FakeTransport(HttpResponse(200, b"x"))),
        VoiceServerTTS(TTSConfig(provider="server"),
                       FakeTransport(HttpResponse(200, b"x"))),
    ):
        with pytest.raises(ValueError):
            synth.synthesize("   ", tmp_path / "r.wav")


def test_online_tts_error_json_surfaced_truncated(tmp_path):
    long_msg = "x" * 500
    transport = FakeTransport(
        HttpResponse(200, json.dumps({"error": long_msg}).encode())
    )
    synth = OpenAICompatTTS(
        TTSConfig(provider="openai", api_key="k"), transport
    )
    with pytest.raises(VoiceProviderError, match="TTS error") as exc:
        synth.synthesize("hi", tmp_path / "r.wav")
    assert len(str(exc.value)) < 400


def test_build_tts_local_keeps_kitten_defaults():
    synth = build_tts(tts_mod.TTSConfig(provider="kittentts"))
    assert isinstance(synth, tts_mod.KittenSynthesizer)
