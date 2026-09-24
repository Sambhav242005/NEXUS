"""Voice server contract tests — fake backends, no models/torch."""
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from nexus.voice.server import MAX_UPLOAD_BYTES, create_app


def make_client(**overrides):
    kwargs = {
        "stt_model": "test-stt",
        "tts_model": "test-tts",
        "stt_fn": lambda path: "hello server",
        "tts_fn": lambda text, voice, speed: b"RIFF-fake",
    }
    kwargs.update(overrides)
    return TestClient(create_app(**kwargs))


def test_health_reports_backends():
    res = make_client().get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "stt": "test-stt", "tts": "test-tts"}


def test_stt_returns_text():
    client = make_client()
    res = client.post("/v1/stt", files={"file": ("cap.wav", b"RIFF" + b"\0" * 64)})
    assert res.status_code == 200
    assert res.json() == {"text": "hello server", "backend": "server"}


def test_stt_empty_upload_is_empty_text():
    client = make_client()
    res = client.post("/v1/stt", files={"file": ("cap.wav", b"")})
    assert res.status_code == 200
    assert res.json()["text"] == ""


def test_stt_failure_is_500():
    def bad(path):
        raise RuntimeError("model exploded")

    res = make_client(stt_fn=bad).post(
        "/v1/stt", files={"file": ("cap.wav", b"RIFF" + b"\0" * 64)}
    )
    assert res.status_code == 500


def test_stt_oversize_rejected():
    client = make_client()
    big = b"\0" * (MAX_UPLOAD_BYTES + 1)
    res = client.post("/v1/stt", files={"file": ("cap.wav", big)})
    assert res.status_code == 413


def test_tts_returns_wav():
    client = make_client()
    res = client.post("/v1/tts", json={"text": "hi", "voice": "J", "speed": 1.0})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert res.content == b"RIFF-fake"


def test_tts_empty_text_is_400():
    res = make_client().post("/v1/tts", json={"text": "   "})
    assert res.status_code == 400


def test_tts_failure_is_500():
    def bad(text, voice, speed):
        raise RuntimeError("no voice")

    res = make_client(tts_fn=bad).post("/v1/tts", json={"text": "hi"})
    assert res.status_code == 500
