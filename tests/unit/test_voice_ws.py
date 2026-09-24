"""WebSocket voice tests — fake backends, no models/network."""
import asyncio
import json

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from nexus.voice.server import RmsVad, WS_MAX_BUFFER_BYTES, create_app, rms_level
from nexus.voice.stream import WsError, WsSttClient, WsTtsClient, _ws_url


def tone(n=3200, amp=10000):
    return (
        (np.sin(2 * np.pi * 440 * np.arange(n) / 16000) * amp)
        .astype(np.int16)
        .tobytes()
    )


SILENCE = bytes(6400)


# --- VAD unit ---


def test_rms_level_silence_and_tone():
    assert rms_level(b"") == 0.0
    assert rms_level(bytes(100)) == 0.0
    assert rms_level(tone()) > 500.0


def test_vad_silence_only_never_endpoints():
    vad = RmsVad(threshold=500.0, silence_timeout=0.1)
    for _ in range(10):
        assert vad.feed(bytes(3200)) == "silence"
    assert not vad.speech_seen


def test_vad_speech_then_silence_endpoints_and_resets():
    vad = RmsVad(threshold=500.0, silence_timeout=0.1)  # needs 3200 silent bytes
    assert vad.feed(tone()) == "speech"
    assert vad.feed(bytes(1600)) == "silence"
    assert vad.feed(bytes(1600)) == "endpoint"
    vad.reset()
    assert not vad.speech_seen
    assert vad.feed(bytes(3200)) == "silence"


# --- /ws/stt ---


def make_client(**overrides):
    calls: list[str] = []

    def stt_fn(path):
        calls.append(str(path))
        return "hi there"

    kwargs = {
        "stt_fn": stt_fn,
        "tts_fn": lambda text, voice, speed: b"RIFF-fake",
        "vad_silence_timeout": 0.1,
    }
    kwargs.update(overrides)
    return TestClient(create_app(**kwargs)), calls


def test_ws_stt_auto_endpoint_two_utterances():
    client, calls = make_client()
    with client.websocket_connect("/ws/stt") as ws:
        for _ in range(2):
            ws.send_bytes(tone())
            assert ws.receive_json() == {"type": "speech_started"}
            ws.send_bytes(SILENCE)
            assert ws.receive_json() == {"type": "final", "text": "hi there"}
    assert len(calls) == 2


def test_ws_stt_stop_with_empty_buffer():
    client, calls = make_client()
    with client.websocket_connect("/ws/stt") as ws:
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json() == {"type": "final", "text": ""}
    assert calls == []


def test_ws_stt_abort_discards_without_transcribing():
    client, calls = make_client()
    with client.websocket_connect("/ws/stt") as ws:
        ws.send_bytes(tone())
        assert ws.receive_json() == {"type": "speech_started"}
        ws.send_text(json.dumps({"type": "abort"}))
        assert ws.receive_json() == {"type": "aborted"}
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json() == {"type": "final", "text": ""}
    assert calls == []


def test_ws_stt_unknown_type_errors_but_stays_open():
    client, _ = make_client()
    with client.websocket_connect("/ws/stt") as ws:
        ws.send_text(json.dumps({"type": "frobnicate"}))
        event = ws.receive_json()
        assert event["type"] == "error"
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json() == {"type": "final", "text": ""}


def test_ws_stt_transcribe_failure_sends_error():
    def bad(path):
        raise RuntimeError("nope")

    client = TestClient(
        create_app(stt_fn=bad, tts_fn=lambda t, v, s: b"x",
                   vad_silence_timeout=0.1)
    )
    with client.websocket_connect("/ws/stt") as ws:
        ws.send_text(json.dumps({"type": "stop"}))
        assert ws.receive_json() == {"type": "final", "text": ""}
        ws.send_bytes(tone())
        assert ws.receive_json() == {"type": "speech_started"}
        ws.send_bytes(SILENCE)
        event = ws.receive_json()
        assert event["type"] == "error"


def test_ws_stt_buffer_cap_finalizes_and_survives():
    client, calls = make_client()
    chunk = tone(n=16000)  # 32000 loud bytes
    total = (WS_MAX_BUFFER_BYTES // len(chunk)) + 2
    with client.websocket_connect("/ws/stt") as ws:
        for _ in range(total):
            ws.send_bytes(chunk)
        # server only speaks on events: speech_started, then final at the cap
        finals = 0
        for _ in range(10):
            event = ws.receive_json()
            assert event["type"] in ("speech_started", "final")
            if event["type"] == "final":
                assert event["text"] == "hi there"
                finals += 1
                break
        assert finals == 1
    assert len(calls) >= 1


# --- /ws/tts ---


def test_ws_tts_chunks_reassemble():
    audio = bytes(range(256)) * 400  # 102400 bytes
    client = TestClient(
        create_app(stt_fn=lambda p: "", tts_fn=lambda t, v, s: audio)
    )
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_text(json.dumps({"text": "hello"}))
        meta = json.loads(ws.receive_text())
        assert meta["type"] == "meta" and meta["bytes"] == len(audio)
        got = bytearray()
        # receive loop: binary chunks until done (raw ASGI message dicts)
        while True:
            msg = ws.receive()
            if msg.get("bytes") is not None:
                got += msg["bytes"]
            elif msg.get("text") is not None:
                assert json.loads(msg["text"]) == {"type": "done"}
                break
            else:
                pytest.fail(f"unexpected WS message: {msg!r}")
        assert bytes(got) == audio


def test_ws_tts_empty_text_errors():
    client = TestClient(
        create_app(stt_fn=lambda p: "", tts_fn=lambda t, v, s: b"x")
    )
    with client.websocket_connect("/ws/tts") as ws:
        ws.send_text(json.dumps({"text": "  "}))
        assert ws.receive_json()["type"] == "error"


# --- streaming clients (fake connections) ---


class FakeConn:
    def __init__(self, incoming):
        self._incoming = list(incoming)
        self.sent: list = []
        self.urls: list[str] = []
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return self._incoming.pop(0)

    async def close(self):
        self.closed = True


def test_ws_url_conversion():
    assert _ws_url("http://h:8001", "/ws/stt") == "ws://h:8001/ws/stt"
    assert _ws_url("https://h/x", "/ws/tts") == "wss://h/x/ws/tts"
    assert _ws_url("ws://h:1", "/ws/stt") == "ws://h:1/ws/stt"


def test_stt_client_finalize():
    seen: list[str] = []

    async def connect(url):
        seen.append(url)
        return FakeConn([json.dumps({"type": "final", "text": "spoken"})])

    async def main():
        async with WsSttClient("http://h:8001", connect_fn=connect) as stt:
            await stt.send_pcm(tone(n=100))
            return await stt.finalize()

    assert asyncio.run(main()) == "spoken"
    assert seen == ["ws://h:8001/ws/stt"]


def test_stt_client_error_and_guards():
    async def connect(url):
        return FakeConn([json.dumps({"type": "error", "detail": "bad audio"})])

    async def main():
        async with WsSttClient("http://h:8001", connect_fn=connect) as stt:
            return await stt.finalize()

    with pytest.raises(WsError, match="bad audio"):
        asyncio.run(main())

    async def unopened():
        await WsSttClient("http://h:8001").send_pcm(b"\x01\x02")

    with pytest.raises(WsError, match="not open"):
        asyncio.run(unopened())

    async def empty_chunk():
        conn = FakeConn([])

        async def fake_connect(url):
            return conn

        stt = WsSttClient("http://h:8001", connect_fn=fake_connect)
        await stt.open()
        await stt.send_pcm(b"")

    with pytest.raises(ValueError):
        asyncio.run(empty_chunk())


def test_tts_client_collect():
    audio = b"A" * 70000

    async def connect(url):
        return FakeConn(
            [json.dumps({"type": "meta", "bytes": len(audio)}), audio[:30000],
             audio[30000:], json.dumps({"type": "done"})]
        )

    async def main():
        async with WsTtsClient("http://h:8001", connect_fn=connect) as tts:
            return await tts.collect("hello")

    assert asyncio.run(main()) == audio


def test_tts_client_error_and_empty_text():
    async def connect(url):
        return FakeConn([json.dumps({"type": "error", "detail": "no voice"})])

    async def main():
        async with WsTtsClient("http://h:8001", connect_fn=connect) as tts:
            return await tts.collect("hi")

    with pytest.raises(WsError, match="no voice"):
        asyncio.run(main())

    async def empty():
        async def fake_connect(url):
            return FakeConn([])

        async with WsTtsClient("http://h:8001", connect_fn=fake_connect) as tts:
            await tts.send_text("  ")

    with pytest.raises(ValueError):
        asyncio.run(empty())
