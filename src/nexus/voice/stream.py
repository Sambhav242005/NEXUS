"""WebSocket streaming voice clients (agent side).

Pairs with ``voice.server`` ``/ws/stt`` + ``/ws/tts``. Async API on top of
the ``websockets`` package (``pip install -e ".[server]"``); pass an async
fake ``connect_fn`` in tests to avoid the network entirely.

STT: send int16 16 kHz mono PCM chunks -> receive ``speech_started`` /
``final`` events. TTS: send text -> receive ``meta`` + binary chunks +
``done``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable


def _ws_url(server_url: str, path: str) -> str:
    base = server_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    elif not base.startswith(("ws://", "wss://")):
        base = "ws://" + base
    return base + path


class WsError(RuntimeError):
    """Streaming voice failure (connect, protocol, timeout)."""


class WsSttClient:
    """Streaming STT: PCM bytes in, transcript events out."""

    PATH = "/ws/stt"

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        connect_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.server_url = server_url
        self.timeout = timeout
        self._connect_fn = connect_fn
        self._conn: Any | None = None

    async def open(self) -> None:
        if self._connect_fn is not None:
            self._conn = await self._connect_fn(_ws_url(self.server_url, self.PATH))
            return
        try:
            import websockets
        except ImportError as exc:
            raise WsError(
                "websockets missing; run: python -m pip install -e \".[server]\""
            ) from exc
        self._conn = await websockets.connect(_ws_url(self.server_url, self.PATH))

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> WsSttClient:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _require_conn(self) -> Any:
        if self._conn is None:
            raise WsError("STT stream not open; call open() first")
        return self._conn

    async def send_pcm(self, frames: bytes) -> None:
        if not frames:
            raise ValueError("cannot send empty PCM chunk")
        await self._require_conn().send(frames)

    async def send_stop(self) -> None:
        await self._require_conn().send(json.dumps({"type": "stop"}))

    async def send_abort(self) -> None:
        await self._require_conn().send(json.dumps({"type": "abort"}))

    async def recv_event(self) -> dict:
        raw = await self._require_conn().recv()
        if isinstance(raw, bytes):
            return {"type": "binary", "data": raw}
        try:
            event = json.loads(raw)
        except ValueError as exc:
            raise WsError(f"STT stream sent invalid JSON: {raw[:120]!r}") from exc
        if not isinstance(event, dict):
            raise WsError(f"STT stream sent non-object event: {event!r}")
        return event

    async def finalize(self, timeout: float | None = None) -> str:
        """Stop and wait for the final transcript (raises WsError on error)."""
        await self.send_stop()
        deadline = timeout if timeout is not None else self.timeout
        try:
            return await asyncio.wait_for(self._wait_final(), deadline)
        except asyncio.TimeoutError as exc:
            raise WsError("timed out waiting for final transcript") from exc

    async def _wait_final(self) -> str:
        while True:
            event = await self.recv_event()
            if event.get("type") == "final":
                return str(event.get("text", ""))
            if event.get("type") == "error":
                raise WsError(f"STT stream error: {event.get('detail')}")


class WsTtsClient:
    """Streaming TTS: text in, audio chunks out."""

    PATH = "/ws/tts"

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        connect_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self.server_url = server_url
        self.timeout = timeout
        self._connect_fn = connect_fn
        self._conn: Any | None = None

    async def open(self) -> None:
        if self._connect_fn is not None:
            self._conn = await self._connect_fn(_ws_url(self.server_url, self.PATH))
            return
        try:
            import websockets
        except ImportError as exc:
            raise WsError(
                "websockets missing; run: python -m pip install -e \".[server]\""
            ) from exc
        self._conn = await websockets.connect(_ws_url(self.server_url, self.PATH))

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> WsTtsClient:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def _require_conn(self) -> Any:
        if self._conn is None:
            raise WsError("TTS stream not open; call open() first")
        return self._conn

    async def send_text(
        self, text: str, voice: str | None = None, speed: float | None = None
    ) -> None:
        if not text.strip():
            raise ValueError("cannot synthesize empty text")
        await self._require_conn().send(
            json.dumps({"text": text.strip(), "voice": voice, "speed": speed})
        )

    async def recv_event(self) -> dict | bytes:
        """Next stream item: dict (meta/done/error) or raw audio bytes."""
        raw = await self._require_conn().recv()
        if isinstance(raw, bytes):
            return raw
        try:
            event = json.loads(raw)
        except ValueError as exc:
            raise WsError(f"TTS stream sent invalid JSON: {raw[:120]!r}") from exc
        if not isinstance(event, dict):
            raise WsError(f"TTS stream sent non-object event: {event!r}")
        return event

    async def collect(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        timeout: float | None = None,
    ) -> bytes:
        """Speak one text and gather all audio bytes (raises WsError on error)."""
        await self.send_text(text, voice, speed)
        deadline = timeout if timeout is not None else self.timeout
        try:
            return await asyncio.wait_for(self._gather(), deadline)
        except asyncio.TimeoutError as exc:
            raise WsError("timed out waiting for streamed audio") from exc

    async def _gather(self) -> bytes:
        chunks = bytearray()
        while True:
            event = await self.recv_event()
            if isinstance(event, bytes):
                chunks += event
            elif event.get("type") == "done":
                if not chunks:
                    raise WsError("TTS stream finished with no audio")
                return bytes(chunks)
            elif event.get("type") == "error":
                raise WsError(f"TTS stream error: {event.get('detail')}")
