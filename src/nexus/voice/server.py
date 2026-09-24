"""Standalone NEXUS voice server (REST).

Host separately from the agent; the agent reaches it via the
``server`` ASR/TTS providers (``server_url``).

Run::

    python -m nexus.voice.server --port 8001

Endpoints::

    GET  /health  -> {"status": "ok", "stt": ..., "tts": ...}
    POST /v1/stt   multipart ``file`` (16 kHz mono WAV) -> {"text": ...}
    POST /v1/tts   JSON {text, voice?, speed?} -> audio/wav bytes

Default backends are the local ones (Whisper STT, KittenTTS); inject
fakes via create_app() in tests.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable

from .dsp import rms_level

try:
    from fastapi import (
        FastAPI,
        File,
        HTTPException,
        UploadFile,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel
except ImportError as exc:
    raise RuntimeError(
        "voice server deps missing; run: python -m pip install -e \".[server]\""
    ) from exc

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
WS_SAMPLE_RATE = 16000
WS_MAX_BUFFER_BYTES = 60 * WS_SAMPLE_RATE * 2  # ~60 s of 16 kHz mono int16
WS_TTS_CHUNK_BYTES = 32 * 1024


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None


class RmsVad:
    """Clock-free endpointing: speech seen, then N silent bytes -> endpoint."""

    def __init__(
        self,
        *,
        threshold: float = 500.0,
        silence_timeout: float = 1.0,
        sample_rate: int = WS_SAMPLE_RATE,
    ) -> None:
        self.threshold = threshold
        self.silence_bytes_needed = int(silence_timeout * sample_rate * 2)
        self.reset()

    def reset(self) -> None:
        self.speech_seen = False
        self._silent_bytes = 0

    def feed(self, pcm: bytes) -> str:
        """Consume PCM bytes. Returns 'silence' | 'speech' | 'endpoint'."""
        if rms_level(pcm) >= self.threshold:
            self.speech_seen = True
            self._silent_bytes = 0
            return "speech"
        if not self.speech_seen:
            return "silence"
        self._silent_bytes += len(pcm)
        if self._silent_bytes >= self.silence_bytes_needed:
            return "endpoint"
        return "silence"


def pcm_to_wav_file(pcm: bytes, sample_rate: int = WS_SAMPLE_RATE) -> Path:
    """Wrap raw int16 mono PCM in a WAV container for stt_fn."""
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return tmp


def _default_stt_fn(model: str) -> Callable[[Path], str]:
    def transcribe(path: Path) -> str:
        from .audio import transcribe_wav

        return transcribe_wav(path, model=model)

    return transcribe


def _default_tts_fn(
    model: str, voice: str, speed: float
) -> Callable[[str, str | None, float | None], bytes]:
    from .tts import KittenSynthesizer

    lock = threading.Lock()
    cache: dict[tuple[str, float], KittenSynthesizer] = {}

    def synthesize(
        text: str, req_voice: str | None, req_speed: float | None
    ) -> bytes:
        key = (req_voice or voice, req_speed or speed)
        with lock:
            synth = cache.get(key)
            if synth is None:
                synth = KittenSynthesizer(model=model, voice=key[0], speed=key[1])
                synth.preload()
                cache[key] = synth
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            synth.synthesize(text, tmp)
            return tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)

    return synthesize


def create_app(
    *,
    stt_model: str = "openai/whisper-large-v3-turbo",
    tts_model: str = "KittenML/kitten-tts-mini-0.8",
    tts_voice: str = "Jasper",
    tts_speed: float = 1.0,
    vad_threshold: float = 500.0,
    vad_silence_timeout: float = 1.0,
    stt_fn: Callable[[Path], str] | None = None,
    tts_fn: Callable[[str, str | None, float | None], bytes] | None = None,
) -> FastAPI:
    transcribe = stt_fn or _default_stt_fn(stt_model)
    synthesize = tts_fn or _default_tts_fn(tts_model, tts_voice, tts_speed)

    app = FastAPI(title="NEXUS voice server")

    def _transcribe_pcm(pcm: bytes) -> str:
        tmp = pcm_to_wav_file(pcm)
        try:
            return transcribe(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "stt": stt_model, "tts": tts_model}

    @app.post("/v1/stt")
    def stt(file: UploadFile = File(...)) -> JSONResponse:
        data = file.file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="upload too large")
        if not data:
            return JSONResponse({"text": "", "backend": "server"})
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        try:
            tmp.write_bytes(data)
            try:
                text = transcribe(tmp)
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"transcription failed: {exc}"
                ) from exc
            return JSONResponse({"text": text, "backend": "server"})
        finally:
            tmp.unlink(missing_ok=True)

    @app.post("/v1/tts")
    def tts(req: TTSRequest) -> Response:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")
        try:
            audio = synthesize(req.text.strip(), req.voice, req.speed)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"synthesis failed: {exc}"
            ) from exc
        if not audio:
            raise HTTPException(status_code=500, detail="synthesis returned no audio")
        return Response(content=audio, media_type="audio/wav")

    @app.websocket("/ws/stt")
    async def ws_stt(ws: WebSocket) -> None:
        """Stream PCM16 16 kHz mono in; get speech_started + final transcripts.

        Binary message: raw int16 mono frames (any chunk size).
        JSON messages: {"type": "stop"} (finalize now, stay connected),
        {"type": "abort"} (discard buffer). Server auto-finalizes after
        vad_silence_timeout of trailing silence. Multiple utterances per
        connection are supported.
        """
        await ws.accept()
        vad = RmsVad(
            threshold=vad_threshold, silence_timeout=vad_silence_timeout
        )
        buf = bytearray()
        announced = False

        async def finalize() -> None:
            nonlocal announced
            if not buf:
                await ws.send_json({"type": "final", "text": ""})
            else:
                try:
                    text = _transcribe_pcm(bytes(buf))
                except Exception as exc:
                    await ws.send_json(
                        {"type": "error", "detail": f"transcription failed: {exc}"}
                    )
                else:
                    await ws.send_json({"type": "final", "text": text})
            buf.clear()
            vad.reset()
            announced = False

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    chunk: bytes = msg["bytes"]
                    if len(buf) + len(chunk) > WS_MAX_BUFFER_BYTES:
                        await finalize()  # never grow unbounded; fresh buffer below
                    buf += chunk
                    state = vad.feed(chunk)
                    if state == "speech":
                        if not announced:
                            announced = True
                            await ws.send_json({"type": "speech_started"})
                    elif state == "endpoint":
                        await finalize()
                elif msg.get("text") is not None:
                    try:
                        ctrl = json.loads(msg["text"])
                    except ValueError:
                        await ws.send_json(
                            {"type": "error", "detail": "expected JSON control message"}
                        )
                        continue
                    kind = ctrl.get("type") if isinstance(ctrl, dict) else None
                    if kind == "stop":
                        await finalize()
                    elif kind == "abort":
                        buf.clear()
                        vad.reset()
                        announced = False
                        await ws.send_json({"type": "aborted"})
                    else:
                        await ws.send_json(
                            {"type": "error", "detail": f"unknown message type: {kind}"}
                        )
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/tts")
    async def ws_tts(ws: WebSocket) -> None:
        """Send {"text", "voice"?, "speed"?}; receive meta + binary chunks + done."""
        await ws.accept()
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await ws.send_json(
                        {"type": "error", "detail": "expected JSON, got binary"}
                    )
                    continue
                try:
                    req = json.loads(msg.get("text") or "")
                except ValueError:
                    await ws.send_json(
                        {"type": "error", "detail": "expected JSON control message"}
                    )
                    continue
                text = req.get("text", "") if isinstance(req, dict) else ""
                if not isinstance(text, str) or not text.strip():
                    await ws.send_json(
                        {"type": "error", "detail": "text must not be empty"}
                    )
                    continue
                try:
                    audio = synthesize(
                        text.strip(),
                        req.get("voice"),
                        req.get("speed"),
                    )
                except Exception as exc:
                    await ws.send_json(
                        {"type": "error", "detail": f"synthesis failed: {exc}"}
                    )
                    continue
                if not audio:
                    await ws.send_json(
                        {"type": "error", "detail": "synthesis returned no audio"}
                    )
                    continue
                await ws.send_json(
                    {
                        "type": "meta",
                        "bytes": len(audio),
                        "content_type": "audio/wav",
                    }
                )
                for offset in range(0, len(audio), WS_TTS_CHUNK_BYTES):
                    await ws.send_bytes(audio[offset : offset + WS_TTS_CHUNK_BYTES])
                await ws.send_json({"type": "done"})
        except WebSocketDisconnect:
            pass

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS standalone voice server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--stt-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--tts-model", default="KittenML/kitten-tts-mini-0.8")
    parser.add_argument("--tts-voice", default="Jasper")
    parser.add_argument("--tts-speed", type=float, default=1.0)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn missing; run: python -m pip install -e \".[server]\""
        ) from exc

    app = create_app(
        stt_model=args.stt_model,
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        tts_speed=args.tts_speed,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
