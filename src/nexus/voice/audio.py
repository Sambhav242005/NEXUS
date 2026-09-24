"""Mic capture + playback + local transcription (Windows + Linux).

Capture: sounddevice (PortAudio) -> 16kHz mono WAV via stdlib wave.
Playback: winsound on Windows, sounddevice stream on Linux.
Linux needs the PortAudio system library plus the voice extra::

    sudo apt install libportaudio2
    python -m pip install -e ".[voice]"

transcribe_wav/preload_transcriber work on any OS with torch/transformers
so the standalone voice server can reuse them.
"""
from __future__ import annotations
import platform
import threading
import time
import wave
from pathlib import Path

import numpy as np


def is_windows() -> bool:
    return platform.system() == "Windows"


def _sounddevice():
    """Import sounddevice with an actionable error when PortAudio is missing."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed; run: python -m pip install -e \".[voice]\""
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "PortAudio library not found (sounddevice needs it). "
            "Linux: sudo apt install libportaudio2. "
            "Windows: reinstall python from python.org (not the Store)."
        ) from exc
    return sd


def _read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono int16 samples + sample rate."""
    with wave.open(str(path), "rb") as wav:
        n_channels = wav.getnchannels()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    data = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    return data, rate


def list_mics() -> list[tuple[int, str]]:
    """Return [(device_index, name)] for input-capable devices."""
    sd = _sounddevice()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append((i, d["name"]))
    return out


def default_mic() -> int | None:
    """System default input device index, or None."""
    sd = _sounddevice()
    try:
        dev = sd.default.device[0]
        return int(dev) if dev is not None and dev >= 0 else None
    except Exception:
        return None


def play_wav(path: str | Path, blocking: bool = True) -> None:
    """Play a WAV file through default speakers. Blocking by default.

    Non-blocking playback can be cut with stop_playback().
    """
    if is_windows():
        import winsound
        flags = winsound.SND_FILENAME
        if not blocking:
            flags |= winsound.SND_ASYNC

        winsound.PlaySound(str(path), flags)
        return
    sd = _sounddevice()
    data, rate = _read_wav_mono(path)
    sd.play(data, rate)
    if blocking:
        sd.wait()


def stop_playback() -> None:
    """Stop any async playback."""
    if is_windows():
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
        return
    _sounddevice().stop()


class MicRecorder:
    """Background mic recorder. start() -> stop() writes WAV, returns path."""

    def __init__(self, samplerate: int = 16000, channels: int = 1,
                 device: int | None = None, silence_timeout: float = 1.2,
                 max_duration: float = 60.0, speech_threshold: float = 500.0):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.silence_timeout = silence_timeout
        self.max_duration = max_duration
        self.speech_threshold = speech_threshold
        self._frames: list = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.path: Path | None = None

    def start(self, path: str | Path) -> Path:
        if self._thread is not None:
            raise RuntimeError("mic already recording — stop first")
        _sounddevice()  # fail fast with an install hint, before threading
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._frames = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self.path

    def _run(self):
        try:
            sd = _sounddevice()
            started = time.monotonic()
            speech_started = False
            silent_since: float | None = None
            with sd.InputStream(samplerate=self.samplerate, channels=self.channels,
                                device=self.device, dtype="int16") as stream:
                while not self._stop.is_set() and time.monotonic() - started < self.max_duration:
                    data, _ = stream.read(int(self.samplerate * 0.1))
                    self._frames.append(data.copy())
                    level = float(np.sqrt(np.mean(np.square(data.astype(np.float32)))))
                    now = time.monotonic()
                    if level >= self.speech_threshold:
                        speech_started = True
                        silent_since = None
                    elif speech_started:
                        silent_since = silent_since or now
                        if now - silent_since >= self.silence_timeout:
                            self._stop.set()
                            break
        except Exception as e:
            self._frames.append(e)

    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def has_session(self) -> bool:
        return self._thread is not None and self.path is not None

    def stop(self) -> Path:
        if self._thread is None or self.path is None:
            raise RuntimeError("mic not recording")
        self._stop.set()
        self._thread.join(timeout=10)
        self._thread = None
        errs = [f for f in self._frames if isinstance(f, Exception)]
        if errs:
            raise RuntimeError(f"mic capture failed: {errs[0]}")
        audio = np.concatenate(self._frames, axis=0) if self._frames else np.zeros(
            (0, self.channels), dtype=np.int16)
        with wave.open(str(self.path), "wb") as w:
            w.setnchannels(self.channels)
            w.setsampwidth(2)
            w.setframerate(self.samplerate)
            w.writeframes(audio.tobytes())
        return self.path


_ASR = None
LAST_DEVICE = "unknown"


def _build_pipeline(model: str, force_cpu: bool = False):
    import torch
    from transformers import pipeline
    use_cuda = torch.cuda.is_available() and not force_cpu
    pipe = pipeline(
        "automatic-speech-recognition", model=model,
        device="cuda:0" if use_cuda else "cpu",
        dtype=torch.float16 if use_cuda else torch.float32,
    )
    return pipe, ("cuda:0" if use_cuda else "cpu")


def preload_transcriber(model: str = "openai/whisper-large-v3-turbo") -> str:
    """Load the transcription model now (call at startup) so first Stop is fast.

    Returns device string. Safe to call from a background thread.
    """
    global _ASR, LAST_DEVICE
    if _ASR is None:
        _ASR, LAST_DEVICE = _build_pipeline(model)
        return LAST_DEVICE
    return "cached"


def transcribe_wav(path: str | Path,
                   model: str = "openai/whisper-large-v3-turbo") -> str:
    """Speech-to-text on a WAV file. CUDA fp16 first, one CPU retry on failure.

    The CUDA init path can hit a meta-tensor error (transformers/torch
    friction) in long-lived processes — retry on CPU instead of failing Stop.
    LAST_DEVICE records which backend produced the result.
    """
    global _ASR, LAST_DEVICE
    audio_path = Path(path)
    try:
        with wave.open(str(audio_path), "rb") as wav:
            if wav.getnframes() == 0:
                (audio_path.parent / "transcript.txt").write_text("(empty)", encoding="utf-8")
                return ""
    except (OSError, wave.Error) as exc:
        # A stop can finalize an empty/corrupt capture when the input device
        # disappears.  Treat it as no speech so a persistent voice session
        # can keep listening instead of crashing.
        print(f"[voice] invalid recording skipped: {exc}")
        (audio_path.parent / "transcript.txt").write_text("(empty)", encoding="utf-8")
        return ""
    inference = {"language": "en", "task": "transcribe"} if "whisper" in model.lower() else {}
    try:
        if _ASR is None:
            _ASR, LAST_DEVICE = _build_pipeline(model)
        out = _ASR(str(path), **inference)
    except Exception as e:
        print(f"transcribe CUDA failed ({str(e)[:120]}), retrying on CPU…")
        _ASR, LAST_DEVICE = _build_pipeline(model, force_cpu=True)
        out = _ASR(str(path), **inference)
    text = out["text"].strip() if isinstance(out, dict) else str(out).strip()
    (Path(path).parent / "transcript.txt").write_text(text or "(empty)", encoding="utf-8")
    return text
