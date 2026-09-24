"""Speak-while-busy voice loop.

The listener records/transcribes continuously; a worker thread executes
commands FIFO. Speech arriving mid-task is queued, never lost, never
acted on twice. Bounded queue: when full the newest utterance is dropped
and the user is told to repeat it (logged, never silent).

Mic capture stays Windows-only (see voice.audio); the queue, worker, and
phrase helpers are pure Python and unit-tested on any OS.
"""
from __future__ import annotations

import itertools
import math
import os
import queue
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .dsp import rms_level

EXIT_PHRASES = frozenset({"exit", "quit", "stop", "stop listening", "goodbye"})

BARGE_WINDOW_S = 0.05  # mic sampling slice while TTS plays
BARGE_DEFAULTS = {"enabled": True, "threshold": 1500.0, "min_speech_ms": 300}


def normalize_utterance(text: str) -> str:
    return text.strip().lower().rstrip(".!?")


def is_exit_phrase(text: str) -> bool:
    return normalize_utterance(text) in EXIT_PHRASES


def ack_text(position: int) -> str:
    return f"Got it, queued as number {position}. I'll get to it next."


def key_pressed() -> bool:
    """Enter key waiting? Non-blocking, never raises, never blocks.

    Windows: msvcrt. Linux/macOS: select on stdin (line buffering still
    delivers the Enter keystroke). False when stdin is not a console.
    EOF on a pipe counts as *no* key, so piped stdin never fake-stops
    a recording.
    """
    if sys.platform == "win32":
        try:
            import msvcrt

            return bool(msvcrt.kbhit())
        except (ImportError, OSError):
            return False
    return _KEY_POLLER.pressed()


def consume_key() -> None:
    """Swallow one pending keystroke. Never raises."""
    if sys.platform == "win32":
        try:
            import msvcrt

            msvcrt.getwch()
        except (ImportError, OSError):
            pass
        return
    _KEY_POLLER.consume()


class _StdinKeyPoller:
    """EOF-aware stdin key poller (any key counts, like msvcrt.kbhit)."""

    def __init__(self) -> None:
        self._eof = False
        self._buffered = False

    def reset(self) -> None:
        self._eof = False
        self._buffered = False

    def pressed(self) -> bool:
        if self._buffered:
            return True
        if self._eof:
            return False
        try:
            import select

            readable = select.select([sys.stdin], [], [], 0)[0]
        except (OSError, ValueError):
            return False
        if not readable:
            return False
        try:
            data = os.read(sys.stdin.fileno(), 64)
        except OSError:
            return False
        if not data:
            self._eof = True  # pipe closed: never a keypress, forever
            return False
        self._buffered = True
        return True

    def consume(self) -> None:
        self._buffered = False


_KEY_POLLER = _StdinKeyPoller()


class BargeDetector:
    """Trigger the moment mic stays hot for N consecutive windows.

    Threshold is deliberately higher than the record VAD threshold:
    the mic also hears our own TTS through the speakers. Best with a
    headset; speaker bleed at high volume can false-trigger.
    """

    def __init__(self, *, threshold: float, windows_needed: int) -> None:
        if windows_needed < 1:
            raise ValueError("windows_needed must be >= 1")
        self.threshold = threshold
        self.windows_needed = windows_needed
        self._hot = 0

    def reset(self) -> None:
        self._hot = 0

    def update(self, level: float) -> bool:
        """Feed one window level. True the moment speech is confirmed."""
        self._hot = self._hot + 1 if level >= self.threshold else 0
        return self._hot >= self.windows_needed


def wav_duration_s(path: str | Path) -> float:
    """Playback length of a WAV file in seconds."""
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        if frames <= 0 or rate <= 0:
            raise ValueError(f"cannot determine duration of {path}")
        return frames / rate


@dataclass
class Command:
    seq: int
    text: str
    wav: str


class PendingQueue:
    """Bounded FIFO of voice commands. submit() never blocks."""

    def __init__(self, max_pending: int = 5) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self._queue: queue.Queue[Command] = queue.Queue(maxsize=max_pending)
        self._seq = itertools.count(1)
        self._seq_lock = threading.Lock()

    def submit(self, text: str, wav: str | Path) -> tuple[str, int]:
        """Enqueue; returns ("queued", position) or ("full", pending)."""
        with self._seq_lock:
            cmd = Command(next(self._seq), text, str(wav))
        try:
            self._queue.put_nowait(cmd)
        except queue.Full:
            return "full", self._queue.qsize()
        return "queued", self._queue.qsize()

    def get(self, timeout: float = 0.2) -> Command | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        self._queue.task_done()

    @property
    def pending(self) -> int:
        return self._queue.qsize()


def pump(
    commands: PendingQueue,
    *,
    on_task: Callable[[str], str],
    on_reply: Callable[[str, Command], None],
    stop_event: threading.Event,
    busy: threading.Event | None = None,
) -> int:
    """Worker: run queued commands FIFO until stopped. Returns processed count."""
    processed = 0
    while not stop_event.is_set():
        cmd = commands.get()
        if cmd is None:
            continue
        if busy is not None:
            busy.set()
        try:
            try:
                reply = on_task(cmd.text)
            except Exception as exc:  # one bad task never kills the worker
                print(f"[voice] task failed: {exc}")
                reply = "The task failed."
            try:
                on_reply(reply, cmd)
            except Exception as exc:
                print(f"[voice] reply failed: {exc}")
        finally:
            if busy is not None:
                busy.clear()
            commands.task_done()
            processed += 1
    return processed


class VoiceLoop:
    """Mic listener + worker wiring. run() blocks (Windows mic)."""

    def __init__(
        self,
        *,
        asr,
        synth,
        recordings: str | Path,
        voice_cfg: dict | None = None,
        device: int | None = None,
        max_pending: int = 5,
        on_task: Callable[[str], str],
        speak_fn: Callable[[str, Path], None] | None = None,
        mic_factory: Callable[[], object] | None = None,
    ) -> None:
        self.asr = asr
        self.synth = synth
        self.recordings = Path(recordings)
        self.voice_cfg = voice_cfg or {}
        self.device = device
        self.on_task = on_task
        self._speak_fn = speak_fn
        self._mic_factory = mic_factory
        self.commands = PendingQueue(max_pending=max_pending)
        self.stop_event = threading.Event()
        self.busy = threading.Event()

    def _barge_config(self) -> dict:
        raw = self.voice_cfg.get("barge_in", {})
        cfg = dict(BARGE_DEFAULTS)
        if isinstance(raw, dict):
            cfg.update({k: v for k, v in raw.items() if k in cfg})
        return cfg

    def _speak_interruptible(self, text: str, output: Path) -> bool:
        """Speak a reply. Returns True if cut short by barge-in.

        Barge-in = Enter key, or mic hot (user talking over the TTS).
        Cut speech is not captured; the loop records fresh right after.
        """
        if self._speak_fn is not None:  # tests / custom playback: blocking
            self._speak_fn(text, output)
            return False
        from . import audio as audio_mod

        path = self.synth.synthesize(text, output)
        cfg = self._barge_config()
        if not cfg["enabled"]:
            audio_mod.play_wav(path)
            return False
        try:
            import sounddevice  # noqa: F401 (availability probe)
        except Exception as exc:
            print(f"[voice] barge-in monitor unavailable ({exc}); playing through")
            audio_mod.play_wav(path)
            return False
        try:
            duration = wav_duration_s(path)
        except (OSError, wave.Error, ValueError) as exc:
            print(f"[voice] cannot size reply ({exc}); playing through")
            audio_mod.play_wav(path)
            return False
        audio_mod.play_wav(path, blocking=False)
        try:
            return self._monitor_for_barge(duration, cfg)
        finally:
            try:
                audio_mod.stop_playback()
            except Exception as exc:
                print(f"[voice] playback stop failed: {exc}")

    def _monitor_for_barge(self, duration_s: float, cfg: dict) -> bool:
        """Watch the mic while async playback runs. True on interrupt."""
        import sounddevice as sd

        windows_needed = max(
            1, math.ceil(float(cfg["min_speech_ms"]) / (BARGE_WINDOW_S * 1000))
        )
        detector = BargeDetector(
            threshold=float(cfg["threshold"]), windows_needed=windows_needed
        )
        start = time.monotonic()
        deadline = start + duration_s + 0.5
        try:
            with sd.InputStream(
                samplerate=16000, channels=1, device=self.device, dtype="int16"
            ) as stream:
                while time.monotonic() < deadline:
                    if key_pressed():
                        consume_key()
                        print("[voice] playback cut (key)")
                        return True
                    data, _ = stream.read(int(16000 * BARGE_WINDOW_S))
                    if detector.update(rms_level(bytes(data))):
                        print("[voice] barge-in heard; cutting playback")
                        return True
        except Exception as exc:
            # Never truncate a reply on monitor failure; wait it out instead.
            print(f"[voice] barge-in monitor failed ({exc}); playing through")
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        return False

    def _new_mic(self):
        if self._mic_factory is not None:
            return self._mic_factory()
        from . import audio as audio_mod

        return audio_mod.MicRecorder(
            device=self.device,
            silence_timeout=float(self.voice_cfg.get("silence_timeout", 1.0)),
            max_duration=float(self.voice_cfg.get("max_duration", 60.0)),
            speech_threshold=float(self.voice_cfg.get("speech_threshold", 500)),
        )

    def _record_once(self, recorder, wav: Path) -> Path:
        print("[voice] recording; speak now; auto-stops after silence (Enter also stops)")
        recorder.start(wav)
        try:
            while recorder.is_recording():
                if key_pressed():
                    consume_key()
                    recorder.stop()
                    break
                time.sleep(0.05)
        finally:
            if recorder.has_session():
                wav = recorder.stop()
        return wav

    def _on_reply(self, reply: str, cmd: Command) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Cut by barge-in is fine here: the listener is already recording
        # the interruption concurrently.
        self._speak_interruptible(reply, self.recordings / f"voice-{stamp}-reply.wav")

    def run(self) -> str:
        """Listen until an exit phrase or Ctrl-C. Returns 'exit'/'interrupted'."""
        print("[voice] loading STT model...")
        if hasattr(self.asr, "preload"):
            self.asr.preload()
        print("[voice] loading TTS model...")
        if hasattr(self.synth, "preload"):
            self.synth.preload()
        self.recordings.mkdir(parents=True, exist_ok=True)

        worker = threading.Thread(
            target=pump,
            kwargs={
                "commands": self.commands,
                "on_task": self.on_task,
                "on_reply": self._on_reply,
                "stop_event": self.stop_event,
                "busy": self.busy,
            },
            daemon=True,
        )
        worker.start()
        print("[voice] ready; say 'exit' to stop")
        try:
            while True:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                wav = self._record_once(
                    self._new_mic(), self.recordings / f"voice-{stamp}.wav"
                )
                try:
                    transcription = self.asr.transcribe(wav)
                    task = transcription.text
                except Exception as exc:
                    print(f"[voice] transcription failed; listening again: {exc}")
                    continue
                print(f"[voice] STT backend={transcription.backend} text={task!r}")
                if not task:
                    print("[voice] no speech recognized; listening again")
                    continue
                if is_exit_phrase(task):
                    print("[voice] stopping")
                    self._shutdown()
                    return "exit"
                status, position = self.commands.submit(task, wav)
                if status == "full":
                    print(f"[voice] queue full ({position} pending); utterance dropped")
                    self._speak_interruptible(
                        "I'm still working through earlier commands. "
                        "Please repeat that in a moment.",
                        self.recordings / f"voice-{stamp}-ack.wav",
                    )
                    continue
                if self.busy.is_set() or position > 1:
                    print(f"[voice] queued #{position}; worker busy")
                    if self._speak_interruptible(
                        ack_text(position),
                        self.recordings / f"voice-{stamp}-ack.wav",
                    ):
                        # Barge-in cut the ack; the interrupting speech was
                        # not captured, so record fresh right away.
                        print("[voice] ack cut; listening for the interruption")
                        continue
                else:
                    print("[voice] worker idle; running now")
        except KeyboardInterrupt:
            print("\n[voice] stopping")
            self._shutdown()
            return "interrupted"

    def _shutdown(self) -> None:
        self.stop_event.set()
        abandoned = self.commands.pending
        if abandoned:
            print(f"[voice] discarding {abandoned} queued command(s)")
