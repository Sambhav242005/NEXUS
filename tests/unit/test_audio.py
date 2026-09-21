"""Windows audio tests — no mic needed except live-capture test (skipped without HW)."""
import wave
import numpy as np
import pytest
from pathlib import Path
from nexus.voice.audio import MicRecorder, list_mics, default_mic, transcribe_wav
from nexus.ui.recorder import TaskRecorder


def test_wav_roundtrip(tmp_path: Path):
    p = tmp_path / "tone.wav"
    tone = (np.sin(2 * np.pi * 440 * np.arange(1600) / 16000) * 10000).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(tone.tobytes())
    with wave.open(str(p), "rb") as r:
        assert r.getnchannels() == 1
        assert r.getframerate() == 16000
        assert len(r.readframes(r.getnframes())) > 0


def test_mic_stop_without_start():
    with pytest.raises(RuntimeError):
        MicRecorder().stop()


def test_empty_or_malformed_recording_is_no_speech(tmp_path: Path):
    empty = tmp_path / "empty.wav"
    with wave.open(str(empty), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
    assert transcribe_wav(empty) == ""

    malformed = tmp_path / "malformed.wav"
    malformed.write_bytes(b"not a wav")
    assert transcribe_wav(malformed) == ""


def test_recorder_with_fake_mic(tmp_path: Path):
    class FakeMic:
        device = None

        def __init__(self):
            self.started = None

        def start(self, path):
            self.started = Path(path)
            self.started.parent.mkdir(parents=True, exist_ok=True)
            self.started.write_bytes(b"fake")
            return self.started

        def stop(self):
            return self.started

    class FakeCtl:
        def execute(self, action):
            class R:
                detail = "screenshot-stub"
            return R()

    rec = TaskRecorder(FakeCtl(), workdir=tmp_path, interval=0.05, mic=FakeMic())
    r = rec.start("hi")
    import time
    time.sleep(0.12)
    out = rec.stop()
    assert out.audio_path is not None and out.audio_path.exists()


def test_list_mics_live():
    try:
        mics = list_mics()
    except Exception as e:
        pytest.skip(f"no audio backend: {e}")
    assert isinstance(mics, list)
    _ = default_mic()  # None ok, must not raise
