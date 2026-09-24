"""Linux voice-path tests — faked sounddevice, no HW/mic."""
import os
import struct
import sys
import types
import wave

import pytest

import nexus.voice.audio as audio_mod
from nexus.voice.audio import (
    MicRecorder,
    _read_wav_mono,
    _sounddevice,
    is_windows,
    play_wav,
    stop_playback,
)
from nexus.voice.audio_loop import consume_key as loop_consume_key
from nexus.voice.audio_loop import key_pressed as loop_key_pressed


def write_wav(path, samples, nchannels=1, rate=16000):
    flat = [s for frame in samples for s in frame] if nchannels > 1 else samples
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(nchannels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{len(flat)}h", *flat))


def make_fake_sd(calls):
    mod = types.ModuleType("sounddevice")

    def play(data, rate):
        calls.append(("play", len(data), rate))

    def wait():
        calls.append(("wait",))

    def stop():
        calls.append(("stop",))

    mod.play, mod.wait, mod.stop = play, wait, stop
    return mod


# --- wav reading ---


def test_read_wav_mono_passthrough(tmp_path):
    wav = tmp_path / "m.wav"
    write_wav(wav, [0, 1000, -1000, 3000])
    data, rate = _read_wav_mono(wav)
    assert rate == 16000
    assert data.tolist() == [0, 1000, -1000, 3000]


def test_read_wav_stereo_averaged_to_mono(tmp_path):
    wav = tmp_path / "s.wav"
    write_wav(wav, [(1000, 3000), (-1000, 1000)], nchannels=2)
    data, rate = _read_wav_mono(wav)
    assert rate == 16000
    assert data.tolist() == [2000, 0]


# --- install hints ---


def test_sounddevice_missing_gives_install_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match="pip install"):
        _sounddevice()
    with pytest.raises(RuntimeError, match="pip install"):
        MicRecorder().start(tmp_path / "x.wav")


# --- playback dispatch ---


def test_play_wav_linux_branch(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setitem(sys.modules, "sounddevice", make_fake_sd(calls))
    monkeypatch.setattr(audio_mod, "is_windows", lambda: False)
    wav = tmp_path / "p.wav"
    write_wav(wav, [100] * 1600)

    play_wav(wav, blocking=True)
    assert calls[0] == ("play", 1600, 16000)
    assert ("wait",) in calls

    calls.clear()
    play_wav(wav, blocking=False)
    assert calls[0] == ("play", 1600, 16000)
    assert ("wait",) not in calls

    calls.clear()
    stop_playback()
    assert calls == [("stop",)]


def test_play_wav_windows_branch(monkeypatch, tmp_path):
    calls: list = []
    winsound = types.ModuleType("winsound")
    winsound.SND_FILENAME = 1
    winsound.SND_ASYNC = 2
    winsound.SND_PURGE = 4

    def PlaySound(arg, flags):
        calls.append((arg, flags))

    winsound.PlaySound = PlaySound
    monkeypatch.setitem(sys.modules, "winsound", winsound)
    monkeypatch.setattr(audio_mod, "is_windows", lambda: True)
    wav = tmp_path / "p.wav"
    write_wav(wav, [1, 2, 3])

    play_wav(wav, blocking=True)
    assert calls[-1] == (str(wav), 1)
    play_wav(wav, blocking=False)
    assert calls[-1] == (str(wav), 3)
    stop_playback()
    assert calls[-1] == (None, 4)  # PlaySound(None, SND_PURGE)


def test_is_windows_reflects_platform():
    assert is_windows() == (sys.platform == "win32")


# --- key helpers: never hang, never raise ---


def test_key_helpers_are_safe_without_console():
    for fn in (loop_key_pressed, loop_consume_key):
        result = fn()
        assert result is None or isinstance(result, bool)


def test_key_pressed_returns_bool():
    assert isinstance(loop_key_pressed(), bool)


class FakeStdin:
    def fileno(self):
        return 0


def test_stdin_eof_is_never_a_keypress(monkeypatch):
    import select as select_mod

    from nexus.voice import audio_loop as audio_loop_mod

    audio_loop_mod._KEY_POLLER.reset()
    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.setattr(select_mod, "select", lambda r, w, x, timeout=0: (r, [], []))
    monkeypatch.setattr(os, "read", lambda fd, n: b"")
    assert loop_key_pressed() is False
    assert loop_key_pressed() is False  # sticky EOF
    loop_consume_key()  # must not raise
    audio_loop_mod._KEY_POLLER.reset()


def test_stdin_keypress_detected_then_consumed(monkeypatch):
    import select as select_mod

    from nexus.voice import audio_loop as audio_loop_mod

    audio_loop_mod._KEY_POLLER.reset()
    reads = [b"\n"]
    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.setattr(select_mod, "select", lambda r, w, x, timeout=0: (r, [], []))
    monkeypatch.setattr(os, "read", lambda fd, n: reads.pop(0) if reads else b"")
    assert loop_key_pressed() is True
    loop_consume_key()
    # buffer consumed; next read hits EOF -> never a keypress again
    assert loop_key_pressed() is False
    audio_loop_mod._KEY_POLLER.reset()
