"""Barge-in tests — faked audio/monitor, no mic/speakers."""
import wave

import pytest

import nexus.voice.audio as audio_mod
from nexus.voice.audio_loop import (
    BARGE_DEFAULTS,
    BargeDetector,
    VoiceLoop,
    wav_duration_s,
)


def make_loop(**overrides):
    kwargs = {
        "asr": object(),
        "synth": object(),
        "recordings": "/tmp/nexus-barge-test",
        "on_task": lambda text: "ok",
    }
    kwargs.update(overrides)
    return VoiceLoop(**kwargs)


def write_wav(path, frames=16000, rate=16000):
    import struct

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))


# --- detector ---


def test_detector_needs_consecutive_windows():
    det = BargeDetector(threshold=100.0, windows_needed=3)
    assert not det.update(50.0)
    assert not det.update(150.0)
    assert not det.update(150.0)
    assert det.update(150.0)  # third consecutive hot window
    det.reset()
    assert not det.update(150.0)


def test_detector_drops_streak_on_cool_window():
    det = BargeDetector(threshold=100.0, windows_needed=2)
    assert not det.update(150.0)
    assert not det.update(10.0)
    assert not det.update(150.0)
    assert det.update(150.0)


def test_detector_rejects_bad_config():
    with pytest.raises(ValueError):
        BargeDetector(threshold=1.0, windows_needed=0)


# --- wav duration ---


def test_wav_duration(tmp_path):
    wav = tmp_path / "one.wav"
    write_wav(wav, frames=16000, rate=16000)
    assert wav_duration_s(wav) == pytest.approx(1.0)
    with pytest.raises(Exception):
        wav_duration_s(tmp_path / "missing.wav")


# --- barge config ---


def test_barge_config_defaults_and_overrides():
    assert make_loop()._barge_config() == BARGE_DEFAULTS
    loop = make_loop(voice_cfg={"barge_in": {"enabled": False, "bogus": 1}})
    cfg = loop._barge_config()
    assert cfg["enabled"] is False
    assert cfg["threshold"] == BARGE_DEFAULTS["threshold"]
    assert "bogus" not in cfg


# --- interruptible speak ---


def test_custom_speak_fn_is_blocking_and_uncut(tmp_path):
    seen: list = []
    loop = make_loop(speak_fn=lambda text, path: seen.append((text, path)))
    assert loop._speak_interruptible("hi", tmp_path / "r.wav") is False
    assert seen[0][0] == "hi"


def test_disabled_barge_plays_blocking(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(
        audio_mod, "play_wav", lambda path, blocking=True: calls.append(blocking)
    )
    monkeypatch.setattr(audio_mod, "stop_playback", lambda: calls.append("stop"))

    class Synth:
        def synthesize(self, text, output):
            return output

    loop = make_loop(voice_cfg={"barge_in": {"enabled": False}})
    loop.synth = Synth()
    assert loop._speak_interruptible("hi", tmp_path / "r.wav") is False
    assert calls == [True]  # blocking play, no purge


def test_barge_cut_and_playthrough(monkeypatch, tmp_path):
    import sys
    import types

    # async path needs the availability probe to pass; mic itself is faked
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    calls: list = []
    monkeypatch.setattr(
        audio_mod, "play_wav", lambda path, blocking=True: calls.append(blocking)
    )
    monkeypatch.setattr(audio_mod, "stop_playback", lambda: calls.append("stop"))

    class Synth:
        def synthesize(self, text, output):
            write_wav(output)
            return output

    loop = make_loop()
    loop.synth = Synth()
    out = tmp_path / "reply.wav"

    monkeypatch.setattr(loop, "_monitor_for_barge", lambda *a, **k: True)
    assert loop._speak_interruptible("hi", out) is True
    assert calls == [False, "stop"]  # async play, then purge

    calls.clear()
    monkeypatch.setattr(loop, "_monitor_for_barge", lambda *a, **k: False)
    assert loop._speak_interruptible("hi", out) is False
    assert calls == [False, "stop"]


def test_synth_failure_propagates(monkeypatch, tmp_path):
    class Synth:
        def synthesize(self, text, output):
            raise RuntimeError("tts down")

    loop = make_loop()
    loop.synth = Synth()
    with pytest.raises(RuntimeError, match="tts down"):
        loop._speak_interruptible("hi", tmp_path / "r.wav")
