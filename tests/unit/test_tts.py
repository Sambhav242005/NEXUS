from pathlib import Path

import pytest

from nexus.voice.tts import KittenSynthesizer


def test_kitten_synthesizes_to_requested_wav(monkeypatch, tmp_path: Path):
    class FakeModel:
        def generate_to_file(self, text, output, **kwargs):
            assert text == "hello"
            assert kwargs["voice"] == "Jasper"
            Path(output).write_bytes(b"RIFF-fake")

    synthesizer = KittenSynthesizer()
    monkeypatch.setattr(synthesizer, "_load", lambda: FakeModel())
    output = synthesizer.synthesize("hello", tmp_path / "reply.wav")
    assert output.read_bytes() == b"RIFF-fake"


def test_piper_rejects_empty_text(tmp_path: Path):
    with pytest.raises(ValueError):
        KittenSynthesizer().synthesize("  ", tmp_path / "reply.wav")


def test_kitten_reports_missing_dependency(monkeypatch, tmp_path: Path):
    synthesizer = KittenSynthesizer()
    def missing():
        raise RuntimeError("KittenTTS is not installed")
    monkeypatch.setattr(synthesizer, "_load", missing)
    with pytest.raises(RuntimeError, match="KittenTTS is not installed"):
        synthesizer.synthesize("hello", tmp_path / "reply.wav")
