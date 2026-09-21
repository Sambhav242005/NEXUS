"""Transcribe every recordings/*/audio.wav with local Whisper (CUDA if present)."""
import sys
from pathlib import Path

sys.path.insert(0, "src")

import torch
from transformers import pipeline

USE_CUDA = torch.cuda.is_available()
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device="cuda:0" if USE_CUDA else "cpu",
    torch_dtype=torch.float16 if USE_CUDA else torch.float32,
)
print(f"asr device: {'cuda:0 ' + torch.cuda.get_device_name(0) if USE_CUDA else 'cpu'}")

base = Path("recordings")
wavs = sorted(base.glob("*/audio.wav"))
print(f"found {len(wavs)} audio files")
if not wavs:
    sys.exit(0)
for w in wavs:
    out = asr(str(w))
    text = out["text"].strip() if isinstance(out, dict) else str(out).strip()
    (w.parent / "transcript.txt").write_text(text or "(empty)", encoding="utf-8")
    print(f"--- {w.parent.name} ---")
    print(text or "(empty)")
print("done")
