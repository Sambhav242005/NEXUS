"""Tiny audio DSP helpers (stdlib only, shared by server + voice loop)."""
from __future__ import annotations

import math
from array import array


def rms_level(pcm: bytes) -> float:
    """RMS level of little-endian int16 mono PCM."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples:
        return 0.0
    if len(samples) > 8000:  # subsample; endpointing needs a level, not precision
        samples = samples[:: len(samples) // 8000]
    return math.sqrt(sum(s * s for s in samples) / len(samples))
