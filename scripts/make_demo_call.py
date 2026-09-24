"""Synthesise the demo sales calls with macOS `say` (SYNTHETIC audio, two TTS voices).

Writes data/audio/<call>.wav (16 kHz mono) and data/demo/<call>.segments.json with the gold
line timings and labels. Deterministic: fixed voices, fixed rate, fixed gaps.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from sales_copilot.audio import SAMPLE_RATE, read_wav, write_wav

ROOT = Path(__file__).resolve().parents[1]
VOICES = {
    "call_a": {"rep": "Samantha", "prospect": "Daniel"},
    "call_b": {"rep": "Karen", "prospect": "Rishi"},
}
GAPS_S = (0.9, 1.2, 1.0, 1.4, 0.8)  # silence between turns (cycled)
RATE_WPM = 185


def tts(text: str, voice: str, tmp: Path) -> np.ndarray:
    aiff, wav = tmp / "line.aiff", tmp / "line.wav"
    subprocess.run(["say", "-v", voice, "-r", str(RATE_WPM), "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(aiff), str(wav)],
        check=True,
    )
    return read_wav(wav)


def build(call: str) -> None:
    rows = list(csv.DictReader((ROOT / "data/demo" / f"{call}.tsv").open(), delimiter="\t"))
    rng = np.random.default_rng(0)
    parts: list[np.ndarray] = [np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)]
    t = 0.5
    segs = []
    with tempfile.TemporaryDirectory() as td:
        for i, row in enumerate(rows):
            audio = tts(row["text"], VOICES[call][row["speaker"]], Path(td))
            # trim TTS leading/trailing digital silence so gold times are speech times
            voiced = np.flatnonzero(np.abs(audio) > 1e-3)
            audio = audio[voiced[0] : voiced[-1] + 1] if len(voiced) else audio
            segs.append(
                {
                    "index": i,
                    "speaker": row["speaker"],
                    "labels": [x for x in row["labels"].split(";") if x and x != "none"],
                    "text": row["text"],
                    "start_s": round(t, 3),
                    "end_s": round(t + len(audio) / SAMPLE_RATE, 3),
                }
            )
            gap = GAPS_S[i % len(GAPS_S)]
            parts += [audio, np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)]
            t += len(audio) / SAMPLE_RATE + gap
    mix = np.concatenate(parts)
    mix = mix + rng.normal(0, 10 ** (-60 / 20), size=len(mix)).astype(np.float32)  # -60 dBFS hiss
    (ROOT / "data/audio").mkdir(parents=True, exist_ok=True)
    write_wav(ROOT / "data/audio" / f"{call}.wav", mix)
    meta = {
        "call": call,
        "synthetic": True,
        "tts": "macOS say",
        "voices": VOICES[call],
        "rate_wpm": RATE_WPM,
        "duration_s": round(len(mix) / SAMPLE_RATE, 3),
        "segments": segs,
    }
    (ROOT / "data/demo" / f"{call}.segments.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{call}: {meta['duration_s']}s, {len(segs)} turns")


if __name__ == "__main__":
    for c in VOICES:
        build(c)
