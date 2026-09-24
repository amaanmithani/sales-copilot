"""Audio I/O and a chunked stream that can simulate real-time arrival."""

from __future__ import annotations

import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
Audio = NDArray[np.float32]


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


@dataclass(frozen=True)
class Chunk:
    start_s: float  # stream time of first sample
    samples: Audio
    received_wall: float  # clock time at which the chunk became available

    @property
    def end_s(self) -> float:
        return self.start_s + len(self.samples) / SAMPLE_RATE


def resample_linear(x: Audio, sr_in: int, sr_out: int = SAMPLE_RATE) -> Audio:
    if sr_in == sr_out or len(x) == 0:
        return x.astype(np.float32, copy=False)
    n_out = round(len(x) * sr_out / sr_in)
    t_in = np.arange(len(x), dtype=np.float64) / sr_in
    t_out = np.arange(n_out, dtype=np.float64) / sr_out
    return np.interp(t_out, t_in, x).astype(np.float32)


def read_wav(path: str | Path) -> Audio:
    """Read a PCM WAV (8/16/32-bit int), downmix to mono, resample to 16 kHz float32."""
    with wave.open(str(path), "rb") as wf:
        sr, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise ValueError(f"unsupported sample width: {width} bytes")
    data = np.frombuffer(raw, dtype=dtypes[width]).astype(np.float32)
    if width == 1:
        data = (data - 128.0) / 128.0
    else:
        data /= float(2 ** (8 * width - 1))
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return resample_linear(data, sr)


def write_wav(path: str | Path, audio: Audio, sr: int = SAMPLE_RATE) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def iter_chunks(
    audio: Audio,
    chunk_ms: int = 100,
    realtime: bool = True,
    clock: Clock | None = None,
) -> Iterator[Chunk]:
    """Yield fixed-size chunks. In real-time mode a chunk is released only once its last
    sample would have been captured (t0 + chunk end); if the consumer is behind, chunks are
    released immediately, exactly like a capture buffer that has filled up."""
    clk = clock or MonotonicClock()
    step = max(1, SAMPLE_RATE * chunk_ms // 1000)
    t0 = clk.now()
    for i in range(0, len(audio), step):
        piece = audio[i : i + step]
        start_s = i / SAMPLE_RATE
        if realtime:
            clk.sleep(t0 + start_s + len(piece) / SAMPLE_RATE - clk.now())
        yield Chunk(start_s=start_s, samples=piece, received_wall=clk.now())


def iter_mic_chunks(chunk_ms: int = 100, clock: Clock | None = None) -> Iterator[Chunk]:
    """Live microphone chunks (needs the optional `mic` extra). Not exercised in CI."""
    import queue

    import sounddevice as sd

    clk = clock or MonotonicClock()
    q: queue.Queue[Audio] = queue.Queue()
    step = SAMPLE_RATE * chunk_ms // 1000

    def _cb(indata: NDArray[np.float32], frames: int, t: object, status: object) -> None:
        q.put(indata[:, 0].copy())

    pos = 0
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=step, callback=_cb
    ):
        while True:
            piece = q.get()
            yield Chunk(start_s=pos / SAMPLE_RATE, samples=piece, received_wall=clk.now())
            pos += len(piece)
