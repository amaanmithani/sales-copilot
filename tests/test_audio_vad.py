from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from sales_copilot.audio import SAMPLE_RATE, iter_chunks, read_wav, resample_linear, write_wav
from sales_copilot.vad import Endpointer, VadConfig, frame_db

from .conftest import FakeClock, silence, tone


def test_wav_roundtrip(tmp_path: Path) -> None:
    x = tone(0.5)
    p = tmp_path / "a.wav"
    write_wav(p, x)
    y = read_wav(p)
    assert len(y) == len(x)
    assert np.max(np.abs(x - y)) < 1e-3


def test_read_wav_stereo_8bit_and_resample(tmp_path: Path) -> None:
    p = tmp_path / "s.wav"
    n = 8000
    frames = np.full((n, 2), 200, dtype=np.uint8)  # 8-bit unsigned, constant offset
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(frames.tobytes())
    y = read_wav(p)
    assert len(y) == SAMPLE_RATE  # 1 s resampled to 16 kHz
    assert np.allclose(y, (200 - 128) / 128, atol=1e-4)


def test_read_wav_rejects_24bit(tmp_path: Path) -> None:
    p = tmp_path / "x.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 30)
    with pytest.raises(ValueError):
        read_wav(p)


def test_resample_identity_and_empty() -> None:
    x = tone(0.1)
    assert resample_linear(x, SAMPLE_RATE) is x or np.array_equal(resample_linear(x, 16000), x)
    assert len(resample_linear(np.zeros(0, dtype=np.float32), 8000)) == 0


def test_iter_chunks_realtime_pacing() -> None:
    clk = FakeClock()
    t0 = clk.now()
    chunks = list(iter_chunks(tone(1.0), chunk_ms=100, realtime=True, clock=clk))
    assert len(chunks) == 10
    # chunk k is released when its last sample has been "captured"
    for c in chunks:
        assert c.received_wall == pytest.approx(t0 + c.end_s)


def test_iter_chunks_fast_mode_does_not_sleep() -> None:
    clk = FakeClock()
    t0 = clk.now()
    chunks = list(iter_chunks(tone(1.0), chunk_ms=250, realtime=False, clock=clk))
    assert len(chunks) == 4 and clk.now() == t0


def test_frame_db() -> None:
    assert frame_db(np.zeros(10, dtype=np.float32)) < -150
    assert frame_db(np.ones(10, dtype=np.float32)) == pytest.approx(0.0, abs=1e-6)
    assert frame_db(np.zeros(0, dtype=np.float32)) < -150


def test_endpointer_finds_two_utterances(two_bursts: np.ndarray) -> None:
    ep = Endpointer()
    utts = []
    for i in range(0, len(two_bursts), 1600):  # 100 ms chunks
        utts += ep.push(two_bursts[i : i + 1600])
    utts += ep.flush()
    assert len(utts) == 2
    a, b = utts
    assert a.end_s == pytest.approx(1.5, abs=0.04)
    assert b.end_s == pytest.approx(3.3, abs=0.04)
    assert a.start_s == pytest.approx(0.5 - 0.15, abs=0.1)  # includes pre-roll
    assert a.endpoint_s - a.end_s == pytest.approx(0.45, abs=0.04)
    assert not a.forced and not b.forced
    assert abs(len(a.audio) / SAMPLE_RATE - (a.end_s - a.start_s)) < 0.05


def test_endpointer_forces_long_utterances_and_flushes() -> None:
    ep = Endpointer(VadConfig(max_utterance_s=2.0))
    utts = ep.push(tone(5.0))
    assert len(utts) >= 2 and all(u.forced for u in utts)
    tail = ep.flush()
    assert len(tail) == 1 and tail[0].forced
    assert ep.flush() == []


def test_endpointer_ignores_clicks() -> None:
    ep = Endpointer()
    x = np.concatenate([silence(0.5), tone(0.03), silence(1.0)])
    assert ep.push(x) == [] and ep.flush() == []
