from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sales_copilot.audio import SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.t += seconds


class StubTranscriber:
    """Returns scripted texts in order and advances a fake clock by `cost_s` per call."""

    name = "stub"

    def __init__(self, texts: list[str], clock: FakeClock | None = None, cost_s: float = 0.2):
        self.texts = list(texts)
        self.clock = clock
        self.cost_s = cost_s
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> str:
        if self.clock:
            self.clock.sleep(self.cost_s)
        text = self.texts[self.calls] if self.calls < len(self.texts) else ""
        self.calls += 1
        return text


def tone(seconds: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def two_bursts() -> np.ndarray:
    """0.5 s silence, 1.0 s voice, 1.0 s silence, 0.8 s voice, 1.0 s silence."""
    return np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(0.8), silence(1.0)])
