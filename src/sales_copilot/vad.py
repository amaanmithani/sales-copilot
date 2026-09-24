"""Energy-based voice activity detection + utterance endpointing for a chunked stream."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from sales_copilot.audio import SAMPLE_RATE, Audio


@dataclass(frozen=True)
class Utterance:
    start_s: float  # stream time speech started (incl. pre-roll)
    end_s: float  # stream time speech ended (last voiced frame)
    endpoint_s: float  # stream time at which the endpoint was decided
    audio: Audio
    forced: bool = False  # cut because max length reached


@dataclass
class VadConfig:
    frame_ms: int = 30
    threshold_db: float = -42.0  # dBFS; frames above are "voiced"
    min_speech_ms: int = 90  # consecutive voiced time needed to open an utterance
    min_silence_ms: int = 450  # trailing silence that closes an utterance
    preroll_ms: int = 150
    max_utterance_s: float = 12.0


def frame_db(frame: Audio) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if len(frame) else 0.0
    return 20.0 * float(np.log10(rms + 1e-10))


class Endpointer:
    """Feed arbitrary-sized chunks; returns finished utterances as soon as they end."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self.cfg = config or VadConfig()
        self.frame = SAMPLE_RATE * self.cfg.frame_ms // 1000
        self._pending = np.zeros(0, dtype=np.float32)
        self._t = 0.0  # stream time of the start of _pending
        self._pre: deque[Audio] = deque(maxlen=max(1, self.cfg.preroll_ms // self.cfg.frame_ms))
        self._voiced_run = 0
        self._silence_run = 0
        self._in_speech = False
        self._buf: list[Audio] = []
        self._utt_start = 0.0
        self._last_voiced_end = 0.0

    def _frames_for(self, ms: int) -> int:
        return max(1, ms // self.cfg.frame_ms)

    def _close(self, endpoint_s: float, forced: bool) -> Utterance:
        # keep up to the last voiced frame plus one frame of tail
        tail = self.frame / SAMPLE_RATE
        keep_until = min(self._last_voiced_end + tail, endpoint_s)
        audio = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)
        n_keep = round((keep_until - self._utt_start) * SAMPLE_RATE)
        utt = Utterance(
            start_s=self._utt_start,
            end_s=self._last_voiced_end,
            endpoint_s=endpoint_s,
            audio=audio[: max(0, n_keep)],
            forced=forced,
        )
        self._in_speech = False
        self._buf = []
        self._silence_run = 0
        self._voiced_run = 0
        self._pre.clear()
        return utt

    def push(self, samples: Audio) -> list[Utterance]:
        out: list[Utterance] = []
        self._pending = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n = self.frame
        while len(self._pending) >= n:
            fr, self._pending = self._pending[:n], self._pending[n:]
            f_start = self._t
            f_end = f_start + n / SAMPLE_RATE
            self._t = f_end
            voiced = frame_db(fr) > self.cfg.threshold_db
            if not self._in_speech:
                self._pre.append(fr)
                self._voiced_run = self._voiced_run + 1 if voiced else 0
                if self._voiced_run >= self._frames_for(self.cfg.min_speech_ms):
                    self._in_speech = True
                    self._buf = list(self._pre)
                    self._utt_start = f_end - len(self._pre) * n / SAMPLE_RATE
                    self._last_voiced_end = f_end
                    self._silence_run = 0
                continue
            self._buf.append(fr)
            if voiced:
                self._silence_run = 0
                self._last_voiced_end = f_end
            else:
                self._silence_run += 1
            if self._silence_run >= self._frames_for(self.cfg.min_silence_ms):
                out.append(self._close(f_end, forced=False))
            elif f_end - self._utt_start >= self.cfg.max_utterance_s:
                self._last_voiced_end = f_end
                out.append(self._close(f_end, forced=True))
        return out

    def flush(self) -> list[Utterance]:
        """Close any open utterance at end of stream."""
        if self._in_speech:
            return [self._close(self._t, forced=True)]
        return []
