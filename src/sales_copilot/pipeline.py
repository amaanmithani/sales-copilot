"""Streaming pipeline: chunks -> endpointer -> ASR -> cue engine, with latency accounting.

Latency definition (per segment / cue):
    speech_end_wall = wall-clock time at which the last voiced sample of the segment was
                      captured. In real-time replay this is t0 + end_s, so any backlog
                      (ASR slower than real time) is included.
    shown_wall      = wall-clock time after classification (the card is ready to render).
    e2e latency     = shown_wall - speech_end_wall
                    = endpoint wait (trailing-silence detection + chunk quantisation)
                      + queueing + ASR + classification.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from sales_copilot.asr import Transcriber
from sales_copilot.audio import Chunk, Clock, MonotonicClock
from sales_copilot.engine import CueCard, CueEngine
from sales_copilot.vad import Endpointer, Utterance


@dataclass(frozen=True)
class SegmentEvent:
    index: int
    text: str
    start_s: float
    end_s: float
    audio_s: float
    asr_s: float  # ASR wall time
    classify_s: float
    endpoint_wait_s: float  # endpoint_s - end_s (stream time)
    e2e_s: float  # speech end -> cues ready
    forced: bool
    cards: tuple[CueCard, ...]
    labels: tuple[str, ...] = ()  # raw classifier labels for this segment (before cooldown)


@dataclass
class RunStats:
    audio_s: float = 0.0
    asr_s: float = 0.0
    wall_s: float = 0.0
    segments: list[SegmentEvent] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        """ASR real-time factor: ASR compute time / audio duration (lower is faster)."""
        return self.asr_s / self.audio_s if self.audio_s else 0.0


class Copilot:
    def __init__(
        self,
        transcriber: Transcriber,
        engine: CueEngine,
        endpointer: Endpointer | None = None,
        clock: Clock | None = None,
        realtime: bool = True,
    ) -> None:
        self.transcriber = transcriber
        self.engine = engine
        self.endpointer = endpointer or Endpointer()
        self.clock = clock or MonotonicClock()
        self.realtime = realtime
        self.stats = RunStats()

    def _process(
        self, utt: Utterance, t0: float, last_chunk: Chunk | None, index: int
    ) -> SegmentEvent:
        if self.realtime:
            speech_end_wall = t0 + utt.end_s
        else:  # virtual: when this sample would have arrived had we been paced
            ref_wall = last_chunk.received_wall if last_chunk else self.clock.now()
            ref_s = last_chunk.end_s if last_chunk else utt.endpoint_s
            speech_end_wall = ref_wall - (ref_s - utt.end_s)
        a0 = self.clock.now()
        text = self.transcriber.transcribe(utt.audio)
        a1 = self.clock.now()
        n_before = len(self.engine.classify_times_s)
        n_lab = len(self.engine.raw_labels)
        cards = self.engine.on_segment(text, utt.end_s)
        labels = tuple(
            sorted(self.engine.raw_labels[n_lab]) if len(self.engine.raw_labels) > n_lab else ()
        )
        shown = self.clock.now()
        classify_s = sum(self.engine.classify_times_s[n_before:])
        audio_s = len(utt.audio) / 16_000
        self.stats.asr_s += a1 - a0
        return SegmentEvent(
            index=index,
            text=text,
            start_s=utt.start_s,
            end_s=utt.end_s,
            audio_s=audio_s,
            asr_s=a1 - a0,
            classify_s=classify_s,
            endpoint_wait_s=utt.endpoint_s - utt.end_s,
            e2e_s=shown - speech_end_wall,
            forced=utt.forced,
            cards=tuple(cards),
            labels=labels,
        )

    def run(
        self, chunks: Iterable[Chunk], on_event: Callable[[SegmentEvent], None] | None = None
    ) -> Iterator[SegmentEvent]:
        t0 = self.clock.now()
        last: Chunk | None = None
        idx = 0
        for chunk in chunks:
            last = chunk
            self.stats.audio_s = chunk.end_s
            for utt in self.endpointer.push(chunk.samples):
                ev = self._process(utt, t0, last, idx)
                idx += 1
                self.stats.segments.append(ev)
                if on_event:
                    on_event(ev)
                yield ev
        for utt in self.endpointer.flush():
            ev = self._process(utt, t0, last, idx)
            idx += 1
            self.stats.segments.append(ev)
            if on_event:
                on_event(ev)
            yield ev
        self.stats.wall_s = self.clock.now() - t0
