"""Cue engine: turns finalized transcript segments into de-duplicated cue cards."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sales_copilot.cues.base import CueClassifier
from sales_copilot.labels import CUE_TYPES
from sales_copilot.llm import ResponseRewriter
from sales_copilot.playbook import Playbook


@dataclass(frozen=True)
class CueCard:
    cue_type: str
    title: str
    trigger_text: str
    response: str
    score: float
    audio_end_s: float  # stream time the triggering speech ended
    competitor: str | None = None
    battlecard: str | None = None


@dataclass
class CueEngine:
    classifier: CueClassifier
    playbook: Playbook
    threshold: float = 0.5
    cooldown_s: float = 20.0  # same cue (and same competitor) not re-shown within this window
    short_context_words: int = 4  # very short segments are classified with the previous one
    rewriter: ResponseRewriter = field(default_factory=ResponseRewriter)
    history: list[str] = field(default_factory=list)
    classify_times_s: list[float] = field(default_factory=list)
    raw_labels: list[frozenset[str]] = field(default_factory=list)  # pre-cooldown, per segment
    _last_shown: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def on_segment(self, text: str, audio_end_s: float) -> list[CueCard]:
        text = text.strip()
        if not text:
            return []
        query = text
        if len(text.split()) < self.short_context_words and self.history:
            query = f"{self.history[-1]} {text}"
        self.history.append(text)
        t0 = time.perf_counter()
        scores = self.classifier.predict_scores([query])[0]
        self.classify_times_s.append(time.perf_counter() - t0)
        self.raw_labels.append(
            frozenset(c for c in CUE_TYPES if scores.get(c, 0.0) >= self.threshold)
        )

        cards: list[CueCard] = []
        for cue in CUE_TYPES:
            score = scores.get(cue, 0.0)
            if score < self.threshold:
                continue
            if cue == "competitor_mention":
                comps = self.playbook.find_competitors(query)
                if not comps:
                    cards.extend(self._maybe(cue, text, score, audio_end_s, None, None, key=cue))
                for comp in comps:
                    cards.extend(
                        self._maybe(
                            cue,
                            text,
                            score,
                            audio_end_s,
                            comp.name,
                            comp.battlecard,
                            key=f"{cue}:{comp.name}",
                        )
                    )
            else:
                cards.extend(self._maybe(cue, text, score, audio_end_s, None, None, key=cue))
        return cards

    def _maybe(
        self,
        cue: str,
        text: str,
        score: float,
        audio_end_s: float,
        competitor: str | None,
        battlecard: str | None,
        key: str,
    ) -> list[CueCard]:
        last = self._last_shown.get(key)
        if last is not None and audio_end_s - last < self.cooldown_s:
            return []
        self._last_shown[key] = audio_end_s
        n = self._counts.get(cue, 0)
        self._counts[cue] = n + 1
        response = self.rewriter.rewrite(self.playbook.response_for(cue, n), text)
        return [
            CueCard(
                cue_type=cue,
                title=self.playbook.title_for(cue) + (f": {competitor}" if competitor else ""),
                trigger_text=text,
                response=response,
                score=score,
                audio_end_s=audio_end_s,
                competitor=competitor,
                battlecard=battlecard,
            )
        ]
