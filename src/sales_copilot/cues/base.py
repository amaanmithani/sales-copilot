"""Common classifier interface."""

from __future__ import annotations

from typing import Protocol

from sales_copilot.labels import CUE_TYPES

Scores = dict[str, float]


class CueClassifier(Protocol):
    name: str

    def predict_scores(self, texts: list[str]) -> list[Scores]:
        """Per-text score in [0, 1] for every cue type."""
        ...


def scores_to_labels(scores: Scores, threshold: float = 0.5) -> frozenset[str]:
    return frozenset(c for c in CUE_TYPES if scores.get(c, 0.0) >= threshold)
