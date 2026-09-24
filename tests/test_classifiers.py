from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from sales_copilot.cues.base import scores_to_labels
from sales_copilot.cues.factory import build_classifier
from sales_copilot.cues.learned import LearnedClassifier, TfidfEmbedder
from sales_copilot.cues.rules import RuleClassifier
from sales_copilot.labels import CUE_TYPES, load_tsv

from .conftest import ROOT


class HashEmbedder:
    """Stub embedder: hashed bag of words, no model download."""

    name = "hash"

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, sum(map(ord, w.strip(".,?!"))) % 256] += 1.0
        return out


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("That's way too expensive for us.", {"objection_price"}),
        ("How much does it cost per seat?", {"pricing_question"}),
        ("We're also talking to Gong.", {"competitor_mention"}),
        ("Let me run this by my manager.", {"objection_authority"}),
        ("Can you send me the contract?", {"next_step"}),
        ("How was your weekend?", set()),
    ],
)
def test_rules(text: str, expected: set[str]) -> None:
    assert RuleClassifier().labels_for(text) == expected


def test_rules_scores_shape() -> None:
    scores = RuleClassifier().predict_scores(["hi", "too expensive"])
    assert all(set(s) == set(CUE_TYPES) for s in scores)
    assert scores_to_labels(scores[1]) == {"objection_price"}


def test_learned_with_stub_embedder_fits_train() -> None:
    train = load_tsv(ROOT / "data/cues/train.tsv")
    clf = LearnedClassifier(HashEmbedder(), c_grid=(1.0, 4.0)).fit(train)
    assert set(clf.chosen_c) == set(CUE_TYPES)
    scores = clf.predict_scores([e.text for e in train[:40]])
    hits = sum(
        bool(scores_to_labels(s) & e.labels) for s, e in zip(scores, train[:40], strict=True)
    )
    assert hits >= 30  # memorises its own training data
    assert clf.predict_scores([]) == []


def test_learned_errors_before_fit() -> None:
    with pytest.raises(RuntimeError):
        LearnedClassifier(HashEmbedder()).predict_scores(["x"])
    with pytest.raises(RuntimeError):
        TfidfEmbedder().encode(["x"])


def test_factory_tfidf_and_errors() -> None:
    clf = build_classifier("tfidf")
    s = clf.predict_scores(["how much does this cost per user"])[0]
    assert s["pricing_question"] > 0.5
    assert build_classifier("rules").name == "rules"
    with pytest.raises(ValueError):
        build_classifier("gpt")
