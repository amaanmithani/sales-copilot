"""Build a cue classifier by name."""

from __future__ import annotations

from pathlib import Path

from sales_copilot.cues.base import CueClassifier
from sales_copilot.cues.learned import LearnedClassifier, SentenceTransformerEmbedder, TfidfEmbedder
from sales_copilot.cues.rules import RuleClassifier
from sales_copilot.labels import load_tsv
from sales_copilot.playbook import Playbook

CLASSIFIERS = ("rules", "tfidf", "minilm")
DEFAULT_TRAIN = Path(__file__).resolve().parents[3] / "data" / "cues" / "train.tsv"


def build_classifier(
    kind: str, playbook: Playbook | None = None, train_path: str | Path | None = None
) -> CueClassifier:
    if kind == "rules":
        return RuleClassifier(playbook)
    if kind not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {kind!r}; choose from {CLASSIFIERS}")
    examples = load_tsv(train_path or DEFAULT_TRAIN)
    embedder = TfidfEmbedder() if kind == "tfidf" else SentenceTransformerEmbedder()
    return LearnedClassifier(embedder).fit(examples)
