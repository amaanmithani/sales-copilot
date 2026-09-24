"""Cue label set and labelled-utterance loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

CUE_TYPES: tuple[str, ...] = (
    "objection_price",
    "objection_timing",
    "objection_authority",
    "objection_status_quo",
    "competitor_mention",
    "pricing_question",
    "next_step",
)
NONE_LABEL = "none"


@dataclass(frozen=True)
class Example:
    text: str
    labels: frozenset[str]


def parse_labels(field: str) -> frozenset[str]:
    parts = {p.strip() for p in field.split(";") if p.strip()}
    parts.discard(NONE_LABEL)
    unknown = parts - set(CUE_TYPES)
    if unknown:
        raise ValueError(f"unknown label(s): {sorted(unknown)}")
    return frozenset(parts)


def load_tsv(path: str | Path) -> list[Example]:
    """Load a two-column TSV (labels, text). Labels are ';'-separated; 'none' = no cue."""
    out: list[Example] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            out.append(Example(text=text, labels=parse_labels(row.get("labels") or "")))
    return out


def to_matrix(examples: list[Example]) -> list[list[int]]:
    """Binary indicator matrix (n_examples x n_cue_types)."""
    return [[int(c in ex.labels) for c in CUE_TYPES] for ex in examples]
