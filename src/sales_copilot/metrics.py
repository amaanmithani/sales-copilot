"""Evaluation metrics: per-label P/R/F1 with bootstrap CIs, and word error rate."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int_]


def prf(y_true: IntArray, y_pred: IntArray) -> tuple[float, float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def macro_f1(y_true: IntArray, y_pred: IntArray) -> float:
    return float(np.mean([prf(y_true[:, j], y_pred[:, j])[2] for j in range(y_true.shape[1])]))


def micro_prf(y_true: IntArray, y_pred: IntArray) -> tuple[float, float, float]:
    return prf(y_true.ravel(), y_pred.ravel())


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict[str, float]:
        return {"point": round(self.point, 4), "lo": round(self.lo, 4), "hi": round(self.hi, 4)}


def _ci(point: float, samples: Sequence[float] | NDArray[np.float64], alpha: float) -> Interval:
    arr = np.asarray(samples, dtype=np.float64)
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(lo), float(hi))


def bootstrap_report(
    y_true: IntArray,
    y_pred: IntArray,
    labels: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Percentile bootstrap over utterances (rows) for per-label P/R/F1, macro-F1, micro-F1."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    per: dict[str, dict[str, object]] = {}
    for j, lab in enumerate(labels):
        p, r, f = prf(y_true[:, j], y_pred[:, j])
        bs = np.array([prf(y_true[ix, j], y_pred[ix, j]) for ix in idx])
        per[lab] = {
            "support": int(y_true[:, j].sum()),
            "precision": _ci(p, bs[:, 0], alpha).as_dict(),
            "recall": _ci(r, bs[:, 1], alpha).as_dict(),
            "f1": _ci(f, bs[:, 2], alpha).as_dict(),
        }
    mac = [macro_f1(y_true[ix], y_pred[ix]) for ix in idx]
    mic = [micro_prf(y_true[ix], y_pred[ix])[2] for ix in idx]
    return {
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
        "per_label": per,
        "macro_f1": _ci(macro_f1(y_true, y_pred), mac, alpha).as_dict(),
        "micro_f1": _ci(micro_prf(y_true, y_pred)[2], mic, alpha).as_dict(),
    }


def paired_bootstrap_diff(
    y_true: IntArray,
    pred_a: IntArray,
    pred_b: IntArray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """CI for macro-F1(a) - macro-F1(b) using the same resampled rows for both systems."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    diffs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        diffs.append(macro_f1(y_true[ix], pred_a[ix]) - macro_f1(y_true[ix], pred_b[ix]))
    point = macro_f1(y_true, pred_a) - macro_f1(y_true, pred_b)
    out = _ci(point, diffs, alpha).as_dict()
    out["p_b_better_or_equal"] = round(float(np.mean(np.asarray(diffs) <= 0)), 4)
    return out


# ---------------------------------------------------------------- word error rate

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    """Spell out 0 <= n < 1_000_000 (enough for LibriSpeech-style references)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    rest = n % 1000
    head = number_to_words(n // 1000) + " thousand"
    return head + ("" if rest == 0 else " " + number_to_words(rest))


_CONTRACTIONS = {"mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint"}


def normalize_text(text: str) -> str:
    """Lower-case, spell out integers, drop punctuation (keeps apostrophes inside words)."""
    t = text.lower().replace("-", " ")
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(
        r"\d+",
        lambda m: (
            f" {number_to_words(int(m.group()))} " if int(m.group()) < 1_000_000 else m.group()
        ),
        t,
    )
    t = re.sub(r"[^a-z' ]+", " ", t)
    words = [w.strip("'") for w in t.split()]
    return " ".join(_CONTRACTIONS.get(w, w) for w in words if w)


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def corpus_wer(refs: Sequence[str], hyps: Sequence[str]) -> tuple[float, int, int]:
    """Corpus WER = total edits / total reference words, after normalisation."""
    errs = words = 0
    for r, h in zip(refs, hyps, strict=True):
        rw, hw = normalize_text(r).split(), normalize_text(h).split()
        errs += edit_distance(rw, hw)
        words += len(rw)
    return (errs / words if words else 0.0), errs, words


def bootstrap_wer(
    refs: Sequence[str], hyps: Sequence[str], n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    per = [
        (
            edit_distance(normalize_text(r).split(), normalize_text(h).split()),
            len(normalize_text(r).split()),
        )
        for r, h in zip(refs, hyps, strict=True)
    ]
    e = np.array([p[0] for p in per], dtype=np.float64)
    w = np.array([p[1] for p in per], dtype=np.float64)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(per), size=(n_boot, len(per)))
    samples = e[ix].sum(axis=1) / np.maximum(w[ix].sum(axis=1), 1)
    return _ci(float(e.sum() / max(w.sum(), 1)), samples.tolist(), 0.05).as_dict()
