from __future__ import annotations

import numpy as np

from sales_copilot.metrics import (
    bootstrap_report,
    bootstrap_wer,
    corpus_wer,
    edit_distance,
    macro_f1,
    normalize_text,
    number_to_words,
    paired_bootstrap_diff,
    prf,
)


def test_prf_basic() -> None:
    p, r, f = prf(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (p, r, f) == (0.5, 0.5, 0.5)
    assert prf(np.array([0, 0]), np.array([0, 0])) == (0.0, 0.0, 0.0)


def test_bootstrap_report_perfect_and_ci_bounds() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=(60, 3))
    rep = bootstrap_report(y, y, ["a", "b", "c"], n_boot=200)
    assert rep["macro_f1"] == {"point": 1.0, "lo": 1.0, "hi": 1.0}
    noisy = y.copy()
    noisy[:10] = 1 - noisy[:10]
    rep2 = bootstrap_report(y, noisy, ["a", "b", "c"], n_boot=200)
    m = rep2["macro_f1"]
    assert isinstance(m, dict) and m["lo"] <= m["point"] <= m["hi"] < 1.0
    per = rep2["per_label"]
    assert isinstance(per, dict) and per["a"]["support"] == int(y[:, 0].sum())


def test_paired_diff_sign() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=(80, 2))
    worse = y.copy()
    worse[:30] = 1 - worse[:30]
    d = paired_bootstrap_diff(y, y, worse, n_boot=300)
    assert d["point"] > 0 and d["lo"] > 0 and d["p_b_better_or_equal"] == 0.0
    assert macro_f1(y, y) == 1.0


def test_numbers_and_normalizer() -> None:
    assert number_to_words(0) == "zero"
    assert number_to_words(42) == "forty two"
    assert number_to_words(1905) == "one thousand nine hundred five"
    assert number_to_words(300) == "three hundred"
    assert number_to_words(20_000) == "twenty thousand"
    assert normalize_text("Mr. Smith paid $1,200!") == "mister smith paid one thousand two hundred"
    assert normalize_text("don't -- STOP") == "don't stop"


def test_wer() -> None:
    assert edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    wer, errs, words = corpus_wer(["the cat sat", "hello"], ["the cat sat", "hello there"])
    assert (errs, words) == (1, 4) and wer == 0.25
    ci = bootstrap_wer(["the cat sat", "hello"], ["the bat sat", "hello"], n_boot=100)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert corpus_wer([], []) == (0.0, 0, 0)
