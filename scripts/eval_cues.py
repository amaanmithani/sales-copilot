"""Evaluate cue classifiers on the held-out test set -> results/cue_eval.json.

Systems: rules (regex baseline), lr+tfidf, lr+all-MiniLM-L6-v2. All learned models are
trained on data/cues/train.tsv only; the test file is read once, for scoring.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from sales_copilot.cues.base import CueClassifier, scores_to_labels
from sales_copilot.cues.learned import LearnedClassifier, SentenceTransformerEmbedder, TfidfEmbedder
from sales_copilot.cues.rules import RuleClassifier
from sales_copilot.labels import CUE_TYPES, Example, load_tsv, to_matrix
from sales_copilot.metrics import bootstrap_report, paired_bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]


def predict(clf: CueClassifier, examples: list[Example]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    scores = clf.predict_scores([e.text for e in examples])
    per_utt_ms = (time.perf_counter() - t0) / max(1, len(examples)) * 1000
    pred = np.asarray([[int(c in scores_to_labels(s)) for c in CUE_TYPES] for s in scores])
    return pred, per_utt_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-minilm", action="store_true", help="skip the embedding model")
    ap.add_argument("--out", default=str(ROOT / "results/cue_eval.json"))
    args = ap.parse_args()

    train = load_tsv(ROOT / "data/cues/train.tsv")
    test = load_tsv(ROOT / "data/cues/test.tsv")
    train_norm = {e.text.lower().strip(" .?!") for e in train}
    overlap = sum(e.text.lower().strip(" .?!") in train_norm for e in test)
    y = np.asarray(to_matrix(test))

    systems: list[tuple[str, CueClassifier]] = [("rules", RuleClassifier())]
    systems.append(("lr+tfidf", LearnedClassifier(TfidfEmbedder()).fit(train)))
    if not args.no_minilm:
        systems.append(
            ("lr+all-MiniLM-L6-v2", LearnedClassifier(SentenceTransformerEmbedder()).fit(train))
        )

    preds: dict[str, np.ndarray] = {}
    results: dict[str, object] = {}
    for name, clf in systems:
        pred, ms = predict(clf, test)
        preds[name] = pred
        rep = bootstrap_report(y, pred, CUE_TYPES, n_boot=2000, seed=0)
        rep["ms_per_utterance_batch"] = round(ms, 3)
        if isinstance(clf, LearnedClassifier):
            rep["chosen_C"] = clf.chosen_c
        results[name] = rep
        print(f"{name:24s} macro-F1 {rep['macro_f1']}")

    comparisons = {}
    for name in preds:
        if name != "rules":
            comparisons[f"{name} minus rules"] = paired_bootstrap_diff(
                y, preds[name], preds["rules"], n_boot=2000, seed=0
            )
    out = {
        "train_file": "data/cues/train.tsv",
        "test_file": "data/cues/test.tsv",
        "n_train": len(train),
        "n_test": len(test),
        "exact_text_overlap_train_test": overlap,
        "labels": list(CUE_TYPES),
        "threshold": 0.5,
        "authored_by": "project author (single author for train and test; see README limits)",
        "bootstrap": "percentile, 2000 resamples of test utterances, seed 0, 95% CI",
        "machine": f"{platform.system()} {platform.machine()}",
        "systems": results,
        "paired_macro_f1_differences": comparisons,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
