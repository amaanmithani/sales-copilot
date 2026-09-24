"""Learned cue classifier: sentence embeddings (or TF-IDF) + one-vs-rest logistic regression.

Hyper-parameters (C) are chosen by cross-validation on the *training* file only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from sales_copilot.cues.base import Scores
from sales_copilot.labels import CUE_TYPES, Example, to_matrix

FloatArray = NDArray[np.float64]


class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None: ...

    def encode(self, texts: Sequence[str]) -> FloatArray: ...


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 (22M params, 384-d) on CPU. Imported lazily; never used in tests."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name.split("/")[-1]
        self._model: Any = SentenceTransformer(model_name, device="cpu")

    def fit(self, texts: Sequence[str]) -> None:  # pre-trained, nothing to fit
        return None

    def encode(self, texts: Sequence[str]) -> FloatArray:
        vecs = self._model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float64)


class TfidfEmbedder:
    """Word + char n-gram TF-IDF; a no-download learned baseline."""

    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self._vec: Any = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> None:
        self._vec.fit([t.lower() for t in texts])
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> FloatArray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.encode called before fit")
        return np.asarray(self._vec.transform([t.lower() for t in texts]).toarray())


class LearnedClassifier:
    """One logistic regression per cue type over a shared text embedding."""

    def __init__(
        self,
        embedder: Embedder,
        c_grid: Sequence[float] = (0.25, 1.0, 4.0, 16.0),
        threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.embedder = embedder
        self.name = f"lr+{embedder.name}"
        self.c_grid = tuple(c_grid)
        self.threshold = threshold
        self.seed = seed
        self.chosen_c: dict[str, float] = {}
        self._models: dict[str, Any] = {}

    def fit(self, examples: list[Example]) -> LearnedClassifier:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        texts = [e.text for e in examples]
        self.embedder.fit(texts)
        x = self.embedder.encode(texts)
        y_all = np.asarray(to_matrix(examples))
        for j, cue in enumerate(CUE_TYPES):
            y = y_all[:, j]
            best_c, best = self.c_grid[0], -1.0
            n_pos = int(y.sum())
            if len(self.c_grid) > 1 and n_pos >= 3:
                cv = StratifiedKFold(n_splits=min(5, n_pos), shuffle=True, random_state=self.seed)
                for c in self.c_grid:
                    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
                    score = float(cross_val_score(clf, x, y, cv=cv, scoring="f1").mean())
                    if score > best:
                        best_c, best = c, score
            model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000)
            model.fit(x, y)
            self._models[cue] = model
            self.chosen_c[cue] = best_c
        return self

    def predict_scores(self, texts: list[str]) -> list[Scores]:
        if not self._models:
            raise RuntimeError("LearnedClassifier.predict_scores called before fit")
        if not texts:
            return []
        x = self.embedder.encode(texts)
        cols = {cue: self._models[cue].predict_proba(x)[:, 1] for cue in CUE_TYPES}
        return [{cue: float(cols[cue][i]) for cue in CUE_TYPES} for i in range(len(texts))]
