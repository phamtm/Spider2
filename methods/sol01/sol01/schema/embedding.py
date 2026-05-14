"""BM25 sparse index and dense embedding provider protocol."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_BM25_K1 = 1.2
_BM25_B = 0.75


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for text embedding providers."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized (len(texts), dim) float32 embeddings."""
        ...


class FakeEmbeddingProvider:
    """Deterministic fake embeddings for testing without Ollama."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(abs(hash(text)) % (2**31))
            v = rng.normal(size=self._dim).astype(np.float32)
            vectors[i] = normalize_vector(v)
        return vectors


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D float32 vector; returns zeros if norm is near zero."""
    norm = float(np.linalg.norm(v))
    return (v / norm).astype(np.float32) if norm > 1e-12 else np.zeros_like(v, dtype=np.float32)


def cosine_scores(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarities between a pre-normalized query vector and corpus matrix."""
    return (corpus_vecs @ query_vec).astype(np.float32)


class BM25Index:
    """In-memory BM25 index over a list of text documents."""

    def __init__(self, documents: Sequence[str]) -> None:
        self._n = len(documents)
        tokenized = [_tokenize(doc) for doc in documents]
        self._avg_dl = sum(len(t) for t in tokenized) / max(1, self._n)
        self._doc_lengths = [len(t) for t in tokenized]

        tf_by_term: dict[str, Counter[int]] = defaultdict(Counter)
        for doc_id, tokens in enumerate(tokenized):
            for token in tokens:
                tf_by_term[token][doc_id] += 1

        self._tf_by_term: dict[str, dict[int, int]] = {
            term: dict(counter) for term, counter in tf_by_term.items()
        }
        self._idf: dict[str, float] = {
            term: math.log((self._n - len(docs) + 0.5) / (len(docs) + 0.5) + 1)
            for term, docs in self._tf_by_term.items()
        }

    def scores(self, query: str, *, top_k: int | None = None) -> list[tuple[int, float]]:
        """Return (doc_id, bm25_score) pairs sorted by descending score."""
        score_map: dict[int, float] = defaultdict(float)
        for token in _tokenize(query):
            idf = self._idf.get(token)
            if idf is None:
                continue
            for doc_id, tf in self._tf_by_term[token].items():
                dl = self._doc_lengths[doc_id]
                tf_norm = (
                    tf
                    * (_BM25_K1 + 1)
                    / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(self._avg_dl, 1)))
                )
                score_map[doc_id] += idf * tf_norm
        ranked = sorted(score_map.items(), key=lambda x: -x[1])
        return ranked[:top_k] if top_k is not None else ranked

    def exact_match_boosts(
        self, identifiers: Sequence[str], *, boost: float = 5.0
    ) -> dict[int, float]:
        """Extra score for docs containing exact lowercase tokens from identifiers."""
        boosts: dict[int, float] = defaultdict(float)
        for term in identifiers:
            for token in _tokenize(term):
                for doc_id, tf in self._tf_by_term.get(token, {}).items():
                    boosts[doc_id] += boost * min(tf, 3)
        return dict(boosts)


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]
