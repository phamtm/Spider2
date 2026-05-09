"""Provider contracts and test doubles for schema retrieval scoring."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Protocol


class SchemaProviderError(RuntimeError):
    """Raised when a schema retrieval provider cannot produce usable scores."""


class EmbeddingProvider(Protocol):
    """Embeds retrieval text into dense vectors."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one normalized embedding vector per input text."""


class RerankerProvider(Protocol):
    """Scores query and schema text pairs for semantic relevance."""

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return one relevance score per input text."""


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Return a unit-length copy of one embedding vector."""

    values = [float(value) for value in vector]
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        raise SchemaProviderError("embedding provider returned a zero-length vector")
    return [value / magnitude for value in values]


class FakeEmbeddingProvider:
    """Deterministic embedding provider for tests that should not call Ollama."""

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]] | None = None,
        *,
        dimensions: int = 8,
    ) -> None:
        self._vectors = dict(vectors or {})
        self._dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return configured vectors or stable hash-derived vectors."""

        return [
            normalize_vector(self._vectors.get(text, _hash_vector(text, self._dimensions)))
            for text in texts
        ]


class FakeRerankerProvider:
    """Deterministic reranker provider for tests that should not call Ollama."""

    def __init__(
        self,
        scores: Mapping[tuple[str, str], float] | None = None,
        *,
        default_score: float = 0.5,
    ) -> None:
        self._scores = dict(scores or {})
        self._default_score = default_score

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return configured scores keyed by query and text."""

        return [self._scores.get((query, text), self._default_score) for text in texts]


def _hash_vector(text: str, dimensions: int) -> list[float]:
    """Build a small stable non-zero vector from text bytes."""

    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [float(digest[index % len(digest)] + 1) for index in range(dimensions)]
    return values
