"""Ollama embedding and reranker providers for local dense retrieval."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Sequence

import numpy as np

from sol01.schema.embedding import normalize_vector


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns an unexpected or unusable response."""


class OllamaEmbeddingProvider:
    """Calls POST /api/embed on a local Ollama server for dense text embeddings."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.embed(["warmup"]).shape[1]
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized (len(texts), dim) float32 embeddings."""
        payload = json.dumps({"model": self._model, "input": list(texts)}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"Ollama is not reachable at {self._base_url}: {exc}\n"
                "Start Ollama with: ollama serve"
            ) from exc
        raw = body.get("embeddings")
        if not isinstance(raw, list) or not raw:
            raise OllamaResponseError(f"Ollama /api/embed returned unexpected response: {body!r}")
        vectors = np.array(raw, dtype=np.float32)
        for i in range(len(vectors)):
            vectors[i] = normalize_vector(vectors[i])
        return vectors


class OllamaRerankerProvider:
    """Scores (query, passage) pairs using yes/no logprobs from a Qwen3-Reranker Ollama model."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return a relevance score in [0, 1] for each passage."""
        return [self._score_one(query, p) for p in passages]

    def _score_one(self, query: str, passage: str) -> float:
        prompt = (
            "Given a search query and a passage, classify whether the passage is relevant "
            "to the query. Answer with 'Yes' or 'No'.\n\n"
            f"Query: {query}\n\nPassage: {passage}\n\nRelevant:"
        )
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 1},
                "logprobs": True,
                "top_logprobs": 5,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"Ollama is not reachable at {self._base_url}: {exc}"
            ) from exc
        logprobs_data = body.get("logprobs")
        if not logprobs_data:
            raise OllamaResponseError(
                f"Ollama reranker model {self._model!r} did not return logprobs. "
                "The reranker requires a model that supports logprob output."
            )
        top_logprobs = logprobs_data.get("top_logprobs") or []
        if not top_logprobs:
            raise OllamaResponseError(
                f"Ollama reranker model {self._model!r} returned empty top_logprobs."
            )
        first = top_logprobs[0] if isinstance(top_logprobs[0], dict) else {}
        yes_lp = _token_logprob(first, ("Yes", "yes", "YES"))
        no_lp = _token_logprob(first, ("No", "no", "NO"))
        if yes_lp is None and no_lp is None:
            raise OllamaResponseError(
                f"Reranker {self._model!r} did not produce usable yes/no logprobs. "
                f"Available tokens: {list(first.keys())[:10]}"
            )
        yes_p = math.exp(yes_lp) if yes_lp is not None else 0.0
        no_p = math.exp(no_lp) if no_lp is not None else 0.0
        total = yes_p + no_p
        return yes_p / total if total > 1e-12 else 0.5


def _token_logprob(logprob_map: dict[str, float], tokens: tuple[str, ...]) -> float | None:
    for token in tokens:
        if token in logprob_map:
            return logprob_map[token]
    return None
