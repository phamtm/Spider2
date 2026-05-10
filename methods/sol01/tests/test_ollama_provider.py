"""Tests for local Ollama schema retrieval providers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from sol01.infra.config import DEFAULT_SCHEMA_EMBEDDING_MODEL, DEFAULT_SCHEMA_RERANKER_MODEL
from sol01.schema.embedding import FakeEmbeddingProvider, FakeRerankerProvider, SchemaProviderError
from sol01.schema.ollama_provider import (
    EMBED_ENDPOINT,
    GENERATE_ENDPOINT,
    OllamaEmbeddingProvider,
    OllamaRerankerProvider,
    OllamaTransportError,
)


class FakeTransport:
    """Captures provider requests and returns queued responses."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def post_json(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((endpoint, payload))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_fake_providers_return_deterministic_scores_without_ollama():
    embedding_provider = FakeEmbeddingProvider({"orders": [3.0, 4.0]})
    reranker_provider = FakeRerankerProvider({("status?", "STATUS column"): 0.91})

    assert embedding_provider.embed_texts(["orders"]) == [[0.6, 0.8]]
    assert len(embedding_provider.embed_texts(["unknown text"])[0]) == 8
    assert reranker_provider.score_pairs("status?", ["STATUS column", "OTHER"]) == [0.91, 0.5]


def test_ollama_embedding_provider_batches_embed_requests_and_normalizes_vectors():
    transport = FakeTransport(
        [
            {"embeddings": [[3.0, 4.0], [0.0, 2.0]]},
            {"embeddings": [[10.0, 0.0]]},
        ]
    )
    provider = OllamaEmbeddingProvider(transport=transport, batch_size=2)

    embeddings = provider.embed_texts(["one", "two", "three"])

    assert embeddings == [[0.6, 0.8], [0.0, 1.0], [1.0, 0.0]]
    assert [endpoint for endpoint, _payload in transport.calls] == [EMBED_ENDPOINT, EMBED_ENDPOINT]
    assert transport.calls[0][1] == {
        "model": DEFAULT_SCHEMA_EMBEDDING_MODEL,
        "input": ["one", "two"],
    }
    assert transport.calls[1][1]["input"] == ["three"]


def test_ollama_embedding_provider_names_missing_model_errors():
    transport = FakeTransport([{"error": "model qwen3-embedding:4b not found"}])
    provider = OllamaEmbeddingProvider(transport=transport)

    with pytest.raises(SchemaProviderError, match="qwen3-embedding:4b.*missing"):
        provider.embed_texts(["schema text"])


def test_ollama_embedding_provider_names_unreachable_endpoint():
    transport = FakeTransport([OllamaTransportError(EMBED_ENDPOINT, "connection refused")])
    provider = OllamaEmbeddingProvider(transport=transport)

    with pytest.raises(SchemaProviderError, match="/api/embed.*qwen3-embedding:4b.*server"):
        provider.embed_texts(["schema text"])


def test_ollama_reranker_provider_uses_generate_logprobs_for_yes_no_score():
    transport = FakeTransport(
        [
            {
                "response": "yes",
                "logprobs": [
                    {
                        "token": " yes",
                        "logprob": -0.2,
                        "top_logprobs": [
                            {"token": " yes", "logprob": -0.2},
                            {"token": " no", "logprob": -2.2},
                        ],
                    }
                ],
            }
        ]
    )
    provider = OllamaRerankerProvider(transport=transport)

    scores = provider.score_pairs("find order status", ["ORDERS.STATUS text"])

    assert scores == [pytest.approx(1 / (1 + math.exp(-2.0)))]
    endpoint, payload = transport.calls[0]
    assert endpoint == GENERATE_ENDPOINT
    assert payload["model"] == DEFAULT_SCHEMA_RERANKER_MODEL
    assert payload["stream"] is False
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 5
    assert "find order status" in str(payload["prompt"])
    assert "ORDERS.STATUS text" in str(payload["prompt"])


def test_ollama_reranker_provider_rejects_missing_logprobs():
    transport = FakeTransport([{"response": "yes"}])
    provider = OllamaRerankerProvider(transport=transport)

    with pytest.raises(SchemaProviderError, match="/api/generate.*qwen3-reranker:4b.*logprobs"):
        provider.score_pairs("question", ["schema"])


def test_ollama_reranker_provider_rejects_unusable_yes_no_logprobs():
    transport = FakeTransport(
        [
            {
                "response": "maybe",
                "logprobs": [
                    {
                        "token": " maybe",
                        "logprob": -0.1,
                        "top_logprobs": [{"token": " maybe", "logprob": -0.1}],
                    }
                ],
            }
        ]
    )
    provider = OllamaRerankerProvider(transport=transport)

    with pytest.raises(SchemaProviderError, match="yes/no top_logprobs"):
        provider.score_pairs("question", ["schema"])
