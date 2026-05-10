"""Ollama-backed embedding and reranker providers for schema retrieval."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request

from sol01.infra.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_SCHEMA_EMBEDDING_MODEL,
    DEFAULT_SCHEMA_RERANKER_MODEL,
)
from sol01.infra.logging import get_logger
from sol01.schema.embedding import SchemaProviderError, normalize_vector

EMBED_ENDPOINT = "/api/embed"
GENERATE_ENDPOINT = "/api/generate"
logger = get_logger(__name__)


@dataclass(frozen=True)
class OllamaTransportError(RuntimeError):
    """Low-level HTTP error from an Ollama JSON endpoint."""

    endpoint: str
    message: str
    status: int | None = None

    def __str__(self) -> str:
        status_text = f" HTTP {self.status}" if self.status is not None else ""
        return f"{self.endpoint}{status_text}: {self.message}"


class JsonTransport(Protocol):
    """Transport boundary used by tests to avoid live Ollama calls."""

    def post_json(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST one JSON payload and return the decoded JSON object."""


class OllamaHTTPTransport:
    """Small stdlib JSON transport for local Ollama endpoints."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def post_json(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """POST JSON to an Ollama endpoint and decode the JSON response."""

        url = f"{self.base_url}{endpoint}"
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            message = _read_http_error(exc)
            raise OllamaTransportError(endpoint, message, status=exc.code) from exc
        except (TimeoutError, OSError, error.URLError) as exc:
            raise OllamaTransportError(endpoint, str(exc)) from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise OllamaTransportError(endpoint, "response was not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise OllamaTransportError(endpoint, "response JSON was not an object")
        return decoded


class OllamaEmbeddingProvider:
    """Embedding provider that calls Ollama's batch `/api/embed` endpoint."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_SCHEMA_EMBEDDING_MODEL,
        transport: JsonTransport | None = None,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.transport = transport or OllamaHTTPTransport(base_url=base_url)
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches and return unit-normalized vectors."""

        embeddings: list[list[float]] = []
        total_texts = len(texts)
        if total_texts:
            logger.info(
                "ollama embedding start",
                model=self.model,
                text_count=total_texts,
                batch_size=self.batch_size,
            )
        total_batches = math.ceil(total_texts / self.batch_size) if total_texts else 0
        for start in range(0, total_texts, self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            if not batch:
                continue
            batch_number = start // self.batch_size + 1
            logger.info(
                "ollama embedding batch start",
                model=self.model,
                batch_number=batch_number,
                batch_count=total_batches,
                batch_size=len(batch),
                text_count=total_texts,
            )
            response = _post_or_raise(
                self.transport,
                EMBED_ENDPOINT,
                {"model": self.model, "input": batch},
                model=self.model,
                provider_name="Ollama embedding provider",
            )
            raw_embeddings = _extract_embeddings(response, expected_count=len(batch))
            try:
                embeddings.extend(normalize_vector(vector) for vector in raw_embeddings)
            except (TypeError, ValueError) as exc:
                raise SchemaProviderError(
                    f"Ollama embedding endpoint {EMBED_ENDPOINT} with model {self.model} "
                    "returned an unusable embedding vector"
                ) from exc
            logger.info(
                "ollama embedding batch complete",
                model=self.model,
                batch_number=batch_number,
                batch_count=total_batches,
                embedded_count=len(embeddings),
                text_count=total_texts,
            )
        if total_texts:
            logger.info(
                "ollama embedding complete",
                model=self.model,
                text_count=total_texts,
                embedding_count=len(embeddings),
            )
        return embeddings


class OllamaRerankerProvider:
    """Reranker provider that scores query/schema pairs with yes/no logprobs."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_SCHEMA_RERANKER_MODEL,
        transport: JsonTransport | None = None,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.transport = transport or OllamaHTTPTransport(base_url=base_url)

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return yes-probability scores for query and schema text pairs."""

        return [self._score_pair(query, text) for text in texts]

    def _score_pair(self, query: str, text: str) -> float:
        payload = {
            "model": self.model,
            "prompt": _reranker_prompt(query, text),
            "stream": False,
            "logprobs": True,
            "top_logprobs": 5,
            "options": {
                "temperature": 0,
                "num_predict": 1,
            },
        }
        response = _post_or_raise(
            self.transport,
            GENERATE_ENDPOINT,
            payload,
            model=self.model,
            provider_name="Ollama reranker provider",
        )
        return _yes_no_probability(response, endpoint=GENERATE_ENDPOINT, model=self.model)


def _post_or_raise(
    transport: JsonTransport,
    endpoint: str,
    payload: Mapping[str, Any],
    *,
    model: str,
    provider_name: str,
) -> dict[str, Any]:
    """POST to Ollama and convert transport/model errors into actionable messages."""

    try:
        response = transport.post_json(endpoint, payload)
    except OllamaTransportError as exc:
        raise _provider_error(provider_name, endpoint, model, str(exc), status=exc.status) from exc

    error_message = response.get("error")
    if isinstance(error_message, str) and error_message.strip():
        raise _provider_error(provider_name, endpoint, model, error_message)
    return response


def _provider_error(
    provider_name: str,
    endpoint: str,
    model: str,
    detail: str,
    *,
    status: int | None = None,
) -> SchemaProviderError:
    """Build one clear provider error for Ollama connection and model failures."""

    detail_lower = detail.lower()
    if status == 404 or "not found" in detail_lower or "pull model" in detail_lower:
        guidance = f"Required Ollama model {model!r} is missing; pull it before running retrieval."
    else:
        guidance = "Ensure the local Ollama server is running and reachable."
    return SchemaProviderError(
        f"{provider_name} failed at {endpoint} for model {model!r}: {detail}. {guidance}"
    )


def _extract_embeddings(
    response: Mapping[str, Any], *, expected_count: int
) -> list[Sequence[float]]:
    """Extract batch embeddings from Ollama `/api/embed` response JSON."""

    embeddings = response.get("embeddings")
    if isinstance(embeddings, list) and len(embeddings) == expected_count:
        return embeddings

    embedding = response.get("embedding")
    if expected_count == 1 and isinstance(embedding, list):
        return [embedding]

    raise SchemaProviderError(
        f"Ollama embedding endpoint {EMBED_ENDPOINT} returned {type(embeddings).__name__} "
        f"for embeddings; expected {expected_count} vectors"
    )


def _yes_no_probability(response: Mapping[str, Any], *, endpoint: str, model: str) -> float:
    """Compute P(yes) from the first token's yes/no top-logprobs."""

    token_logprobs = _extract_top_logprobs(response)
    yes_logprob = token_logprobs.get("yes")
    no_logprob = token_logprobs.get("no")
    if yes_logprob is None or no_logprob is None:
        raise SchemaProviderError(
            f"Ollama reranker endpoint {endpoint} with model {model!r} did not return usable "
            "yes/no top_logprobs. Use an Ollama build and model tag that expose logprobs."
        )

    baseline = max(yes_logprob, no_logprob)
    yes_weight = math.exp(yes_logprob - baseline)
    no_weight = math.exp(no_logprob - baseline)
    return yes_weight / (yes_weight + no_weight)


def _extract_top_logprobs(response: Mapping[str, Any]) -> dict[str, float]:
    """Extract normalized yes/no logprobs from common Ollama/OpenAI-like shapes."""

    for entry in _candidate_logprob_entries(response):
        top_logprobs = _coerce_top_logprobs(entry.get("top_logprobs"))
        if top_logprobs:
            return top_logprobs
    return {}


def _candidate_logprob_entries(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return possible per-token logprob entries from known response layouts."""

    direct = response.get("logprobs")
    entries = _entries_from_logprobs(direct)
    if entries:
        return entries

    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            entries = _entries_from_logprobs(choice.get("logprobs"))
            if entries:
                return entries
    return []


def _entries_from_logprobs(value: object) -> list[Mapping[str, Any]]:
    """Normalize several logprob container shapes into token entries."""

    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, Mapping)]
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, list):
            return [entry for entry in content if isinstance(entry, Mapping)]
        tokens = value.get("tokens")
        token_logprobs = value.get("token_logprobs")
        top_logprobs = value.get("top_logprobs")
        if isinstance(tokens, list) and isinstance(token_logprobs, list):
            entries: list[Mapping[str, Any]] = []
            for index, token in enumerate(tokens):
                entry: dict[str, Any] = {
                    "token": token,
                    "logprob": token_logprobs[index] if index < len(token_logprobs) else None,
                }
                if isinstance(top_logprobs, list) and index < len(top_logprobs):
                    entry["top_logprobs"] = top_logprobs[index]
                entries.append(entry)
            return entries
    return []


def _coerce_top_logprobs(value: object) -> dict[str, float]:
    """Normalize top-logprob candidates into yes/no logprob values."""

    if isinstance(value, Mapping):
        scores: dict[str, float] = {}
        for token, logprob in value.items():
            normalized = _normalize_yes_no_token(str(token))
            coerced = _coerce_float(logprob)
            if normalized in {"yes", "no"} and coerced is not None:
                scores[normalized] = coerced
        return scores

    if isinstance(value, list):
        scores: dict[str, float] = {}
        for item in value:
            if not isinstance(item, Mapping):
                continue
            token = item.get("token")
            logprob = _coerce_float(item.get("logprob"))
            normalized = _normalize_yes_no_token(str(token)) if token is not None else ""
            if normalized in {"yes", "no"} and logprob is not None:
                scores[normalized] = logprob
        return scores

    return {}


def _coerce_float(value: object) -> float | None:
    """Return a float for numeric logprob values."""

    if not isinstance(value, int | float):
        return None
    return float(value)


def _normalize_yes_no_token(token: str) -> str:
    """Normalize a model token to the yes/no labels used for reranking."""

    return token.strip().lower().lstrip("▁Ġ").strip(".,:;!?'\"")


def _reranker_prompt(query: str, text: str) -> str:
    """Build the minimal binary prompt used by the local reranker model."""

    return (
        "Answer with exactly one token: yes or no.\n"
        "Is the schema text relevant to the user question?\n\n"
        f"Question:\n{query}\n\n"
        f"Schema text:\n{text}\n\n"
        "Relevant?"
    )


def _read_http_error(exc: error.HTTPError) -> str:
    """Read an HTTP error body without hiding the original status."""

    try:
        body = exc.read().decode("utf-8")
    except OSError:
        return str(exc.reason)
    if not body:
        return str(exc.reason)
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(decoded, dict) and isinstance(decoded.get("error"), str):
        return decoded["error"]
    return body
