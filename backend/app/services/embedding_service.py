import hashlib
from typing import Protocol

from openai import OpenAI

from ..core.config import get_settings


class EmbeddingError(RuntimeError):
    """Raised when text embedding cannot be completed."""


class EmbeddingProvider(Protocol):
    model: str

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a non-empty batch of cleaned texts."""


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider with lazy client initialization."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.embedding_model
        self._api_key = api_key if api_key is not None else settings.openai_api_key
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if not self._api_key:
            raise EmbeddingError("OPENAI_API_KEY is required to generate embeddings.")
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._get_client().embeddings.create(
                model=self.model,
                input=texts,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding request failed: {exc}") from exc

        embeddings = [item.embedding for item in response.data]
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        return embeddings


class FakeEmbeddingProvider:
    """Deterministic local embedding provider for unit tests."""

    def __init__(self, dimensions: int = 8, model: str = "fake-embedding-model"):
        self.dimensions = max(1, int(dimensions or 8))
        self.model = model
        self.batches: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for index in range(self.dimensions):
            byte = digest[index % len(digest)]
            values.append(round((byte / 255.0) * 2.0 - 1.0, 6))
        return values


class EmbeddingService:
    """Batching and validation layer over an embedding provider."""

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        batch_size: int | None = None,
    ):
        settings = get_settings()
        self.provider = provider or _provider_from_settings(settings.embedding_provider)
        self.batch_size = _positive_int(batch_size or settings.embedding_batch_size, default=64)
        self.model = self.provider.model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise EmbeddingError("texts must be a list of strings")

        cleaned_texts = []
        for index, text in enumerate(texts):
            cleaned = _clean_text(text)
            if not cleaned:
                raise EmbeddingError(f"Cannot embed empty text at index {index}")
            cleaned_texts.append(cleaned)

        embeddings: list[list[float]] = []
        for start in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[start:start + self.batch_size]
            embeddings.extend(self.provider.embed_batch(batch))

        if len(embeddings) != len(cleaned_texts):
            raise EmbeddingError(
                f"Embedding count mismatch: expected {len(cleaned_texts)}, got {len(embeddings)}"
            )
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        cleaned = _clean_text(query)
        if not cleaned:
            raise EmbeddingError("Cannot embed an empty query")
        return self.embed_texts([cleaned])[0]


def _provider_from_settings(provider_name: str) -> EmbeddingProvider:
    provider = (provider_name or "openai").strip().lower()
    if provider != "openai":
        raise EmbeddingError(f"Unsupported embedding provider: {provider_name}")
    return OpenAIEmbeddingProvider()


def _positive_int(value: int | str | None, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split())
