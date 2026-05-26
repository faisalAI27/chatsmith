import pytest

from backend.app.services.embedding_service import (
    EmbeddingError,
    EmbeddingService,
    FakeEmbeddingProvider,
    OpenAIEmbeddingProvider,
)


def test_embedding_service_batches_and_preserves_order():
    provider = FakeEmbeddingProvider(dimensions=4)
    service = EmbeddingService(provider=provider, batch_size=2)

    embeddings = service.embed_texts(["alpha", "beta", "gamma"])

    assert len(embeddings) == 3
    assert len(embeddings[0]) == 4
    assert provider.batches == [["alpha", "beta"], ["gamma"]]
    assert embeddings[0] == provider._embed_text("alpha")
    assert embeddings[1] == provider._embed_text("beta")
    assert embeddings[2] == provider._embed_text("gamma")


def test_embedding_service_rejects_empty_texts_and_queries():
    service = EmbeddingService(provider=FakeEmbeddingProvider())

    with pytest.raises(EmbeddingError, match="empty text"):
        service.embed_texts(["valid", "   "])

    with pytest.raises(EmbeddingError, match="empty query"):
        service.embed_query("")


def test_openai_embedding_provider_fails_clearly_without_api_key():
    provider = OpenAIEmbeddingProvider(api_key="")

    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        provider.embed_batch(["hello"])
