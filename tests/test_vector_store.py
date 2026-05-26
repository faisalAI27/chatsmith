from backend.app.services.embedding_service import EmbeddingService, FakeEmbeddingProvider
from backend.app.services.vector_store import ChromaVectorStore, flatten_chunk_metadata


def _chunk(chunk_id: str, website_id: str, text: str, chunk_type: str = "section") -> dict:
    return {
        "chunk_id": chunk_id,
        "website_id": website_id,
        "source_url": f"https://example.com/{chunk_id}",
        "page_title": f"Page {chunk_id}",
        "page_type": "about",
        "chunk_type": chunk_type,
        "chunk_index": 0,
        "text": text,
        "metadata": {
            "website_url": "https://example.com",
            "normalized_url": f"https://example.com/{chunk_id}",
            "source_type": "primary",
            "reliability": "high",
            "extraction_method": "static",
            "heading": "Heading",
            "nested": {"a": 1},
            "none_value": None,
        },
    }


def _embeddings_for(chunks: list[dict]) -> list[list[float]]:
    return EmbeddingService(provider=FakeEmbeddingProvider(), batch_size=2).embed_texts(
        [chunk["text"] for chunk in chunks]
    )


def test_flatten_chunk_metadata_is_chroma_compatible():
    metadata = flatten_chunk_metadata(_chunk("c1", "site-a", "Example text"))

    assert metadata["website_id"] == "site-a"
    assert metadata["source_url"] == "https://example.com/c1"
    assert metadata["page_title"] == "Page c1"
    assert metadata["chunk_type"] == "section"
    assert metadata["chunk_index"] == 0
    assert metadata["nested"] == '{"a": 1}'
    assert metadata["none_value"] == ""


def test_chroma_vector_store_upsert_query_count_and_delete(tmp_path):
    chunks = [
        _chunk("c1", "site-a", "services pricing support", "section"),
        _chunk("c2", "site-b", "different website content", "section"),
    ]
    embeddings = _embeddings_for(chunks)
    store = ChromaVectorStore(db_dir=tmp_path / "chroma", collection_name="test_chunks")

    upsert_result = store.upsert_chunks(chunks, embeddings)
    site_a_results = store.query_similar(embeddings[0], website_id="site-a", top_k=5)
    site_b_results = store.query_similar(embeddings[0], website_id="site-b", top_k=5)

    assert upsert_result["chunks_upserted"] == 2
    assert store.count_website_chunks("site-a") == 1
    assert site_a_results[0]["chunk_id"] == "c1"
    assert site_a_results[0]["source_url"] == "https://example.com/c1"
    assert site_a_results[0]["page_title"] == "Page c1"
    assert site_a_results[0]["chunk_type"] == "section"
    assert site_b_results[0]["chunk_id"] == "c2"
    assert store.delete_website_chunks("site-a") == 1
    assert store.count_website_chunks("site-a") == 0
    assert store.count_website_chunks("site-b") == 1


def test_chroma_query_accepts_additional_metadata_filters(tmp_path):
    chunks = [
        _chunk("c1", "site-a", "services overview", "section"),
        _chunk("c2", "site-a", "question answer", "faq"),
    ]
    embeddings = _embeddings_for(chunks)
    store = ChromaVectorStore(db_dir=tmp_path / "chroma", collection_name="filtered_chunks")
    store.upsert_chunks(chunks, embeddings)

    results = store.query_similar(
        embeddings[0],
        website_id="site-a",
        top_k=5,
        filters={"chunk_type": ["faq"]},
    )

    assert [result["chunk_type"] for result in results] == ["faq"]
