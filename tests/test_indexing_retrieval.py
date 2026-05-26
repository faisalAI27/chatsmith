import json

from backend.app.services.embedding_service import EmbeddingService, FakeEmbeddingProvider
from backend.app.services.indexing_service import index_knowledge, index_knowledge_file
from backend.app.services.retrieval_service import retrieve_relevant_chunks
from backend.app.services.vector_store import ChromaVectorStore


def _knowledge(website_id: str = "site-a") -> dict:
    return {
        "metadata": {
            "website_id": website_id,
            "url": "https://example.com",
            "normalized_url": "https://example.com",
            "name": "Example",
            "scraping_version": "v2",
        },
        "pages": [
            {
                "page_url": "https://example.com/services",
                "normalized_page_url": "https://example.com/services",
                "page_type": "services",
                "title": "Services",
                "description": "Services page",
                "extraction_method": "static",
                "sections": [
                    {
                        "heading": "Automation",
                        "content": "We offer workflow automation and support services.",
                    }
                ],
                "paragraphs": ["Customers use our automation tools to launch faster."],
                "faqs": [{"question": "Do you offer support?", "answer": "Yes, support is included."}],
                "tables": [],
                "images": [],
                "structured_data": [],
                "content": "We offer workflow automation and support services.",
            }
        ],
        "primary_content": {"source": "website_scraping", "reliability": "high", "pages": []},
    }


def _services(tmp_path):
    embedder = EmbeddingService(provider=FakeEmbeddingProvider(), batch_size=2)
    store = ChromaVectorStore(db_dir=tmp_path / "chroma", collection_name="index_chunks")
    return embedder, store


def test_index_knowledge_builds_embeds_and_upserts_chunks(tmp_path):
    embedder, store = _services(tmp_path)

    summary = index_knowledge(_knowledge(), embedding_service=embedder, vector_store=store)

    assert summary["website_id"] == "site-a"
    assert summary["chunks_built"] > 0
    assert summary["chunks_indexed"] == summary["chunks_built"]
    assert summary["chunks_deleted"] == 0
    assert summary["by_type"]["section"] == 1
    assert summary["collection_name"] == "index_chunks"
    assert summary["vector_db_provider"] == "chroma"
    assert summary["embedding_model"] == "fake-embedding-model"
    assert summary["errors"] == []
    assert store.count_website_chunks("site-a") == summary["chunks_indexed"]


def test_force_reindex_deletes_existing_chunks_before_upsert(tmp_path):
    embedder, store = _services(tmp_path)

    first = index_knowledge(_knowledge(), embedding_service=embedder, vector_store=store)
    second = index_knowledge(_knowledge(), embedding_service=embedder, vector_store=store)

    assert first["chunks_deleted"] == 0
    assert second["chunks_deleted"] == first["chunks_indexed"]
    assert store.count_website_chunks("site-a") == second["chunks_indexed"]


def test_index_knowledge_reports_embedding_failures(tmp_path):
    class FailingEmbeddingService:
        model = "failing"

        def embed_texts(self, texts):
            raise RuntimeError("embedding service unavailable")

    _, store = _services(tmp_path)

    summary = index_knowledge(
        _knowledge(),
        embedding_service=FailingEmbeddingService(),
        vector_store=store,
    )

    assert summary["chunks_built"] > 0
    assert summary["chunks_indexed"] == 0
    assert summary["errors"] == ["embedding service unavailable"]


def test_index_knowledge_file_loads_json_and_indexes(tmp_path):
    embedder, store = _services(tmp_path)
    path = tmp_path / "knowledge.json"
    path.write_text(json.dumps(_knowledge()), encoding="utf-8")

    summary = index_knowledge_file(path, embedding_service=embedder, vector_store=store)

    assert summary["website_id"] == "site-a"
    assert summary["chunks_indexed"] > 0


def test_retrieval_filters_by_website_id_and_returns_source_fields(tmp_path):
    embedder = EmbeddingService(provider=FakeEmbeddingProvider(), batch_size=2)
    store = ChromaVectorStore(db_dir=tmp_path / "chroma", collection_name="retrieve_chunks")
    index_knowledge(_knowledge("site-a"), embedding_service=embedder, vector_store=store)
    other = _knowledge("site-b")
    other["metadata"]["url"] = "https://other.example.com"
    other["metadata"]["normalized_url"] = "https://other.example.com"
    other["pages"][0]["page_url"] = "https://other.example.com/services"
    other["pages"][0]["normalized_page_url"] = "https://other.example.com/services"
    index_knowledge(other, embedding_service=embedder, vector_store=store)

    results = retrieve_relevant_chunks(
        "site-a",
        "What services are offered?",
        top_k=3,
        embedding_service=embedder,
        vector_store=store,
    )

    assert results
    assert all(result["metadata"]["website_id"] == "site-a" for result in results)
    assert {"chunk_id", "text", "score", "source_url", "page_title", "chunk_type", "metadata"}.issubset(
        results[0].keys()
    )
    assert results[0]["source_url"] == "https://example.com/services"
    assert results[0]["page_title"] == "Services"


def test_retrieval_can_filter_by_chunk_type(tmp_path):
    embedder, store = _services(tmp_path)
    index_knowledge(_knowledge(), embedding_service=embedder, vector_store=store)

    results = retrieve_relevant_chunks(
        "site-a",
        "support question",
        top_k=5,
        chunk_types=["faq"],
        embedding_service=embedder,
        vector_store=store,
    )

    assert results
    assert {result["chunk_type"] for result in results} == {"faq"}
