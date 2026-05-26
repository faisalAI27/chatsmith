from backend.app.services import indexing_service, scrape_pipeline


def test_vector_indexing_best_effort_adds_success_stats(monkeypatch):
    def fake_index_knowledge(knowledge, force_reindex=True):
        return {
            "website_id": "site-a",
            "chunks_built": 3,
            "chunks_indexed": 3,
            "chunks_deleted": 0,
            "by_type": {"section": 3},
            "collection_name": "website_chunks",
            "vector_db_provider": "chroma",
            "embedding_model": "fake",
            "errors": [],
        }

    monkeypatch.setattr(indexing_service, "index_knowledge", fake_index_knowledge)
    stats = {}

    scrape_pipeline._index_knowledge_best_effort({"metadata": {}}, stats, force_reindex=True)

    assert stats["website_id"] == "site-a"
    assert stats["chunk_count"] == 3
    assert stats["vector_indexed"] is True
    assert stats["vector_indexing_summary"]["website_id"] == "site-a"
    assert "vector_indexing_warning" not in stats


def test_vector_indexing_best_effort_reports_warnings(monkeypatch):
    def fake_index_knowledge(knowledge, force_reindex=True):
        return {
            "website_id": "site-a",
            "chunks_built": 3,
            "chunks_indexed": 0,
            "chunks_deleted": 0,
            "by_type": {"section": 3},
            "collection_name": "website_chunks",
            "vector_db_provider": "chroma",
            "embedding_model": "fake",
            "errors": ["OPENAI_API_KEY is required to generate embeddings."],
        }

    monkeypatch.setattr(indexing_service, "index_knowledge", fake_index_knowledge)
    stats = {}

    scrape_pipeline._index_knowledge_best_effort({"metadata": {}}, stats, force_reindex=False)

    assert stats["chunk_count"] == 3
    assert stats["vector_indexed"] is False
    assert "OPENAI_API_KEY" in stats["vector_indexing_warning"]
