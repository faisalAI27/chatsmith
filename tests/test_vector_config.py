from backend.app.core.config import get_settings


def test_vector_and_embedding_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("VECTOR_DB_PROVIDER", "chroma")
    monkeypatch.setenv("CHROMA_DB_DIR", "tmp/chroma")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "32")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.vector_db_provider == "chroma"
    assert settings.chroma_db_dir == "tmp/chroma"
    assert settings.chroma_collection_name == "test_collection"
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_batch_size == 32
    assert settings.retrieval_top_k == 7

    get_settings.cache_clear()
