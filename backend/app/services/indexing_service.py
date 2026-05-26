import json
from pathlib import Path
from typing import Any

from ..core.config import get_settings
from .chunker import build_chunks_from_knowledge, summarize_chunks
from .embedding_service import EmbeddingService
from .scraper_schema import ensure_v2_knowledge_shape, make_website_id, normalize_url_for_cache
from .vector_store import ChromaVectorStore


def index_knowledge(
    knowledge: dict,
    force_reindex: bool = True,
    embedding_service: EmbeddingService | None = None,
    vector_store: ChromaVectorStore | None = None,
) -> dict:
    """Build chunks, embed them, and upsert them into the configured vector store."""
    settings = get_settings()
    shaped = ensure_v2_knowledge_shape(knowledge)
    metadata = shaped.get("metadata", {})
    normalized_url = metadata.get("normalized_url") or normalize_url_for_cache(metadata.get("url", ""))
    website_id = metadata.get("website_id") or make_website_id(normalized_url)
    chunks = build_chunks_from_knowledge(shaped)
    chunk_summary = summarize_chunks(chunks)
    errors: list[str] = []

    summary = {
        "website_id": website_id,
        "chunks_built": chunk_summary["total_chunks"],
        "chunks_indexed": 0,
        "chunks_deleted": 0,
        "by_type": chunk_summary["by_type"],
        "collection_name": getattr(vector_store, "collection_name", settings.chroma_collection_name),
        "vector_db_provider": getattr(vector_store, "provider", settings.vector_db_provider),
        "embedding_model": getattr(embedding_service, "model", settings.embedding_model),
        "errors": errors,
    }

    if not chunks:
        return summary

    if vector_store is None and (settings.vector_db_provider or "chroma").lower() != "chroma":
        errors.append(f"Unsupported vector DB provider: {settings.vector_db_provider}")
        return summary

    embedder = embedding_service or EmbeddingService()
    store = vector_store or ChromaVectorStore()
    summary["collection_name"] = store.collection_name
    summary["vector_db_provider"] = store.provider
    summary["embedding_model"] = embedder.model

    try:
        embeddings = embedder.embed_texts([chunk["text"] for chunk in chunks])
        if force_reindex:
            summary["chunks_deleted"] = store.delete_website_chunks(website_id)
        upsert_result = store.upsert_chunks(chunks, embeddings)
        summary["chunks_indexed"] = int(upsert_result.get("chunks_upserted", 0) or 0)
    except Exception as exc:
        errors.append(str(exc))

    return summary


def index_knowledge_file(
    path: str | Path,
    force_reindex: bool = True,
    embedding_service: EmbeddingService | None = None,
    vector_store: ChromaVectorStore | None = None,
) -> dict:
    """Load a knowledge JSON file and index it into the configured vector store."""
    knowledge_path = Path(path)
    try:
        knowledge = json.loads(knowledge_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "website_id": "",
            "chunks_built": 0,
            "chunks_indexed": 0,
            "chunks_deleted": 0,
            "by_type": {},
            "collection_name": _collection_name(vector_store),
            "vector_db_provider": _provider_name(vector_store),
            "embedding_model": _embedding_model(embedding_service),
            "errors": [f"Could not load knowledge file {knowledge_path}: {exc}"],
        }

    return index_knowledge(
        knowledge,
        force_reindex=force_reindex,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def _collection_name(vector_store: Any) -> str:
    if vector_store is not None:
        return getattr(vector_store, "collection_name", "")
    return get_settings().chroma_collection_name


def _provider_name(vector_store: Any) -> str:
    if vector_store is not None:
        return getattr(vector_store, "provider", "")
    return get_settings().vector_db_provider


def _embedding_model(embedding_service: Any) -> str:
    if embedding_service is not None:
        return getattr(embedding_service, "model", "")
    return get_settings().embedding_model
