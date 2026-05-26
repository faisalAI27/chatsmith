from ..core.config import get_settings
from .embedding_service import EmbeddingService
from .vector_store import ChromaVectorStore


def retrieve_relevant_chunks(
    website_id: str,
    query: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
    embedding_service: EmbeddingService | None = None,
    vector_store: ChromaVectorStore | None = None,
) -> list[dict]:
    """Retrieve semantically similar chunks for a website without generating an answer."""
    settings = get_settings()
    clean_website_id = (website_id or "").strip()
    if not clean_website_id:
        raise ValueError("website_id is required")

    clean_query = " ".join(str(query or "").split())
    if not clean_query:
        raise ValueError("query is required")

    requested_top_k = top_k if top_k is not None else settings.retrieval_top_k
    try:
        resolved_top_k = int(requested_top_k)
    except (TypeError, ValueError):
        resolved_top_k = settings.retrieval_top_k
    resolved_top_k = max(1, resolved_top_k)

    filters = {}
    if chunk_types:
        filters["chunk_type"] = [chunk_type for chunk_type in chunk_types if chunk_type]

    embedder = embedding_service or EmbeddingService()
    store = vector_store or ChromaVectorStore()
    query_embedding = embedder.embed_query(clean_query)
    return store.query_similar(
        query_embedding=query_embedding,
        website_id=clean_website_id,
        top_k=resolved_top_k,
        filters=filters,
    )
