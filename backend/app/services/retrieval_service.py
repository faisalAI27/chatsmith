from ..core.config import get_settings
from .embedding_service import EmbeddingService
from .retrieval_ranker import rerank_chunks
from .vector_store import ChromaVectorStore


def retrieve_relevant_chunks(
    website_id: str,
    query: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
    embedding_service: EmbeddingService | None = None,
    vector_store: ChromaVectorStore | None = None,
    include_debug: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
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
    hybrid_enabled = bool(getattr(settings, "hybrid_retrieval_enabled", True))
    vector_multiplier = _positive_int(
        getattr(settings, "hybrid_vector_candidate_multiplier", 4),
        default=4,
    )
    lexical_limit = _positive_int(
        getattr(settings, "hybrid_lexical_candidate_limit", 2000),
        default=2000,
    )

    filters = {}
    if chunk_types:
        filters["chunk_type"] = [chunk_type for chunk_type in chunk_types if chunk_type]

    embedder = embedding_service or EmbeddingService()
    store = vector_store or ChromaVectorStore()
    query_embedding = embedder.embed_query(clean_query)
    vector_candidate_k = resolved_top_k
    if hybrid_enabled:
        vector_candidate_k = max(resolved_top_k * vector_multiplier, 20)

    vector_candidates = store.query_similar(
        query_embedding=query_embedding,
        website_id=clean_website_id,
        top_k=vector_candidate_k,
        filters=filters,
    )
    lexical_candidates = []
    if hybrid_enabled and hasattr(store, "get_website_chunks"):
        lexical_candidates = store.get_website_chunks(
            website_id=clean_website_id,
            limit=lexical_limit,
            chunk_types=filters.get("chunk_type"),
        )

    if hybrid_enabled:
        combined_candidates = _dedupe_candidates(vector_candidates + lexical_candidates)
        final_results, rank_debug = rerank_chunks(clean_query, combined_candidates, resolved_top_k)
    else:
        combined_candidates = vector_candidates
        final_results = vector_candidates[:resolved_top_k]
        rank_debug = {
            "query_intent": "semantic_only",
            "input_count": len(vector_candidates),
            "deduped_count": len(vector_candidates),
            "final_count": len(final_results),
            "top_scores": [],
        }

    debug = {
        "hybrid_enabled": hybrid_enabled,
        "vector_candidate_k": vector_candidate_k,
        "candidate_count_vector": len(vector_candidates),
        "candidate_count_lexical": len(lexical_candidates),
        "candidate_count_combined": len(combined_candidates),
        "final_count": len(final_results),
        "query_intent": rank_debug.get("query_intent", ""),
        "rerank": rank_debug,
    }

    if include_debug:
        return final_results, debug
    return final_results


def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
    by_id = {}
    no_id = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        chunk_id = str(candidate.get("chunk_id") or "").strip()
        if not chunk_id:
            no_id.append(candidate)
            continue
        existing = by_id.get(chunk_id)
        if existing is None or _has_vector_score(candidate) and not _has_vector_score(existing):
            by_id[chunk_id] = candidate
    return list(by_id.values()) + no_id


def _has_vector_score(candidate: dict) -> bool:
    return candidate.get("score") is not None or candidate.get("distance") is not None


def _positive_int(value: int | str | None, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
