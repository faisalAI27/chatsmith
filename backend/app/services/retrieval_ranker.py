import re
from typing import Any, Dict, List, Tuple


CONTACT_KEYWORDS = {
    "contact",
    "phone",
    "number",
    "call",
    "whatsapp",
    "email",
    "customer",
    "service",
    "support",
    "helpline",
}
LOCATION_KEYWORDS = {
    "store",
    "location",
    "locations",
    "address",
    "branch",
    "branches",
    "where",
    "lahore",
    "islamabad",
    "karachi",
    "mall",
}
POLICY_KEYWORDS = {
    "privacy",
    "terms",
    "policy",
    "refund",
    "return",
    "exchange",
    "shipping",
    "delivery",
}
PRODUCT_KEYWORDS = {
    "price",
    "product",
    "size",
    "color",
    "material",
    "pants",
    "shirt",
    "shoes",
    "collection",
}
IMAGE_KEYWORDS = {
    "image",
    "picture",
    "photo",
    "look",
    "design",
    "style",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "me",
    "of",
    "on",
    "or",
    "the",
    "them",
    "this",
    "to",
    "what",
    "with",
}

PHONE_RE = re.compile(r"(\+92|0092|03\d{2}[\s-]?\d{6,7}|\b\d{4}[\s-]?\d{7}\b)")
EMAIL_RE = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")


def normalize_text(text: str) -> str:
    """Normalize text for deterministic lexical matching."""
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9+@./:-]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_query_terms(query: str) -> list[str]:
    """Extract useful query terms for lexical overlap scoring."""
    normalized = normalize_text(query)
    terms = re.findall(r"[a-z0-9+@./:-]{2,}", normalized)
    return [term for term in terms if term not in STOPWORDS]


def classify_query_intent(query: str) -> str:
    """Classify a query into a small deterministic retrieval intent."""
    normalized = normalize_text(query)
    terms = set(extract_query_terms(normalized))

    if "customer service" in normalized or terms & CONTACT_KEYWORDS:
        return "contact"
    if "store location" in normalized or terms & LOCATION_KEYWORDS:
        return "location"
    if "privacy policy" in normalized or "terms of service" in normalized or terms & POLICY_KEYWORDS:
        return "policy"
    if terms & IMAGE_KEYWORDS:
        return "image"
    if terms & PRODUCT_KEYWORDS:
        return "product"
    if "faq" in terms or "question" in terms or "help" in terms:
        return "faq"
    return "general"


def score_keyword_overlap(query: str, chunk: dict) -> float:
    """Score direct query-term overlap against chunk text and source metadata."""
    terms = extract_query_terms(query)
    if not terms:
        return 0.0
    searchable = _chunk_search_text(chunk)
    hits = sum(1 for term in terms if term in searchable)
    return round(hits / len(terms), 6)


def score_intent_match(query_intent: str, chunk: dict) -> float:
    """Score exact factual signals for the query intent."""
    searchable = _chunk_search_text(chunk)
    source_text = _source_text(chunk)

    if query_intent == "contact":
        score = 0.0
        if PHONE_RE.search(searchable):
            score += 0.55
        if EMAIL_RE.search(searchable):
            score += 0.35
        for phrase in ("whatsapp", "contact us", "customer service", "support", "helpline", "call us"):
            if phrase in searchable:
                score += 0.18
        if any(path in source_text for path in ("contact", "support", "customer-service")):
            score += 0.18
        return min(1.0, round(score, 6))

    if query_intent == "location":
        score = 0.0
        for phrase in (
            "store location",
            "address",
            "branch",
            "branches",
            "lahore",
            "islamabad",
            "karachi",
            "mall",
            "markaz",
            "block",
            "floor",
            "shop",
            "maps",
        ):
            if phrase in searchable:
                score += 0.16
        if any(path in source_text for path in ("store", "location", "branch", "contact")):
            score += 0.16
        return min(1.0, round(score, 6))

    if query_intent == "policy":
        score = 0.0
        for phrase in (
            "refund",
            "return",
            "exchange",
            "privacy policy",
            "terms of service",
            "shipping",
            "delivery",
        ):
            if phrase in searchable:
                score += 0.2
        if any(path in source_text for path in ("privacy", "terms", "refund", "return", "shipping")):
            score += 0.16
        return min(1.0, round(score, 6))

    if query_intent == "image":
        score = 0.0
        if _chunk_type(chunk) == "image_context":
            score += 0.45
        for phrase in ("image", "picture", "photo", "look", "design", "style", "color"):
            if phrase in searchable:
                score += 0.12
        return min(1.0, round(score, 6))

    if query_intent == "product":
        score = 0.0
        for phrase in ("price", "size", "color", "material", "collection", "product"):
            if phrase in searchable:
                score += 0.14
        if "/products/" in source_text or "product" in source_text:
            score += 0.2
        return min(1.0, round(score, 6))

    if query_intent == "faq":
        score = 0.0
        if _chunk_type(chunk) == "faq":
            score += 0.45
        if any(phrase in searchable for phrase in ("faq", "question", "answer", "help", "support")):
            score += 0.35
        return min(1.0, round(score, 6))

    return 0.0


def score_chunk_quality(chunk: dict) -> float:
    """Score whether a chunk has enough answer-bearing text."""
    text = _chunk_text(chunk)
    word_count = len(text.split())
    chunk_type = _chunk_type(chunk)

    if word_count == 0:
        return 0.0
    if chunk_type == "page_summary" and _is_short_page_summary(text):
        return 0.1
    if word_count < 8:
        return 0.25
    if word_count < 25:
        return 0.55
    if word_count < 120:
        return 0.85
    return 1.0


def calculate_rerank_score(query: str, chunk: dict) -> dict:
    """Return final rerank score and individual score components."""
    query_intent = classify_query_intent(query)
    vector_score = _vector_score(chunk)
    keyword_overlap = score_keyword_overlap(query, chunk)
    intent_match = score_intent_match(query_intent, chunk)
    chunk_type_weight = _chunk_type_weight(query_intent, chunk)
    source_relevance = _source_relevance_score(query_intent, chunk)
    content_quality = score_chunk_quality(chunk)
    penalty = _penalty_score(query_intent, chunk, intent_match)

    final_score = (
        (0.35 * vector_score)
        + (1.20 * keyword_overlap)
        + (1.65 * intent_match)
        + (0.85 * chunk_type_weight)
        + (0.75 * source_relevance)
        + (0.45 * content_quality)
        - penalty
    )

    return {
        "query_intent": query_intent,
        "final_score": round(final_score, 6),
        "vector_score": round(vector_score, 6),
        "keyword_overlap": keyword_overlap,
        "intent_match": intent_match,
        "chunk_type_weight": round(chunk_type_weight, 6),
        "source_relevance": round(source_relevance, 6),
        "content_quality": round(content_quality, 6),
        "penalty": round(penalty, 6),
    }


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> tuple[list[dict], dict]:
    """Rerank and deduplicate vector/lexical candidate chunks."""
    scored = []
    for index, chunk in enumerate(chunks or []):
        if not isinstance(chunk, dict):
            continue
        components = calculate_rerank_score(query, chunk)
        candidate = dict(chunk)
        candidate["rerank_score"] = components["final_score"]
        candidate["rerank_components"] = components
        candidate["query_intent"] = components["query_intent"]
        scored.append((candidate, index))

    scored.sort(
        key=lambda item: (
            item[0]["rerank_score"],
            item[0]["rerank_components"]["intent_match"],
            item[0]["rerank_components"]["keyword_overlap"],
            item[0]["rerank_components"]["vector_score"],
            -item[1],
        ),
        reverse=True,
    )

    deduped = _dedupe_scored_chunks(scored)
    final_results = deduped[: max(1, int(top_k or 5))]
    debug = {
        "query_intent": classify_query_intent(query),
        "input_count": len([chunk for chunk in chunks or [] if isinstance(chunk, dict)]),
        "deduped_count": len(deduped),
        "final_count": len(final_results),
        "top_scores": [
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "page_title": chunk.get("page_title", ""),
                "source_url": chunk.get("source_url", ""),
                "rerank_score": chunk.get("rerank_score"),
                "components": chunk.get("rerank_components", {}),
            }
            for chunk in final_results
        ],
    }
    return final_results, debug


def _dedupe_scored_chunks(scored: List[Tuple[dict, int]]) -> list[dict]:
    by_chunk_id = set()
    by_text = set()
    deduped = []
    for chunk, _index in scored:
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        text_key = normalize_text(_chunk_text(chunk))[:320]
        if chunk_id and chunk_id in by_chunk_id:
            continue
        if text_key and text_key in by_text:
            continue
        if chunk_id:
            by_chunk_id.add(chunk_id)
        if text_key:
            by_text.add(text_key)
        deduped.append(chunk)
    return deduped


def _chunk_type_weight(query_intent: str, chunk: dict) -> float:
    chunk_type = _chunk_type(chunk)
    if chunk_type == "section":
        return 0.95
    if chunk_type == "paragraph_group":
        return 0.9
    if chunk_type == "faq":
        return 0.9 if query_intent in {"contact", "policy", "faq", "general"} else 0.75
    if chunk_type == "table":
        return 0.7
    if chunk_type == "structured_data":
        return 0.45 if query_intent in {"contact", "location", "policy"} else 0.35
    if chunk_type == "image_context":
        if query_intent == "image":
            return 0.9
        if query_intent == "product":
            return 0.45
        return 0.08
    if chunk_type == "page_summary":
        return 0.18
    return 0.45


def _source_relevance_score(query_intent: str, chunk: dict) -> float:
    source = _source_text(chunk)
    score = 0.0
    if query_intent == "contact":
        if "contact" in source:
            score += 0.55
        if any(term in source for term in ("support", "customer-service", "customer_service", "whatsapp")):
            score += 0.35
        if "/pages/contact" in source:
            score += 0.2
    elif query_intent == "location":
        if any(term in source for term in ("store", "location", "branch", "contact")):
            score += 0.55
        if any(city in source for city in ("lahore", "islamabad", "karachi")):
            score += 0.25
    elif query_intent == "policy":
        if any(term in source for term in ("privacy", "terms", "refund", "return", "shipping", "delivery")):
            score += 0.65
    elif query_intent == "faq":
        if any(term in source for term in ("faq", "help", "support")):
            score += 0.6
    elif query_intent == "product":
        if any(term in source for term in ("product", "collection", "shop")):
            score += 0.5
    elif query_intent == "image":
        if any(term in source for term in ("image", "product", "collection")):
            score += 0.35
    return min(1.0, round(score, 6))


def _penalty_score(query_intent: str, chunk: dict, intent_match: float) -> float:
    chunk_type = _chunk_type(chunk)
    text = _chunk_text(chunk)
    source = _source_text(chunk)
    penalty = 0.0

    if chunk_type == "page_summary" and _is_short_page_summary(text):
        penalty += 0.75
    elif chunk_type == "page_summary" and intent_match < 0.4:
        penalty += 0.35

    if chunk_type == "image_context" and query_intent in {"contact", "location", "policy", "faq"}:
        penalty += 0.65
    if chunk_type == "structured_data" and query_intent in {"contact", "location", "policy"} and intent_match < 0.4:
        penalty += 0.25
    if query_intent in {"contact", "location"} and "/products/" in source and intent_match < 0.75:
        penalty += 0.35
    if len(text.split()) < 6:
        penalty += 0.3

    return round(penalty, 6)


def _vector_score(chunk: dict) -> float:
    score = _to_float(chunk.get("score"))
    if score is not None:
        return max(0.0, min(1.0, score))
    distance = _to_float(chunk.get("distance"))
    if distance is not None:
        return max(0.0, min(1.0, 1.0 - distance))
    return 0.0


def _chunk_search_text(chunk: dict) -> str:
    return normalize_text(
        " ".join(
            [
                _chunk_text(chunk),
                str(chunk.get("page_title") or ""),
                str(chunk.get("source_url") or ""),
                str(chunk.get("chunk_type") or ""),
                str((chunk.get("metadata") or {}).get("heading") or ""),
            ]
        )
    )


def _source_text(chunk: dict) -> str:
    return normalize_text(f"{chunk.get('page_title') or ''} {chunk.get('source_url') or ''}")


def _chunk_text(chunk: dict) -> str:
    return re.sub(r"\s+", " ", str(chunk.get("text") or "")).strip()


def _chunk_type(chunk: dict) -> str:
    return re.sub(r"\s+", "_", str(chunk.get("chunk_type") or "").strip().lower())


def _is_short_page_summary(text: str) -> bool:
    normalized = normalize_text(text)
    return len(normalized.split()) <= 12 and normalized.startswith("page title")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
