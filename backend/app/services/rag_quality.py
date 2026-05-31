from collections import Counter
from typing import Any


MIN_USEFUL_TEXT_CHARS = 25
MAX_SOURCE_PREVIEW_CHARS = 240


def normalize_query(query: str) -> str:
    """Normalize a user query for deterministic checks and logging."""
    return " ".join(str(query or "").strip().split())


def is_context_sufficient(retrieved_chunks: list[dict], min_chunks: int = 1) -> bool:
    """Return True when enough retrieved chunks contain usable text."""
    useful_chunks = [
        chunk
        for chunk in retrieved_chunks or []
        if _chunk_text_length(chunk) >= MIN_USEFUL_TEXT_CHARS
    ]
    return len(useful_chunks) >= max(1, int(min_chunks or 1))


def calculate_context_coverage(retrieved_chunks: list[dict]) -> dict:
    """Summarize retrieval coverage for debugging and quality checks."""
    chunks = [chunk for chunk in retrieved_chunks or [] if isinstance(chunk, dict)]
    useful_chunks = [chunk for chunk in chunks if _chunk_text_length(chunk) >= MIN_USEFUL_TEXT_CHARS]
    source_keys = {_source_key(chunk) for chunk in chunks if _source_key(chunk)}
    chunk_types = Counter(_clean_text(chunk.get("chunk_type")) or "unknown" for chunk in chunks)
    scores = [_to_float(chunk.get("score")) for chunk in chunks if _to_float(chunk.get("score")) is not None]
    distances = [
        _to_float(chunk.get("distance"))
        for chunk in chunks
        if _to_float(chunk.get("distance")) is not None
    ]

    return {
        "chunks_retrieved": len(chunks),
        "usable_chunks": len(useful_chunks),
        "source_count": len(source_keys),
        "chunk_types": dict(sorted(chunk_types.items())),
        "missing_source_url_count": sum(1 for chunk in chunks if not _clean_text(chunk.get("source_url"))),
        "missing_page_title_count": sum(1 for chunk in chunks if not _clean_text(chunk.get("page_title"))),
        "empty_text_count": sum(1 for chunk in chunks if not _clean_text(chunk.get("text"))),
        "total_text_chars": sum(_chunk_text_length(chunk) for chunk in chunks),
        "average_score": round(sum(scores) / len(scores), 6) if scores else None,
        "average_distance": round(sum(distances) / len(distances), 6) if distances else None,
    }


def detect_low_quality_retrieval(retrieved_chunks: list[dict]) -> list[str]:
    """Return non-fatal warnings for weak retrieval results."""
    coverage = calculate_context_coverage(retrieved_chunks)
    warnings = []

    if coverage["chunks_retrieved"] == 0:
        return ["No relevant indexed chunks were found for this website."]

    if coverage["usable_chunks"] == 0:
        warnings.append("Retrieved chunks did not contain enough readable website text.")
    elif coverage["usable_chunks"] < coverage["chunks_retrieved"]:
        warnings.append("Some retrieved chunks had weak or empty text and were ignored by the prompt.")

    if coverage["missing_source_url_count"]:
        warnings.append("Some retrieved chunks are missing source URLs.")
    if coverage["missing_page_title_count"]:
        warnings.append("Some retrieved chunks are missing page titles.")

    if (
        coverage["chunks_retrieved"] > 1
        and coverage["source_count"] <= 1
        and (coverage["total_text_chars"] < 500 or coverage["missing_source_url_count"])
    ):
        warnings.append("Retrieved context is concentrated in one weak source, so the answer may be incomplete.")

    return warnings


def build_retrieval_debug_summary(retrieved_chunks: list[dict]) -> dict:
    """Build API-safe retrieval debug details."""
    summary = calculate_context_coverage(retrieved_chunks)
    summary["warnings"] = detect_low_quality_retrieval(retrieved_chunks)
    return summary


def validate_sources(sources: list[dict]) -> list[str]:
    """Validate formatted source records without failing the response."""
    warnings = []
    if not sources:
        return ["No sources were formatted for this answer."]

    seen = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            warnings.append(f"Source {index} is not a valid source object.")
            continue
        source_url = _clean_text(source.get("source_url"))
        page_title = _clean_text(source.get("page_title"))
        chunk_type = _clean_text(source.get("chunk_type"))
        preview = _clean_text(source.get("text_preview"))
        key = (source_url, page_title, chunk_type)
        if key in seen:
            warnings.append("Duplicate source records were returned.")
        seen.add(key)
        if not source_url:
            warnings.append(f"Source {index} is missing source_url.")
        if not page_title:
            warnings.append(f"Source {index} is missing page_title.")
        if not preview:
            warnings.append(f"Source {index} is missing text_preview.")
        if len(preview) > MAX_SOURCE_PREVIEW_CHARS + 3:
            warnings.append(f"Source {index} text_preview is too long.")

    return warnings


def _chunk_text_length(chunk: Any) -> int:
    if not isinstance(chunk, dict):
        return 0
    return len(_clean_text(chunk.get("text")))


def _source_key(chunk: dict) -> str:
    source_url = _clean_text(chunk.get("source_url"))
    page_title = _clean_text(chunk.get("page_title"))
    return "|".join(part for part in (source_url, page_title) if part)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
