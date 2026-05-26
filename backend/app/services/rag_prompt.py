from typing import Any


MAX_CONTEXT_CHUNKS = 8
MAX_CHUNK_TEXT_CHARS = 1400
MAX_SOURCE_PREVIEW_CHARS = 240


def build_rag_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a compact, cited context block."""
    context_parts = []
    for index, chunk in enumerate((retrieved_chunks or [])[:MAX_CONTEXT_CHUNKS], start=1):
        if not isinstance(chunk, dict):
            continue
        text = _clean_text(chunk.get("text"))
        if not text:
            continue
        page_title = _clean_text(chunk.get("page_title")) or "Untitled page"
        source_url = _clean_text(chunk.get("source_url"))
        chunk_type = _clean_text(chunk.get("chunk_type")) or "chunk"
        score = chunk.get("score")
        score_text = f", score={score}" if score is not None else ""
        context_parts.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Page title: {page_title}",
                    f"URL: {source_url}",
                    f"Chunk type: {chunk_type}{score_text}",
                    f"Content: {text[:MAX_CHUNK_TEXT_CHARS]}",
                ]
            )
        )
    return "\n\n".join(context_parts)


def build_rag_messages(
    question: str,
    chat_history: list[dict],
    retrieved_chunks: list[dict],
) -> list[dict]:
    """Build OpenAI chat messages grounded only in retrieved website chunks."""
    context = build_rag_context(retrieved_chunks)
    system_message = (
        "You are ChatSMITH's retrieval-grounded website assistant.\n"
        "Answer only from the retrieved website context below.\n"
        "If the context does not contain enough information, say the website does not provide enough information.\n"
        "Do not invent details, prices, policies, dates, or contact information.\n"
        "Be concise and helpful. Source links are provided separately by the application.\n\n"
        "=== RETRIEVED WEBSITE CONTEXT ===\n"
        f"{context or 'No relevant context was retrieved.'}\n"
        "=== END CONTEXT ==="
    )

    messages = [{"role": "system", "content": system_message}]
    prior_history = _history_without_latest_question(chat_history or [], question)
    messages.extend(prior_history[-6:])
    messages.append({"role": "user", "content": _clean_text(question)})
    return messages


def format_sources(retrieved_chunks: list[dict], max_sources: int = 5) -> list[dict]:
    """Format and deduplicate retrieved chunks for API/frontend source display."""
    sources = []
    seen_keys = set()
    for chunk in retrieved_chunks or []:
        if not isinstance(chunk, dict):
            continue
        source_url = _clean_text(chunk.get("source_url"))
        page_title = _clean_text(chunk.get("page_title"))
        chunk_type = _clean_text(chunk.get("chunk_type"))
        text = _clean_text(chunk.get("text"))
        if not source_url and not text:
            continue

        key = source_url or chunk.get("chunk_id") or text[:80]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        sources.append(
            {
                "source_url": source_url,
                "page_title": page_title,
                "chunk_type": chunk_type,
                "score": chunk.get("score"),
                "text_preview": text[:MAX_SOURCE_PREVIEW_CHARS],
            }
        )
        if len(sources) >= max_sources:
            break
    return sources


def _history_without_latest_question(chat_history: list[dict], question: str) -> list[dict]:
    latest_question = _clean_text(question)
    history = []
    for message in chat_history:
        if not isinstance(message, dict):
            continue
        role = _clean_text(message.get("role"))
        content = _clean_text(message.get("content"))
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content})

    if history and history[-1]["role"] == "user" and history[-1]["content"] == latest_question:
        return history[:-1]
    return history


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())
