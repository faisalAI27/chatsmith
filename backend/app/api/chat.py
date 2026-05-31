import os
from typing import List

from fastapi import APIRouter, HTTPException
from openai import OpenAI

from ..models.chat import ChatMessage, ChatRequest, ChatResponse
from ..services.metrics_logger import (
    log_chat_answer,
    metrics_enabled,
)
from ..services.rag_prompt import build_rag_messages, format_sources
from ..services.rag_quality import (
    build_retrieval_debug_summary,
    detect_low_quality_retrieval,
    is_context_sufficient,
    validate_sources,
)
from ..services.retrieval_service import retrieve_relevant_chunks

router = APIRouter()


def get_openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is required to generate chat responses.",
        )
    return OpenAI()


@router.post("/", response_model=ChatResponse, summary="Chat using RAG retrieval when available")
async def chat(req: ChatRequest):
    question = _latest_question(req)
    warnings: List[str] = []

    if req.website_id:
        try:
            retrieval_result = retrieve_relevant_chunks(
                website_id=req.website_id,
                query=question,
                top_k=req.top_k,
                chunk_types=req.chunk_types,
                include_debug=True,
            )
            if isinstance(retrieval_result, tuple):
                retrieved_chunks, hybrid_debug = retrieval_result
            else:
                retrieved_chunks, hybrid_debug = retrieval_result, {}
        except Exception as exc:
            warnings.append(f"RAG retrieval failed: {exc}")
            if req.system_prompt:
                return _run_legacy_prompt_chat(req, warnings=warnings)
            raise HTTPException(status_code=503, detail=warnings[-1])

        retrieval_debug = build_retrieval_debug_summary(retrieved_chunks)
        retrieval_debug["hybrid"] = hybrid_debug
        warnings.extend(detect_low_quality_retrieval(retrieved_chunks))

        if retrieved_chunks and is_context_sufficient(retrieved_chunks):
            messages = build_rag_messages(
                question=question,
                chat_history=[message.model_dump() for message in req.messages],
                retrieved_chunks=retrieved_chunks,
            )
            answer = _call_openai_chat(messages)
            sources = format_sources(retrieved_chunks)
            warnings.extend(validate_sources(sources))
            _log_chat_if_enabled(question=question, answer=answer, provenance="rag")
            return ChatResponse(
                message=ChatMessage(role="assistant", content=answer),
                answer=answer,
                sources=sources,
                mode="rag",
                warnings=_dedupe_warnings(warnings),
                retrieval_debug=retrieval_debug,
                metadata={"website_id": req.website_id, "retrieved_chunks": len(retrieved_chunks)},
            )

        if req.system_prompt:
            return _run_legacy_prompt_chat(req, warnings=warnings)
        answer = "The website does not provide enough information from indexed context to answer that question."
        return ChatResponse(
            message=ChatMessage(role="assistant", content=answer),
            answer=answer,
            sources=[],
            mode="rag",
            warnings=_dedupe_warnings(warnings),
            retrieval_debug=retrieval_debug,
            metadata={"website_id": req.website_id, "retrieved_chunks": 0},
        )

    return _run_legacy_prompt_chat(req, warnings=warnings)


def _run_legacy_prompt_chat(req: ChatRequest, warnings: List[str] | None = None) -> ChatResponse:
    if not req.system_prompt:
        raise HTTPException(status_code=400, detail="website_id or system_prompt is required")

    messages = [{"role": "system", "content": req.system_prompt}]
    for message in req.messages:
        messages.append({"role": message.role, "content": message.content})

    answer = _call_openai_chat(messages)
    provenance = "primary_plus_secondary" if "SECONDARY SOURCE" in req.system_prompt else "primary_only"
    _log_chat_if_enabled(question=_latest_question(req), answer=answer, provenance=provenance)
    return ChatResponse(
        message=ChatMessage(role="assistant", content=answer),
        answer=answer,
        sources=[],
        mode="legacy_prompt",
        warnings=_dedupe_warnings(warnings or []),
    )


def _call_openai_chat(messages: list[dict]) -> str:
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return response.choices[0].message.content or ""
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _latest_question(req: ChatRequest) -> str:
    question = (req.question or "").strip()
    if question:
        return question

    for message in reversed(req.messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()

    raise HTTPException(status_code=400, detail="A user question is required")


def _log_chat_if_enabled(question: str, answer: str, provenance: str) -> None:
    if not metrics_enabled():
        return
    try:
        log_chat_answer(
            question=question,
            answer=answer,
            provenance=provenance,
            user=None,
        )
    except Exception as log_exc:
        print(f"⚠️ Metrics logging skipped: {log_exc}")


def _dedupe_warnings(warnings: List[str]) -> List[str]:
    deduped = []
    for warning in warnings:
        if warning and warning not in deduped:
            deduped.append(warning)
    return deduped
