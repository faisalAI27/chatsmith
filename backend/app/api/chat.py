import os

from fastapi import APIRouter, HTTPException
from openai import OpenAI

from ..models.chat import ChatRequest, ChatResponse, ChatMessage
from ..services.metrics_logger import (
    log_chat_answer,
    save_chat_answer_to_supabase,
    metrics_enabled,
)

router = APIRouter()


def get_openai_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is required to generate chat responses.",
        )
    return OpenAI()


@router.post("/", response_model=ChatResponse, summary="Chat using generated system prompt")
async def chat(req: ChatRequest):
    if not req.system_prompt:
        raise HTTPException(status_code=400, detail="system_prompt is required")

    messages = [{"role": "system", "content": req.system_prompt}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        answer = resp.choices[0].message.content or ""

        if metrics_enabled():
            provenance = "primary_plus_secondary" if "SECONDARY SOURCE" in req.system_prompt else "primary_only"
            try:
                # Best-effort file log
                log_chat_answer(
                    question=req.messages[-1].content if req.messages else "",
                    answer=answer,
                    provenance=provenance,
                    user=None,
                )
                # Best-effort Supabase log
                save_chat_answer_to_supabase(
                    question=req.messages[-1].content if req.messages else "",
                    answer=answer,
                    system_prompt=req.system_prompt,
                    user_id=None,
                    url=None,
                )
            except Exception as log_exc:
                print(f"⚠️ Metrics logging skipped: {log_exc}")

        return ChatResponse(message=ChatMessage(role="assistant", content=answer))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
