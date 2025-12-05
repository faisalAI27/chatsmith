from fastapi import APIRouter, HTTPException
from openai import OpenAI

from ..models.chat import ChatRequest, ChatResponse, ChatMessage

router = APIRouter()


@router.post("/", response_model=ChatResponse, summary="Chat using generated system prompt")
async def chat(req: ChatRequest):
    if not req.system_prompt:
        raise HTTPException(status_code=400, detail="system_prompt is required")

    messages = [{"role": "system", "content": req.system_prompt}]
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        answer = resp.choices[0].message.content or ""
        return ChatResponse(message=ChatMessage(role="assistant", content=answer))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
