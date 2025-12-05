from pydantic import BaseModel
from typing import List


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    system_prompt: str
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    message: ChatMessage
