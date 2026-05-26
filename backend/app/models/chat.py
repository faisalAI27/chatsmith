from pydantic import BaseModel, Field
from typing import Any, List, Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)
    website_id: Optional[str] = None
    question: Optional[str] = None
    system_prompt: Optional[str] = None
    top_k: Optional[int] = None
    chunk_types: Optional[List[str]] = None


class ChatSource(BaseModel):
    source_url: str = ""
    page_title: str = ""
    chunk_type: str = ""
    score: Optional[float] = None
    text_preview: str = ""


class ChatResponse(BaseModel):
    message: ChatMessage
    answer: str = ""
    sources: List[ChatSource] = Field(default_factory=list)
    mode: str = "legacy_prompt"
    warnings: List[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
