from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.app.api import chat as chat_api
from backend.app.models.chat import ChatMessage, ChatRequest


class FakeOpenAIClient:
    def __init__(self, answer="Grounded answer."):
        self.calls = []
        self.answer = answer
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, model, messages):
        self.calls.append({"model": model, "messages": messages})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.answer)
                )
            ]
        )


def _retrieved_chunks():
    return [
        {
            "chunk_id": "chunk-1",
            "text": "The website offers workflow automation services.",
            "score": 0.91,
            "source_url": "https://example.com/services",
            "page_title": "Services",
            "chunk_type": "section",
            "metadata": {"website_id": "site-a"},
        },
        {
            "chunk_id": "chunk-2",
            "text": "Support is included during launch.",
            "score": 0.82,
            "source_url": "https://example.com/faq",
            "page_title": "FAQ",
            "chunk_type": "faq",
            "metadata": {"website_id": "site-a"},
        },
    ]


def test_chat_request_accepts_website_id_and_legacy_system_prompt():
    request = ChatRequest(
        website_id="site-a",
        system_prompt="legacy prompt",
        messages=[ChatMessage(role="user", content="What services?")],
    )

    assert request.website_id == "site-a"
    assert request.system_prompt == "legacy prompt"
    assert request.messages[0].content == "What services?"


@pytest.mark.asyncio
async def test_chat_uses_rag_retrieval_when_website_id_is_provided(monkeypatch):
    fake_client = FakeOpenAIClient(answer="The site offers workflow automation services.")
    calls = {}

    def fake_retrieve_relevant_chunks(website_id, query, top_k=None, chunk_types=None):
        calls["website_id"] = website_id
        calls["query"] = query
        calls["top_k"] = top_k
        calls["chunk_types"] = chunk_types
        return _retrieved_chunks()

    monkeypatch.setattr(chat_api, "retrieve_relevant_chunks", fake_retrieve_relevant_chunks)
    monkeypatch.setattr(chat_api, "get_openai_client", lambda: fake_client)

    response = await chat_api.chat(
        ChatRequest(
            website_id="site-a",
            question="What services are offered?",
            system_prompt="FULL LEGACY SYSTEM PROMPT",
            messages=[ChatMessage(role="user", content="What services are offered?")],
        )
    )

    assert response.mode == "rag"
    assert response.answer == "The site offers workflow automation services."
    assert response.message.content == response.answer
    assert response.sources[0].source_url == "https://example.com/services"
    assert calls["website_id"] == "site-a"
    assert calls["query"] == "What services are offered?"

    sent_messages = fake_client.calls[0]["messages"]
    assert "workflow automation services" in sent_messages[0]["content"]
    assert "FULL LEGACY SYSTEM PROMPT" not in sent_messages[0]["content"]


@pytest.mark.asyncio
async def test_chat_falls_back_to_legacy_prompt_when_retrieval_fails(monkeypatch):
    fake_client = FakeOpenAIClient(answer="Legacy answer.")

    def failing_retrieve(*args, **kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(chat_api, "retrieve_relevant_chunks", failing_retrieve)
    monkeypatch.setattr(chat_api, "get_openai_client", lambda: fake_client)

    response = await chat_api.chat(
        ChatRequest(
            website_id="site-a",
            system_prompt="Legacy system prompt",
            messages=[ChatMessage(role="user", content="Question?")],
        )
    )

    assert response.mode == "legacy_prompt"
    assert response.answer == "Legacy answer."
    assert response.sources == []
    assert response.warnings == ["RAG retrieval failed: vector store unavailable"]
    assert fake_client.calls[0]["messages"][0] == {
        "role": "system",
        "content": "Legacy system prompt",
    }


@pytest.mark.asyncio
async def test_chat_returns_warning_when_no_chunks_and_no_fallback(monkeypatch):
    monkeypatch.setattr(chat_api, "retrieve_relevant_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        chat_api,
        "get_openai_client",
        lambda: (_ for _ in ()).throw(AssertionError("OpenAI should not be called")),
    )

    response = await chat_api.chat(
        ChatRequest(
            website_id="site-a",
            messages=[ChatMessage(role="user", content="Unknown question?")],
        )
    )

    assert response.mode == "rag"
    assert response.sources == []
    assert response.metadata["retrieved_chunks"] == 0
    assert "No relevant indexed chunks" in response.warnings[0]
    assert "could not find relevant indexed website context" in response.answer


@pytest.mark.asyncio
async def test_chat_raises_clear_error_when_retrieval_fails_without_fallback(monkeypatch):
    monkeypatch.setattr(
        chat_api,
        "retrieve_relevant_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db down")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat_api.chat(
            ChatRequest(
                website_id="site-a",
                messages=[ChatMessage(role="user", content="Question?")],
            )
        )

    assert exc_info.value.status_code == 503
    assert "RAG retrieval failed: db down" in exc_info.value.detail


def test_missing_openai_api_key_gives_clear_request_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        chat_api.get_openai_client()

    assert exc_info.value.status_code == 503
    assert "OPENAI_API_KEY is required" in exc_info.value.detail
