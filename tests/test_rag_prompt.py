from backend.app.services.rag_prompt import build_rag_context, build_rag_messages, format_sources


def _chunks():
    return [
        {
            "chunk_id": "c1",
            "text": "The website offers automation consulting and implementation support.",
            "score": 0.92,
            "source_url": "https://example.com/services",
            "page_title": "Services",
            "chunk_type": "section",
        },
        {
            "chunk_id": "c2",
            "text": "Support is included for customers during launch.",
            "score": 0.84,
            "source_url": "https://example.com/services",
            "page_title": "Services",
            "chunk_type": "faq",
        },
        {
            "chunk_id": "c3",
            "text": "Pricing details are available on request.",
            "score": 0.72,
            "source_url": "https://example.com/pricing",
            "page_title": "Pricing",
            "chunk_type": "paragraph_group",
        },
    ]


def test_build_rag_context_includes_retrieved_chunk_text_and_source_labels():
    context = build_rag_context(_chunks())

    assert "[Source 1]" in context
    assert "automation consulting" in context
    assert "https://example.com/services" in context
    assert "Title: Services" in context
    assert "Type: section" in context
    assert "Content:" in context


def test_build_rag_messages_uses_retrieved_context_without_legacy_prompt():
    messages = build_rag_messages(
        question="What services are offered?",
        chat_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What services are offered?"},
        ],
        retrieved_chunks=_chunks(),
    )

    system_content = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "Answer only from the retrieved public website context" in system_content
    assert "The website does not provide enough information" in system_content
    assert "Do not invent details" in system_content
    assert "automation consulting" in system_content
    assert "FULL LEGACY SYSTEM PROMPT" not in system_content
    assert messages[-1] == {"role": "user", "content": "What services are offered?"}
    assert {"role": "user", "content": "Hello"} in messages
    assert {"role": "assistant", "content": "Hi"} in messages


def test_format_sources_deduplicates_by_source_url_and_keeps_preview():
    sources = format_sources(_chunks())

    assert len(sources) == 3
    assert sources[0] == {
        "source_url": "https://example.com/services",
        "page_title": "Services",
        "chunk_type": "section",
        "score": 0.92,
        "distance": None,
        "text_preview": "The website offers automation consulting and implementation support.",
    }
    assert sources[1]["chunk_type"] == "faq"
    assert sources[2]["source_url"] == "https://example.com/pricing"


def test_format_sources_truncates_previews_and_handles_missing_metadata():
    long_text = "A" * 300
    sources = format_sources(
        [
            {
                "chunk_id": "missing-metadata",
                "text": long_text,
                "score": None,
                "source_url": "",
                "page_title": "",
                "chunk_type": "",
            }
        ]
    )

    assert sources[0]["source_url"] == ""
    assert sources[0]["page_title"] == ""
    assert sources[0]["chunk_type"] == ""
    assert len(sources[0]["text_preview"]) == 243
    assert sources[0]["text_preview"].endswith("...")


def test_rag_prompt_allows_exact_urls_when_present_for_link_questions():
    messages = build_rag_messages(
        question="Give me Lama Retail Instagram link",
        chat_history=[],
        retrieved_chunks=[
            {
                "chunk_id": "social",
                "text": "Social platform: Instagram\nURL: https://www.instagram.com/lamaretail/",
                "source_url": "https://pk.lamaretail.com/",
                "page_title": "LAMA RETAIL",
                "chunk_type": "social_link",
            }
        ],
    )

    system_content = messages[0]["content"]
    assert "For link questions, use the exact URL from the retrieved context when it is present." in system_content
    assert "Never invent social media, profile, product, or contact links." in system_content
    assert "https://www.instagram.com/lamaretail/" in system_content


def test_rag_prompt_distinguishes_platform_mentions_without_links():
    messages = build_rag_messages(
        question="Does Lama Retail have Instagram?",
        chat_history=[],
        retrieved_chunks=[
            {
                "chunk_id": "mention",
                "text": "LAMA is available on Facebook, Instagram and YouTube.",
                "source_url": "https://pk.lamaretail.com/pages/faqs",
                "page_title": "FAQ",
                "chunk_type": "faq",
            }
        ],
    )

    system_content = messages[0]["content"]
    assert "If a platform is mentioned but no URL is present" in system_content
    assert "Do not invent details" in system_content
