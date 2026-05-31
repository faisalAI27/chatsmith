from backend.app.services.chunker import build_chunks_from_knowledge, summarize_chunks


def _sample_v2_knowledge():
    return {
        "metadata": {
            "website_id": "example12345",
            "url": "https://example.com",
            "normalized_url": "https://example.com",
            "name": "Example",
            "scraping_version": "v2",
            "pages_scraped": 1,
        },
        "pages": [
            {
                "page_url": "https://example.com/about",
                "normalized_page_url": "https://example.com/about",
                "page_type": "about",
                "title": "About Example",
                "description": "Example builds useful tools.",
                "extraction_method": "static",
                "headings": [{"level": 1, "text": "About Example"}],
                "sections": [
                    {
                        "heading": "Services",
                        "level": 2,
                        "content": "We build workflow automation tools for small teams.",
                    }
                ],
                "paragraphs": [
                    "We build workflow automation tools for small teams.",
                    "Our support team helps customers launch projects.",
                ],
                "lists": [{"type": "ul", "items": ["Automation", "Support"]}],
                "faqs": [
                    {
                        "question": "What does Example do?",
                        "answer": "Example builds workflow automation tools.",
                    }
                ],
                "tables": [{"headers": ["Plan", "Price"], "rows": [["Basic", "$10"]]}],
                "images": [
                    {
                        "src": "/team.jpg",
                        "absolute_url": "https://example.com/team.jpg",
                        "alt": "Company leadership team",
                        "title": "Leadership",
                        "figcaption": "Our executive team",
                        "nearby_text": "Meet the team behind Example.",
                        "image_type": "team",
                    }
                ],
                "links": [{"text": "Contact", "url": "https://example.com/contact"}],
                "structured_data": [{"@type": "Organization", "name": "Example"}],
                "quality": {"image_count": 1, "link_count": 1},
                "content": "About Example. We build workflow automation tools for small teams.",
            }
        ],
        "primary_content": {"source": "website_scraping", "reliability": "high", "pages": []},
        "secondary_content": {"source": "web_search", "reliability": "medium", "searches": []},
    }


def test_build_chunks_from_v2_knowledge_includes_expected_chunk_types():
    chunks = build_chunks_from_knowledge(_sample_v2_knowledge())
    chunk_types = {chunk["chunk_type"] for chunk in chunks}

    assert {
        "page_summary",
        "section",
        "paragraph_group",
        "faq",
        "table",
        "image_context",
        "structured_data",
    }.issubset(chunk_types)


def test_chunks_preserve_required_source_metadata():
    chunks = build_chunks_from_knowledge(_sample_v2_knowledge())
    section = next(chunk for chunk in chunks if chunk["chunk_type"] == "section")

    assert section["website_id"] == "example12345"
    assert section["source_url"] == "https://example.com/about"
    assert section["page_title"] == "About Example"
    assert section["page_type"] == "about"
    assert section["metadata"]["website_url"] == "https://example.com"
    assert section["metadata"]["normalized_website_url"] == "https://example.com"
    assert section["metadata"]["normalized_url"] == "https://example.com/about"
    assert section["metadata"]["page_url"] == "https://example.com/about"
    assert section["metadata"]["source_type"] == "primary"
    assert section["metadata"]["reliability"] == "high"
    assert section["metadata"]["extraction_method"] == "static"
    assert section["metadata"]["heading"] == "Services"
    assert section["metadata"]["image_count"] == 1
    assert section["metadata"]["table_count"] == 1


def test_faq_table_image_and_structured_data_chunks_are_readable():
    chunks = build_chunks_from_knowledge(_sample_v2_knowledge())

    faq = next(chunk for chunk in chunks if chunk["chunk_type"] == "faq")
    table = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
    image = next(chunk for chunk in chunks if chunk["chunk_type"] == "image_context")
    structured_data = next(chunk for chunk in chunks if chunk["chunk_type"] == "structured_data")

    assert "Question: What does Example do?" in faq["text"]
    assert "Answer: Example builds workflow automation tools." in faq["text"]
    assert "Columns: Plan | Price" in table["text"]
    assert "Row: Basic | $10" in table["text"]
    assert "Alt text: Company leadership team" in image["text"]
    assert image["metadata"]["image_url"] == "https://example.com/team.jpg"
    assert image["metadata"]["image_type"] == "team"
    assert '"@type": "Organization"' in structured_data["text"]


def test_build_chunks_supports_legacy_primary_content_pages():
    legacy_knowledge = {
        "metadata": {
            "url": "https://legacy.example.com",
            "normalized_url": "https://legacy.example.com",
            "website_id": "legacy12345",
        },
        "primary_content": {
            "source": "website_scraping",
            "reliability": "high",
            "pages": [
                {
                    "title": "Legacy Home",
                    "description": "Legacy description.",
                    "sections": [{"heading": "Intro", "content": "Legacy page content for chunks."}],
                    "content": "Legacy page content for chunks.",
                    "url": "https://legacy.example.com",
                    "page_type": "homepage",
                }
            ],
        },
    }

    chunks = build_chunks_from_knowledge(legacy_knowledge)

    assert chunks
    assert chunks[0]["website_id"] == "legacy12345"
    assert chunks[0]["source_url"] == "https://legacy.example.com"
    assert chunks[0]["metadata"]["normalized_url"] == "https://legacy.example.com"
    assert any(chunk["chunk_type"] == "section" for chunk in chunks)


def test_empty_and_noisy_content_is_skipped():
    knowledge = {
        "metadata": {"url": "https://empty.example.com", "normalized_url": "https://empty.example.com"},
        "pages": [
            {
                "page_url": "https://empty.example.com",
                "normalized_page_url": "https://empty.example.com",
                "title": "",
                "description": "",
                "sections": [{"heading": "", "content": ""}],
                "paragraphs": ["   "],
                "faqs": [{"question": "", "answer": ""}],
                "tables": [{"headers": [], "rows": []}],
                "images": [{"alt": "", "title": "", "figcaption": "", "nearby_text": ""}],
                "structured_data": [],
                "content": "",
            }
        ],
    }

    assert build_chunks_from_knowledge(knowledge) == []


def test_duplicate_chunk_text_is_removed():
    knowledge = _sample_v2_knowledge()
    duplicate_text = "This duplicated section text should appear once in chunks."
    knowledge["pages"][0]["sections"] = [
        {"heading": "One", "content": duplicate_text},
        {"heading": "Two", "content": duplicate_text},
    ]
    knowledge["pages"][0]["paragraphs"] = [duplicate_text]
    knowledge["pages"][0]["description"] = ""

    chunks = build_chunks_from_knowledge(knowledge)
    duplicate_chunks = [chunk for chunk in chunks if duplicate_text in chunk["text"]]

    assert len(duplicate_chunks) == 1


def test_chunk_ids_are_stable_for_same_input():
    first = build_chunks_from_knowledge(_sample_v2_knowledge())
    second = build_chunks_from_knowledge(_sample_v2_knowledge())

    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]


def test_long_text_splits_with_overlap():
    words = [f"word{i}" for i in range(1300)]
    knowledge = _sample_v2_knowledge()
    knowledge["pages"][0]["paragraphs"] = [" ".join(words)]
    knowledge["pages"][0]["sections"] = []
    knowledge["pages"][0]["faqs"] = []
    knowledge["pages"][0]["tables"] = []
    knowledge["pages"][0]["images"] = []
    knowledge["pages"][0]["structured_data"] = []

    paragraph_chunks = [
        chunk
        for chunk in build_chunks_from_knowledge(knowledge)
        if chunk["chunk_type"] == "paragraph_group"
    ]

    assert len(paragraph_chunks) == 2
    assert "word880" in paragraph_chunks[1]["text"]
    assert "word999" in paragraph_chunks[1]["text"]


def test_summarize_chunks_counts_types_and_pages():
    chunks = build_chunks_from_knowledge(_sample_v2_knowledge())
    summary = summarize_chunks(chunks)

    assert summary["total_chunks"] == len(chunks)
    assert summary["by_type"]["faq"] == 1
    assert summary["pages_covered"] == 1


def _social_knowledge():
    knowledge = _sample_v2_knowledge()
    knowledge["pages"][0]["page_url"] = "https://pk.lamaretail.com/"
    knowledge["pages"][0]["normalized_page_url"] = "https://pk.lamaretail.com"
    knowledge["pages"][0]["title"] = "LAMA RETAIL - Lama Retail"
    knowledge["pages"][0]["description"] = "Lama Retail home."
    knowledge["pages"][0]["sections"] = [{"heading": "Footer", "content": "Instagram Facebook YouTube TikTok"}]
    knowledge["pages"][0]["paragraphs"] = ["Follow Lama Retail on social media."]
    knowledge["pages"][0]["links"] = [
        {"text": "Instagram", "url": "https://www.instagram.com/lamaretail/"},
        {"text": "Facebook", "url": "https://www.facebook.com/Lama-105719344378446"},
        {"text": "YouTube", "url": "https://www.youtube.com/channel/UCy2wnvTRxfUshTztbdDds2g"},
        {"text": "TikTok", "url": "https://www.tiktok.com/@lamaretail"},
    ]
    knowledge["pages"][0]["structured_data"] = [
        {
            "@type": "Organization",
            "name": "Lama Retail",
            "sameAs": [
                "https://www.instagram.com/lamaretail/",
                "https://www.facebook.com/Lama-105719344378446",
            ],
        }
    ]
    return knowledge


def test_page_social_links_create_social_link_chunks():
    chunks = build_chunks_from_knowledge(_social_knowledge())
    social_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "social_link"]
    platforms = {chunk["metadata"]["social_platform"] for chunk in social_chunks}

    assert {"instagram", "facebook", "youtube", "tiktok"}.issubset(platforms)
    assert any("URL: https://www.instagram.com/lamaretail/" in chunk["text"] for chunk in social_chunks)
    assert any("Social platform: TikTok" in chunk["text"] for chunk in social_chunks)


def test_structured_data_same_as_creates_social_link_chunks():
    knowledge = _sample_v2_knowledge()
    knowledge["pages"][0]["links"] = []
    knowledge["pages"][0]["structured_data"] = [
        {
            "@type": "Organization",
            "sameAs": [
                "https://www.instagram.com/lamaretail/",
                "https://www.youtube.com/channel/UCy2wnvTRxfUshTztbdDds2g",
            ],
        }
    ]

    social_chunks = [
        chunk for chunk in build_chunks_from_knowledge(knowledge) if chunk["chunk_type"] == "social_link"
    ]

    assert {chunk["metadata"]["social_platform"] for chunk in social_chunks} == {"instagram", "youtube"}
    assert all(chunk["metadata"]["source_kind"] == "structured_data" for chunk in social_chunks)


def test_duplicate_social_links_are_deduplicated_by_platform_url_and_source():
    knowledge = _social_knowledge()
    knowledge["pages"][0]["links"].append(
        {"text": "Instagram", "url": "https://www.instagram.com/lamaretail/"}
    )

    instagram_chunks = [
        chunk
        for chunk in build_chunks_from_knowledge(knowledge)
        if chunk["chunk_type"] == "social_link"
        and chunk["metadata"]["social_platform"] == "instagram"
        and chunk["metadata"]["social_url"] == "https://www.instagram.com/lamaretail/"
    ]

    assert len(instagram_chunks) == 1


def test_tiktok_link_from_page_links_is_preserved_as_social_chunk():
    chunks = build_chunks_from_knowledge(_social_knowledge())
    tiktok_chunks = [
        chunk
        for chunk in chunks
        if chunk["chunk_type"] == "social_link" and "https://www.tiktok.com/@lamaretail" in chunk["text"]
    ]

    assert len(tiktok_chunks) == 1
    assert tiktok_chunks[0]["metadata"]["social_platform"] == "tiktok"
