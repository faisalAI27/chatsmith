from backend.app.services.retrieval_ranker import (
    calculate_rerank_score,
    classify_query_intent,
    extract_query_terms,
    rerank_chunks,
    score_intent_match,
    score_keyword_overlap,
)


def _chunk(
    chunk_id,
    text,
    chunk_type="section",
    score=0.5,
    page_title="",
    source_url="",
):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "chunk_type": chunk_type,
        "score": score,
        "page_title": page_title,
        "source_url": source_url,
        "metadata": {"website_id": "lama"},
    }


def test_query_intent_and_terms_for_contact_query():
    assert classify_query_intent("Lama Retail phone number whatsapp customer service") == "contact"
    assert {"lama", "retail", "phone", "number", "whatsapp", "customer", "service"}.issubset(
        set(extract_query_terms("Lama Retail phone number whatsapp customer service"))
    )


def test_contact_detail_section_ranks_above_weak_summary_and_media_chunks():
    chunks = [
        _chunk(
            "summary",
            "Page title: Contact - Lama Retail",
            chunk_type="page_summary",
            score=0.99,
            page_title="Contact - Lama Retail",
            source_url="https://lamaretail.com/pages/contact",
        ),
        _chunk(
            "image",
            "Alt text: Contact page banner",
            chunk_type="image_context",
            score=0.8,
            page_title="Contact - Lama Retail",
            source_url="https://lamaretail.com/pages/contact",
        ),
        _chunk(
            "structured",
            '{"name": "Lama Retail"}',
            chunk_type="structured_data",
            score=0.75,
            page_title="Home",
            source_url="https://lamaretail.com/",
        ),
        _chunk(
            "contact-section",
            "Section: CUSTOMER SERVICE Content: CUSTOMER SERVICE Email | Whatsapp Contact us at 0311-1115262 09:00 AM to 09:00 PM (PST) Monday - Saturday",
            chunk_type="section",
            score=0.2,
            page_title="Contact - Lama Retail",
            source_url="https://lamaretail.com/pages/contact",
        ),
    ]

    reranked, debug = rerank_chunks("contact number of Lama Retail", chunks, top_k=3)

    assert reranked[0]["chunk_id"] == "contact-section"
    assert reranked[0]["rerank_components"]["intent_match"] > 0.8
    assert debug["query_intent"] == "contact"


def test_contact_paragraph_group_ranks_above_homepage_summary():
    chunks = [
        _chunk(
            "home-summary",
            "Page title: Home - Lama Retail",
            chunk_type="page_summary",
            score=0.96,
            page_title="Home - Lama Retail",
            source_url="https://lamaretail.com/",
        ),
        _chunk(
            "contact-paragraph",
            "Contact us at 0311-1115262 for customer service and WhatsApp support.",
            chunk_type="paragraph_group",
            score=0.25,
            page_title="Contact",
            source_url="https://lamaretail.com/pages/contact",
        ),
    ]

    reranked, _debug = rerank_chunks("Lama Retail phone number whatsapp customer service", chunks, top_k=2)

    assert reranked[0]["chunk_id"] == "contact-paragraph"


def test_location_intent_ranks_store_address_chunk_above_summary():
    chunks = [
        _chunk(
            "summary",
            "Page title: Stores - Lama Retail",
            chunk_type="page_summary",
            score=0.95,
            page_title="Stores",
            source_url="https://lamaretail.com/pages/stores",
        ),
        _chunk(
            "locations",
            "Store locations include Lahore, Islamabad, Karachi malls, shop floors and branch addresses.",
            chunk_type="section",
            score=0.35,
            page_title="Store Locations",
            source_url="https://lamaretail.com/pages/store-locations",
        ),
    ]

    reranked, _debug = rerank_chunks("store locations of Lama Retail", chunks, top_k=2)

    assert reranked[0]["chunk_id"] == "locations"


def test_image_context_is_penalized_for_contact_but_allowed_for_image_intent():
    image_chunk = _chunk(
        "image",
        "Image shows the red shirt design and product style.",
        chunk_type="image_context",
        score=0.8,
        page_title="Red Shirt",
        source_url="https://lamaretail.com/products/red-shirt",
    )
    contact_components = calculate_rerank_score("contact number", image_chunk)
    image_components = calculate_rerank_score("show product image design style", image_chunk)

    assert contact_components["penalty"] > 0.5
    assert image_components["chunk_type_weight"] > contact_components["chunk_type_weight"]
    assert image_components["final_score"] > contact_components["final_score"]


def test_short_page_summary_does_not_outrank_detailed_text():
    chunks = [
        _chunk("summary", "Page title: Contact", "page_summary", 1.0, "Contact", "/contact"),
        _chunk(
            "detail",
            "Detailed customer service section with WhatsApp, support hours, and phone number 0311-1115262.",
            "section",
            0.1,
            "Contact",
            "/contact",
        ),
    ]

    reranked, _debug = rerank_chunks("phone number", chunks, top_k=2)

    assert reranked[0]["chunk_id"] == "detail"


def test_product_footer_contact_noise_ranks_below_contact_page_chunk():
    same_contact_text = "Contact us at 0311-1115262 09:00 AM to 09:00 PM Monday - Saturday."
    chunks = [
        _chunk(
            "product-footer",
            same_contact_text,
            chunk_type="section",
            score=0.7,
            page_title="Black Pants",
            source_url="https://lamaretail.com/products/black-pants",
        ),
        _chunk(
            "contact-page",
            same_contact_text,
            chunk_type="section",
            score=0.4,
            page_title="Contact",
            source_url="https://lamaretail.com/pages/contact",
        ),
    ]

    reranked, _debug = rerank_chunks("contact number of Lama Retail", chunks, top_k=2)

    assert reranked[0]["chunk_id"] == "contact-page"


def test_source_url_and_title_boost_contact_query():
    contact_chunk = _chunk(
        "contact",
        "Customer service information and support details.",
        chunk_type="section",
        score=0.3,
        page_title="Contact",
        source_url="https://lamaretail.com/pages/contact",
    )

    components = calculate_rerank_score("how can I contact Lama Retail", contact_chunk)

    assert components["source_relevance"] > 0.5


def test_missing_metadata_does_not_crash_ranking():
    reranked, debug = rerank_chunks(
        "contact number",
        [{"chunk_id": "minimal", "text": "Contact us at 0311-1115262"}],
        top_k=1,
    )

    assert reranked[0]["chunk_id"] == "minimal"
    assert debug["final_count"] == 1


def test_score_helpers_return_positive_values_for_exact_contact_match():
    chunk = _chunk(
        "contact",
        "Whatsapp Contact us at 0311-1115262",
        chunk_type="section",
        page_title="Contact",
        source_url="/pages/contact",
    )

    assert score_keyword_overlap("contact number whatsapp", chunk) > 0
    assert score_intent_match("contact", chunk) > 0.8


def test_social_link_query_intent_detects_platform_and_link_questions():
    assert classify_query_intent("Give me Lama Retail Instagram link") == "social_link"
    assert classify_query_intent("Does Lama Retail have Instagram?") == "social_link"
    assert classify_query_intent("Lama Retail social media links") == "social_link"


def test_social_link_chunk_with_instagram_url_ranks_above_mention_only_faq():
    chunks = [
        _chunk(
            "faq-mention",
            "LAMA is available on social media platforms such as Facebook, Instagram and Youtube.",
            chunk_type="faq",
            score=0.9,
            page_title="FAQ",
            source_url="https://pk.lamaretail.com/pages/faqs",
        ),
        _chunk(
            "instagram-link",
            "Social platform: Instagram URL: https://www.instagram.com/lamaretail/ Link text: Instagram",
            chunk_type="social_link",
            score=0.2,
            page_title="LAMA RETAIL - Lama Retail",
            source_url="https://pk.lamaretail.com/",
        ),
    ]

    reranked, debug = rerank_chunks("Give me Lama Retail Instagram link", chunks, top_k=2)

    assert debug["query_intent"] == "social_link"
    assert reranked[0]["chunk_id"] == "instagram-link"
    assert reranked[0]["rerank_components"]["intent_match"] > 0.8


def test_structured_data_same_as_instagram_ranks_high_for_social_query():
    chunk = _chunk(
        "same-as",
        '{"@type": "Organization", "sameAs": ["https://www.instagram.com/lamaretail/"]}',
        chunk_type="structured_data",
        score=0.3,
        page_title="LAMA RETAIL",
        source_url="https://pk.lamaretail.com/",
    )

    components = calculate_rerank_score("Does Lama Retail have Instagram?", chunk)

    assert components["query_intent"] == "social_link"
    assert components["intent_match"] >= 0.6
    assert components["chunk_type_weight"] > 0.8


def test_section_with_platform_names_but_no_url_ranks_below_social_link_url():
    chunks = [
        _chunk(
            "platform-names",
            "Sign up and save Instagram Facebook YouTube TikTok",
            chunk_type="section",
            score=0.95,
            page_title="LAMA RETAIL",
            source_url="https://pk.lamaretail.com/",
        ),
        _chunk(
            "instagram-url",
            "Social platform: Instagram URL: https://www.instagram.com/lamaretail/",
            chunk_type="social_link",
            score=0.1,
            page_title="LAMA RETAIL",
            source_url="https://pk.lamaretail.com/",
        ),
    ]

    reranked, _debug = rerank_chunks("Lama Retail Instagram link", chunks, top_k=2)

    assert reranked[0]["chunk_id"] == "instagram-url"


def test_product_footer_chunk_is_penalized_for_social_link_query():
    chunks = [
        _chunk(
            "product-footer",
            "Instagram Facebook YouTube TikTok",
            chunk_type="section",
            score=0.98,
            page_title="Black Pants",
            source_url="https://pk.lamaretail.com/products/black-pants",
        ),
        _chunk(
            "social-link",
            "Social platform: Instagram URL: https://www.instagram.com/lamaretail/",
            chunk_type="social_link",
            score=0.1,
            page_title="LAMA RETAIL",
            source_url="https://pk.lamaretail.com/",
        ),
    ]

    reranked, _debug = rerank_chunks("Instagram profile link", chunks, top_k=2)

    assert reranked[0]["chunk_id"] == "social-link"


def test_social_link_ranking_handles_missing_metadata():
    reranked, debug = rerank_chunks(
        "Instagram link",
        [{"chunk_id": "minimal", "text": "URL: https://www.instagram.com/lamaretail/", "chunk_type": "social_link"}],
        top_k=1,
    )

    assert reranked[0]["chunk_id"] == "minimal"
    assert debug["query_intent"] == "social_link"
