from backend.app.services.rag_quality import (
    build_retrieval_debug_summary,
    calculate_context_coverage,
    detect_low_quality_retrieval,
    is_context_sufficient,
    normalize_query,
    validate_sources,
)


def _good_chunks():
    return [
        {
            "text": "This website explains car inspection services, pricing, and booking details.",
            "source_url": "https://example.com/inspection",
            "page_title": "Car Inspection",
            "chunk_type": "section",
            "score": 0.91,
        },
        {
            "text": "Customers can contact the inspection team through the support form.",
            "source_url": "https://example.com/contact",
            "page_title": "Contact",
            "chunk_type": "faq",
            "score": 0.82,
        },
    ]


def test_normalize_query_collapses_whitespace():
    assert normalize_query("  Does   this site offer inspection?  ") == "Does this site offer inspection?"


def test_context_sufficiency_with_good_and_missing_chunks():
    assert is_context_sufficient(_good_chunks()) is True
    assert is_context_sufficient([]) is False
    assert is_context_sufficient([{"text": "too short"}]) is False


def test_context_coverage_counts_sources_chunk_types_and_scores():
    coverage = calculate_context_coverage(_good_chunks())

    assert coverage["chunks_retrieved"] == 2
    assert coverage["usable_chunks"] == 2
    assert coverage["source_count"] == 2
    assert coverage["chunk_types"] == {"faq": 1, "section": 1}
    assert coverage["missing_source_url_count"] == 0
    assert coverage["average_score"] == 0.865


def test_low_quality_retrieval_warnings_for_empty_text_and_missing_metadata():
    warnings = detect_low_quality_retrieval(
        [
            {"text": "", "source_url": "", "page_title": "", "chunk_type": "section"},
            {"text": "tiny", "source_url": "", "page_title": "", "chunk_type": "section"},
        ]
    )

    assert "Retrieved chunks did not contain enough readable website text." in warnings
    assert "Some retrieved chunks are missing source URLs." in warnings
    assert "Some retrieved chunks are missing page titles." in warnings


def test_low_quality_retrieval_warns_for_single_weak_source():
    warnings = detect_low_quality_retrieval(
        [
            {
                "text": "This chunk has enough readable text but no usable source URL metadata.",
                "source_url": "",
                "page_title": "Untitled",
                "chunk_type": "section",
            },
            {
                "text": "This second chunk also has readable text but still weak source metadata.",
                "source_url": "",
                "page_title": "Untitled",
                "chunk_type": "faq",
            },
        ]
    )

    assert "Retrieved context is concentrated in one weak source, so the answer may be incomplete." in warnings


def test_retrieval_debug_summary_includes_warnings():
    debug = build_retrieval_debug_summary([])

    assert debug["chunks_retrieved"] == 0
    assert debug["source_count"] == 0
    assert debug["warnings"] == ["No relevant indexed chunks were found for this website."]


def test_validate_sources_detects_missing_fields_duplicates_and_long_preview():
    long_preview = "x" * 300
    warnings = validate_sources(
        [
            {
                "source_url": "https://example.com",
                "page_title": "Home",
                "chunk_type": "section",
                "text_preview": long_preview,
            },
            {
                "source_url": "https://example.com",
                "page_title": "Home",
                "chunk_type": "section",
                "text_preview": "duplicate",
            },
            {"source_url": "", "page_title": "", "chunk_type": "", "text_preview": ""},
        ]
    )

    assert "Duplicate source records were returned." in warnings
    assert "Source 1 text_preview is too long." in warnings
    assert "Source 3 is missing source_url." in warnings
    assert "Source 3 is missing page_title." in warnings
    assert "Source 3 is missing text_preview." in warnings
