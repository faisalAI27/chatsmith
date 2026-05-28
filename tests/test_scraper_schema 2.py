import hashlib

from backend.app.services.scraper_schema import (
    convert_v2_page_to_legacy_page,
    create_page_record,
    ensure_v2_knowledge_shape,
    make_website_id,
    normalize_url_for_cache,
)


def test_normalize_url_for_cache_removes_tracking_query_and_fragment():
    assert normalize_url_for_cache("https://Example.com/") == "https://example.com"
    assert normalize_url_for_cache("https://example.com/#section") == "https://example.com"
    assert normalize_url_for_cache("https://example.com/?utm_source=x") == "https://example.com"
    assert (
        normalize_url_for_cache("https://Example.com/path/?utm_medium=x&b=2&a=1&fbclid=abc#top")
        == "https://example.com/path?a=1&b=2"
    )


def test_make_website_id_is_stable_from_normalized_url():
    normalized_url = normalize_url_for_cache("https://Example.com/?utm_campaign=spring")
    expected = hashlib.md5(normalized_url.encode("utf-8")).hexdigest()[:12]

    assert make_website_id(normalized_url) == expected
    assert make_website_id(normalized_url) == make_website_id("https://example.com")


def test_create_page_record_defaults_and_quality_counts():
    page = create_page_record(
        page_url="https://Example.com/about/",
        page_type="about",
        title="About",
        description="About page",
        sections=[{"heading": "Intro", "content": "Hello"}],
        paragraphs=["Hello"],
        images=[{"src": "/team.jpg"}],
        links=[{"href": "/contact"}],
        content="Hello world",
    )

    assert page["page_url"] == "https://Example.com/about/"
    assert page["normalized_page_url"] == "https://example.com/about"
    assert page["extraction_method"] == "static"
    assert page["metadata"] == {"canonical": "", "open_graph": {}, "twitter": {}}
    assert page["quality"]["content_length"] == len("Hello world")
    assert page["quality"]["image_count"] == 1
    assert page["quality"]["link_count"] == 1
    assert page["sections"][0]["heading"] == "Intro"


def test_convert_v2_page_to_legacy_page():
    page = create_page_record(
        page_url="https://example.com/faq",
        page_type="faq",
        title="FAQ",
        description="Questions",
        sections=[{"heading": "Q", "content": "A"}],
        content="Q A",
    )

    legacy = convert_v2_page_to_legacy_page(page)

    assert legacy == {
        "title": "FAQ",
        "description": "Questions",
        "sections": [{"heading": "Q", "content": "A"}],
        "content": "Q A",
        "url": "https://example.com/faq",
        "page_type": "faq",
    }


def test_ensure_v2_knowledge_shape_backfills_old_json_and_preserves_legacy_pages():
    old_knowledge = {
        "metadata": {
            "url": "https://Example.com/?utm_source=x",
            "name": "Example",
            "pages_scraped": 1,
        },
        "primary_content": {
            "source": "website_scraping",
            "reliability": "high",
            "pages": [
                {
                    "title": "Home",
                    "description": "Welcome",
                    "sections": [{"heading": "Intro", "content": "Hello"}],
                    "content": "Hello",
                    "url": "https://Example.com/",
                    "page_type": "homepage",
                }
            ],
        },
    }

    shaped = ensure_v2_knowledge_shape(old_knowledge)

    assert shaped["metadata"]["normalized_url"] == "https://example.com"
    assert shaped["metadata"]["website_id"] == make_website_id("https://example.com")
    assert shaped["metadata"]["scraping_version"] == "v2"
    assert shaped["pages"][0]["page_url"] == "https://Example.com/"
    assert shaped["pages"][0]["normalized_page_url"] == "https://example.com"
    assert shaped["primary_content"]["pages"][0]["title"] == "Home"
    assert shaped["secondary_content"] == {
        "source": "web_search",
        "reliability": "medium",
        "searches": [],
    }


def test_ensure_v2_knowledge_shape_rebuilds_legacy_pages_from_v2_pages():
    page = create_page_record(
        page_url="https://example.com/products",
        page_type="products",
        title="Products",
        content="Product details",
    )

    shaped = ensure_v2_knowledge_shape(
        {
            "metadata": {"url": "https://example.com"},
            "pages": [page],
        }
    )

    assert shaped["primary_content"]["pages"] == [
        {
            "title": "Products",
            "description": "",
            "sections": [],
            "content": "Product details",
            "url": "https://example.com/products",
            "page_type": "products",
        }
    ]
