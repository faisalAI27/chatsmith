from backend.app.services.url_discovery import (
    classify_crawl_intent,
    classify_page_type,
    discover_candidate_urls,
    discover_candidate_urls_with_stats,
    extract_sitemap_urls_from_robots,
    is_utility_or_meta_url,
    is_valid_crawl_url,
    normalize_discovered_url,
    parse_sitemap_xml,
    score_url_priority,
    select_candidate_urls,
)


WIKIPEDIA_MARCUS_URL = "https://en.wikipedia.org/wiki/Marcus_Aurelius"


def test_extract_sitemap_urls_from_robots_supports_multiple_entries():
    robots = """
    User-agent: *
    Disallow: /admin
    Sitemap: https://example.com/sitemap.xml
    Sitemap: https://example.com/blog-sitemap.xml
    """

    assert extract_sitemap_urls_from_robots(robots) == [
        "https://example.com/sitemap.xml",
        "https://example.com/blog-sitemap.xml",
    ]


def test_parse_sitemap_xml_urlset():
    xml = """
    <?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/about</loc></url>
      <url><loc>https://example.com/pricing</loc></url>
    </urlset>
    """

    assert parse_sitemap_xml(xml) == {
        "urls": ["https://example.com/about", "https://example.com/pricing"],
        "sitemaps": [],
    }


def test_parse_sitemap_xml_index():
    xml = """
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.com/pages.xml</loc></sitemap>
      <sitemap><loc>https://example.com/blog.xml</loc></sitemap>
    </sitemapindex>
    """

    assert parse_sitemap_xml(xml) == {
        "urls": [],
        "sitemaps": ["https://example.com/pages.xml", "https://example.com/blog.xml"],
    }


def test_normalize_discovered_url_removes_tracking_fragment_and_default_ports():
    assert normalize_discovered_url("https://Example.com:443/about/?utm_source=x&b=2#a") == (
        "https://example.com/about?b=2"
    )
    assert normalize_discovered_url("https://example.com/docs?utm_term=x&utm_content=y&a=1") == (
        "https://example.com/docs?a=1"
    )
    assert normalize_discovered_url("http://Example.com:80/") == "http://example.com"


def test_is_valid_crawl_url_filters_noise_and_keeps_priority_deep_paths():
    base_domain = "example.com"

    assert is_valid_crawl_url("https://example.com/about", base_domain)
    assert is_valid_crawl_url("https://www.example.com/docs/guides/install/setup", base_domain)
    assert not is_valid_crawl_url("https://other.com/about", base_domain)
    assert not is_valid_crawl_url("https://example.com/login", base_domain)
    assert not is_valid_crawl_url("https://example.com/cart/checkout", base_domain)
    assert not is_valid_crawl_url("https://example.com/brochure.pdf", base_domain)
    assert not is_valid_crawl_url("mailto:hello@example.com", base_domain)
    assert not is_valid_crawl_url("javascript:void(0)", base_domain)
    assert not is_valid_crawl_url("https://example.com/a/b/c/d/e", base_domain)


def test_score_url_priority_orders_useful_pages_before_low_value_pages():
    assert score_url_priority("https://example.com/about") > score_url_priority(
        "https://example.com/blog"
    )
    assert score_url_priority("https://example.com/case-studies/customer-a") > score_url_priority(
        "https://example.com/privacy"
    )
    assert score_url_priority("https://example.com/privacy") < score_url_priority(
        "https://example.com/random"
    )
    assert classify_page_type("https://example.com/docs/getting-started") == "docs"


def test_root_query_url_is_not_classified_as_homepage():
    assert classify_page_type("https://example.com") == "homepage"
    assert classify_page_type("https://example.com/?page=2") == "query"
    assert classify_page_type("https://example.com/?search=nasa") == "search_result"


def test_classify_crawl_intent_for_homepage_and_specific_page_urls():
    assert classify_crawl_intent("https://www.pakwheels.com/") == "homepage_site_crawl"
    assert classify_crawl_intent("https://example.com") == "homepage_site_crawl"
    assert classify_crawl_intent(WIKIPEDIA_MARCUS_URL) == "specific_page_crawl"
    assert classify_crawl_intent("https://example.com/blog/article-name") == "specific_page_crawl"


def test_search_query_urls_are_filtered_from_crawl_candidates():
    base_domain = "example.com"

    assert not is_valid_crawl_url("https://example.com/?q=test", base_domain)
    assert not is_valid_crawl_url("https://example.com/?search=test", base_domain)
    assert not is_valid_crawl_url("https://example.com/?s=test", base_domain)
    assert not is_valid_crawl_url("https://example.com/search?q=test", base_domain)
    assert not is_valid_crawl_url("https://example.com/results?query=test", base_domain)
    assert not is_valid_crawl_url("https://example.com/find?keyword=test", base_domain)
    assert not is_valid_crawl_url("https://example.com/lookup?term=test", base_domain)


def test_query_and_search_urls_do_not_outrank_useful_static_pages():
    assert score_url_priority("https://example.com/search?q=moon") < score_url_priority(
        "https://example.com/about"
    )
    assert score_url_priority("https://example.com/?search=nasa") < score_url_priority(
        "https://example.com/services"
    )
    assert score_url_priority("https://example.com/blog?category=news") < score_url_priority(
        "https://example.com/blog"
    )


def test_useful_static_pages_remain_valid_after_query_filtering():
    base_domain = "example.com"
    useful_pages = [
        "https://example.com/about",
        "https://example.com/services",
        "https://example.com/products",
        "https://example.com/pricing",
        "https://example.com/docs",
        "https://example.com/faq",
        "https://example.com/contact",
        "https://example.com/blog",
    ]

    assert all(is_valid_crawl_url(url, base_domain) for url in useful_pages)


def test_select_candidate_urls_deduplicates_filters_ranks_and_limits():
    selected = select_candidate_urls(
        base_url="https://example.com",
        urls=[
            "https://example.com/about?utm_source=x",
            "https://example.com/about#team",
            "https://example.com/login",
            "https://external.com/contact",
            "https://example.com/products",
            "https://example.com/pricing",
            "https://example.com/privacy",
            "https://example.com/download.pdf",
        ],
        max_pages=4,
    )

    assert selected == [
        "https://example.com/about",
        "https://example.com/pricing",
        "https://example.com/products",
        "https://example.com/privacy",
    ]


def test_select_candidate_urls_skips_nasa_style_search_result_urls():
    selected = select_candidate_urls(
        base_url="https://www.nasa.gov",
        urls=[
            "https://www.nasa.gov?search=Artemis",
            "https://www.nasa.gov/?search=Climate+Change",
            "https://www.nasa.gov/search?query=Mars",
            "https://www.nasa.gov/about",
            "https://www.nasa.gov/contact",
            "https://www.nasa.gov/news",
        ],
        max_pages=4,
    )

    assert selected == [
        "https://www.nasa.gov/about",
        "https://www.nasa.gov/contact",
        "https://www.nasa.gov/news",
    ]


def test_specific_page_selection_keeps_original_url_first():
    selected = select_candidate_urls(
        base_url=WIKIPEDIA_MARCUS_URL,
        urls=[
            "https://en.wikipedia.org/wiki/Stoicism",
            "https://en.wikipedia.org/wiki/Help:Contents",
            "https://en.wikipedia.org/wiki/Antoninus_Pius",
        ],
        max_pages=4,
    )

    assert selected[0] == WIKIPEDIA_MARCUS_URL
    assert "https://en.wikipedia.org/wiki/Help:Contents" not in selected


def test_wikipedia_meta_namespaces_are_utility_for_normal_article_crawl():
    utility_urls = [
        "https://en.wikipedia.org/wiki/Help:Authority_control",
        "https://en.wikipedia.org/wiki/Category:Roman_emperors",
        "https://en.wikipedia.org/wiki/File:Marcus_Aurelius.jpg",
        "https://en.wikipedia.org/wiki/Talk:Marcus_Aurelius",
        "https://en.wikipedia.org/wiki/Special:Random",
        "https://en.wikipedia.org/wiki/Wikipedia:About",
        "https://en.wikipedia.org/wiki/Template:Roman_emperors",
        "https://en.wikipedia.org/wiki/Portal:Ancient_Rome",
    ]

    assert all(
        is_utility_or_meta_url(url, WIKIPEDIA_MARCUS_URL, "specific_page_crawl")
        for url in utility_urls
    )


def test_specific_page_selection_filters_wikipedia_help_and_category_pages():
    selected = select_candidate_urls(
        base_url=WIKIPEDIA_MARCUS_URL,
        urls=[
            "https://en.wikipedia.org/wiki/Help:Authority_control",
            "https://en.wikipedia.org/wiki/Help:Category",
            "https://en.wikipedia.org/wiki/Help:IPA/English",
            "https://en.wikipedia.org/wiki/Wikipedia:Contact_us",
            "https://en.wikipedia.org/wiki/Category:Marcus_Aurelius",
            "https://en.wikipedia.org/wiki/Stoicism",
            "https://en.wikipedia.org/wiki/Roman_emperor",
        ],
        max_pages=5,
    )

    assert selected == [
        WIKIPEDIA_MARCUS_URL,
        "https://en.wikipedia.org/wiki/Roman_emperor",
        "https://en.wikipedia.org/wiki/Stoicism",
    ]


def test_specific_page_selection_does_not_choose_site_utility_pages_over_articles():
    selected = select_candidate_urls(
        base_url="https://example.com/blog/marcus-aurelius",
        urls=[
            "https://example.com/contact",
            "https://example.com/about",
            "https://example.com/help",
            "https://example.com/privacy",
            "https://example.com/blog/marcus-aurelius-quotes",
            "https://example.com/blog/stoicism",
        ],
        max_pages=5,
    )

    assert selected == [
        "https://example.com/blog/marcus-aurelius",
        "https://example.com/blog/marcus-aurelius-quotes",
        "https://example.com/blog/stoicism",
    ]


def test_discover_candidate_urls_combines_homepage_links_and_sitemap_urls():
    html = """
    <a href="/contact">Contact</a>
    <a href="/account">Account</a>
    <a href="https://example.com/services?utm_campaign=spring">Services</a>
    <a href="https://other.com/about">Other</a>
    """

    selected = discover_candidate_urls(
        homepage_html=html,
        base_url="https://example.com",
        sitemap_urls=[
            "https://example.com/faq",
            "https://example.com/blog",
            "https://example.com/file.zip",
        ],
        max_pages=4,
    )

    assert selected == [
        "https://example.com/contact",
        "https://example.com/faq",
        "https://example.com/services",
        "https://example.com/blog",
    ]


def test_homepage_crawl_still_keeps_business_utility_pages():
    selected = discover_candidate_urls(
        homepage_html="""
        <a href="/about-us">About</a>
        <a href="/used-cars">Used Cars</a>
        <a href="/new-cars">New Cars</a>
        <a href="/contact-us">Contact</a>
        """,
        base_url="https://www.pakwheels.com/",
        sitemap_urls=[],
        max_pages=4,
    )

    assert "https://www.pakwheels.com/about-us" in selected
    assert "https://www.pakwheels.com/contact-us" in selected
    assert "https://www.pakwheels.com/used-cars" in selected


def test_specific_discovery_stats_count_filtered_utility_urls():
    stats = discover_candidate_urls_with_stats(
        homepage_html="""
        <a href="/wiki/Help:Contents">Help</a>
        <a href="/wiki/Wikipedia:About">About Wikipedia</a>
        <a href="/wiki/Stoicism">Stoicism</a>
        """,
        base_url=WIKIPEDIA_MARCUS_URL,
        sitemap_urls=[],
        max_pages=5,
    )

    assert stats["crawl_intent"] == "specific_page_crawl"
    assert stats["selected_urls"][0] == WIKIPEDIA_MARCUS_URL
    assert stats["utility_filtered_count"] == 2
    assert "https://en.wikipedia.org/wiki/Stoicism" in stats["selected_urls"]
