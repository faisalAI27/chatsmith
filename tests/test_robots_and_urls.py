from backend.app.services.robots import parse_robots_txt
from backend.app.services.scraper_schema import normalize_url_for_cache
from backend.app.services.text_fingerprint import content_hash, hamming_distance, simhash
from backend.app.services.url_discovery import (
    is_valid_crawl_url,
    score_url_priority,
    url_key,
)


ROBOTS = """
User-agent: *
Disallow: /private/
Disallow: /*?filter=
Disallow: /*.json$
Allow: /private/press/
Crawl-delay: 2
Sitemap: https://example.com/sitemap.xml

User-agent: Googlebot
Disallow: /
"""


def test_robots_longest_match_allow_and_wildcards():
    rules = parse_robots_txt(ROBOTS)
    assert rules.can_fetch("https://example.com/about")
    assert not rules.can_fetch("https://example.com/private/report")
    assert rules.can_fetch("https://example.com/private/press/release-1")
    assert not rules.can_fetch("https://example.com/shop?filter=red")
    assert not rules.can_fetch("https://example.com/data/feed.json")
    assert rules.can_fetch("https://example.com/data/feed.json?x=1")
    assert rules.crawl_delay == 2
    assert rules.sitemaps == ["https://example.com/sitemap.xml"]


def test_robots_paths_are_case_sensitive_and_specific_agent_wins():
    rules = parse_robots_txt("User-agent: *\nDisallow: /Admin\n")
    assert not rules.can_fetch("https://example.com/Admin/panel")
    assert rules.can_fetch("https://example.com/admin-guide")  # old parser lowercased and blocked this

    specific = parse_robots_txt("User-agent: *\nDisallow: /\n\nUser-agent: ChatSmith\nAllow: /\n")
    assert specific.can_fetch("https://example.com/anything")


def test_robots_empty_disallow_allows_everything():
    assert parse_robots_txt("User-agent: *\nDisallow:\n").can_fetch("https://example.com/x")


def test_skip_keywords_match_whole_tokens_only():
    assert is_valid_crawl_url("https://x.com/services/accounting", "x.com")
    assert is_valid_crawl_url("https://x.com/cartography", "x.com")
    assert not is_valid_crawl_url("https://x.com/my-account", "x.com")
    assert not is_valid_crawl_url("https://x.com/cart", "x.com")
    assert not is_valid_crawl_url("https://x.com/sign-in", "x.com")


def test_priority_keywords_match_whole_tokens_only():
    assert score_url_priority("https://x.com/heritage-sites") == score_url_priority("https://x.com/campus-sites")
    assert score_url_priority("https://x.com/about-us") > score_url_priority("https://x.com/random")


def test_documents_and_subdomains_are_opt_in():
    assert not is_valid_crawl_url("https://x.com/prospectus.pdf", "x.com")
    assert is_valid_crawl_url("https://x.com/prospectus.pdf", "x.com", allow_documents=True)
    assert not is_valid_crawl_url("https://blog.x.com/post", "x.com")
    assert is_valid_crawl_url("https://blog.x.com/post", "x.com", include_subdomains=True)


def test_ad_click_parameters_do_not_change_website_identity():
    clean = normalize_url_for_cache("https://pk.lamaretail.com/")
    with_ads = normalize_url_for_cache(
        "https://pk.lamaretail.com/?gad_source=1&gad_campaignid=1680&gbraid=0AAAA&gclid=Cj0K&srsltid=AfmB"
    )
    assert with_ads == clean


def test_url_key_merges_index_files_scheme_and_www():
    assert url_key("http://www.example.com/index.html") == url_key("https://example.com/")
    assert url_key("https://example.com/docs/index.php") == url_key("https://example.com/docs")
    assert url_key("https://example.com/a?page=2") != url_key("https://example.com/a?page=3")


def test_simhash_flags_near_duplicates_but_not_different_pages():
    base = " ".join(f"word{i}" for i in range(400))
    variant = base.replace("word10 ", "colour-black ").replace("word200 ", "sku-123 ")
    other = " ".join(f"other{i}" for i in range(400))
    assert hamming_distance(simhash(base), simhash(variant)) <= 3
    assert hamming_distance(simhash(base), simhash(other)) > 10
    assert content_hash("Hello,   World") == content_hash("hello world")
