"""End-to-end crawler tests against a local test website (no internet, no browser)."""

import pytest
import pytest_asyncio
from aiohttp import web

from backend.app.core.config import get_settings
# Imported at module level: scrape_pipeline loads .env with override=True on
# import, which must happen before the fixture below forces static mode.
from backend.app.services import scrape_pipeline as sp
from backend.app.services.chunker import build_chunks_from_knowledge
from backend.app.services.crawler import CrawlConfig, SiteCrawler, detect_crawl_scope, is_public_host
from backend.app.services.page_fetcher import extract_page_with_render_mode
from tests.helpers import make_pdf

HEADER = "<header><nav><a href='/'>Home</a> <a href='/about'>About</a> <a href='/shop'>Shop</a></nav></header>"
FOOTER = "<footer><p>Helpline 0311-1115262, Monday to Saturday.</p><p>© 2026 Test Store</p></footer>"


def html_page(title: str, body: str, canonical: str = "") -> str:
    link = f"<link rel='canonical' href='{canonical}'>" if canonical else ""
    return (
        f"<html><head><title>{title} | Test Store</title>{link}</head>"
        f"<body>{HEADER}<main><h1>{title}</h1>{body}</main>{FOOTER}</body></html>"
    )


def product_body(colour: str) -> str:
    description = " ".join(
        f"The relaxed fit trouser uses breathable cotton twill fabric number {i} for everyday comfort."
        for i in range(12)
    )
    return f"<p>Colour: {colour}</p><p>{description}</p><p>Price Rs 3,490</p>"


PAGES = {
    "/": html_page("Welcome", "<p>We sell clothing across Pakistan.</p><a href='/about'>About us</a>"
                   "<a href='/shop'>Shop</a><a href='/private/staff'>Staff</a><a href='https://other.example/x'>Partner</a>"
                   "<a href='/login'>Login</a><a href='/files/size-guide.pdf'>Size guide</a>"),
    "/about": html_page("About", "<p>Founded in 2010 in Lahore with three stores.</p><a href='/about/team'>Team</a>"),
    "/about/team": html_page("Team", "<p>Our founder Ayesha Khan leads design.</p><a href='/about/team/history'>History</a>"),
    "/about/team/history": html_page("History", "<p>The first store opened on Mall Road in 2010.</p>"),
    "/shop": html_page("Shop", "<p>Browse trousers.</p><a href='/shop/trouser-black'>Black</a>"
                       "<a href='/shop/trouser-grey'>Grey</a><a href='/shop/index.html'>Shop index</a>"),
    "/shop/trouser-black": html_page("Relaxed Trouser", product_body("Black")),
    "/shop/trouser-grey": html_page("Relaxed Trouser", product_body("Grey")),
    "/shop/index.html": html_page("Shop", "<p>Browse trousers.</p>"),
    "/print/about": html_page("About", "<p>Founded in 2010 in Lahore with three stores.</p>", canonical="/about"),
    "/sitemap-only": html_page("Returns", "<p>Returns are accepted within 14 days with receipt.</p><a href='/print/about'>Print</a>"),
    "/private/staff": html_page("Staff", "<p>Internal staff page.</p>"),
}
ROBOTS = "User-agent: *\nDisallow: /private/\nSitemap: {base}/sitemap.xml\n"
SITEMAP = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>{base}/sitemap-only</loc></url><url><loc>{base}/about</loc></url></urlset>"""


@pytest.fixture(autouse=True)
def static_mode(monkeypatch):
    monkeypatch.setenv("SCRAPER_RENDER_MODE", "static")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def site():
    app = web.Application()
    state = {"base": ""}

    async def handler(request: web.Request):
        path = request.path
        if path == "/robots.txt":
            return web.Response(text=ROBOTS.format(base=state["base"]), content_type="text/plain")
        if path == "/sitemap.xml":
            return web.Response(text=SITEMAP.format(base=state["base"]), content_type="application/xml")
        if path == "/files/size-guide.pdf":
            return web.Response(body=make_pdf("Waist 30 fits 76 cm."), content_type="application/pdf")
        if path in PAGES:
            return web.Response(text=PAGES[path], content_type="text/html")
        return web.Response(status=404, text="not found")

    app.router.add_route("GET", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    tcp = web.TCPSite(runner, "127.0.0.1", 0)
    await tcp.start()
    port = tcp._server.sockets[0].getsockname()[1]
    state["base"] = f"http://127.0.0.1:{port}"
    yield state["base"]
    await runner.cleanup()


async def _no_browser(url):
    raise AssertionError("static mode must not open a browser")


async def _extract(session, url, page_type="other", allow_http_fallback=False):
    return await extract_page_with_render_mode(
        session, url, page_type=page_type, allow_http_fallback=allow_http_fallback, render_fn=_no_browser,
    )


def _config(**overrides) -> CrawlConfig:
    values = dict(allow_private_hosts=True, politeness_delay=0, max_pages=50, concurrency=3)
    values.update(overrides)
    return CrawlConfig(**values)


@pytest.mark.asyncio
async def test_crawler_follows_links_beyond_the_homepage(site):
    result = await SiteCrawler(site, extract_fn=_extract, config=_config()).run()
    urls = {page["page_url"].replace(site, "") or "/" for page in result["pages"] if page["page_type"] != "site_wide"}

    # Depth 3 page reachable only through /about -> /about/team -> history.
    assert "/about/team/history" in urls
    # Discovered only through the sitemap.
    assert "/sitemap-only" in urls
    # Linked PDF document is extracted as a document page.
    pdf = next(page for page in result["pages"] if page["page_type"] == "document")
    assert "Waist 30 fits 76 cm." in pdf["content"]
    assert result["success"] is True
    assert result["total_pages"] == result["crawl_report"]["pages_stored"]


@pytest.mark.asyncio
async def test_crawler_respects_robots_scope_and_skips_duplicates(site):
    result = await SiteCrawler(site, extract_fn=_extract, config=_config()).run()
    report = result["crawl_report"]
    urls = [page["page_url"].replace(site, "") for page in result["pages"]]

    assert "/private/staff" not in urls
    assert report["skipped"]["robots_txt"] >= 1
    assert "/login" not in urls
    assert not any("other.example" in url for url in urls)
    # /shop/index.html is the same URL as /shop.
    assert "/shop/index.html" not in urls
    # Canonical copy and the grey colour variant are duplicates.
    reasons = {item["url"].replace(site, ""): item["reason"] for item in report["duplicates"]}
    assert reasons.get("/print/about") in {"canonical_duplicate", "identical_content"}
    assert sum(1 for url in urls if url.startswith("/shop/trouser")) == 1
    assert "near_duplicate_content" in reasons.values()


@pytest.mark.asyncio
async def test_crawler_removes_header_footer_and_keeps_them_once(site):
    result = await SiteCrawler(site, extract_fn=_extract, config=_config()).run()
    pages = [page for page in result["pages"] if page["page_type"] not in {"site_wide", "document"}]
    site_wide = next(page for page in result["pages"] if page["page_type"] == "site_wide")

    assert all("Helpline 0311-1115262" not in page["content"] for page in pages)
    assert "Helpline 0311-1115262" in site_wide["content"]
    assert all("_markdown" in page for page in result["pages"])
    about = next(page for page in pages if page["page_url"].endswith("/about"))
    assert "Founded in 2010 in Lahore" in about["content"]


@pytest.mark.asyncio
async def test_page_budget_and_depth_limits(site):
    result = await SiteCrawler(site, extract_fn=_extract, config=_config(max_pages=3)).run()
    assert result["crawl_report"]["pages_stored"] == 3
    assert result["crawl_report"]["skipped"]["page_budget"] > 0

    shallow = await SiteCrawler(site, extract_fn=_extract, config=_config(max_depth=1)).run()
    shallow_urls = {page["page_url"].replace(site, "") for page in shallow["pages"]}
    assert "/about/team" not in shallow_urls


@pytest.mark.asyncio
async def test_section_scope_stays_under_the_path(site):
    result = await SiteCrawler(f"{site}/about/", extract_fn=_extract, config=_config()).run()
    urls = {page["page_url"].replace(site, "") for page in result["pages"] if page["page_type"] != "site_wide"}
    assert urls == {"/about", "/about/team", "/about/team/history"}


def test_detect_crawl_scope():
    assert detect_crawl_scope("https://x.com") == ("site", "")
    assert detect_crawl_scope("https://x.com/docs/") == ("section", "/docs")
    assert detect_crawl_scope("https://x.com/blog/post-1") == ("page", "")


@pytest.mark.asyncio
async def test_private_hosts_are_refused_by_default():
    assert await is_public_host("127.0.0.1") is False
    assert await is_public_host("localhost") is False
    assert await is_public_host("169.254.169.254") is False
    assert await is_public_host("8.8.8.8") is True

    result = await SiteCrawler("http://127.0.0.1:9", extract_fn=_extract, config=CrawlConfig()).run()
    assert result["success"] is False
    assert result["crawl_report"]["skipped"]["private_host"] == 1


@pytest.mark.asyncio
async def test_crawl_to_knowledge_json_exports_and_chunks(site, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(sp, "KNOWLEDGE_DIR", tmp_path)
    scraped = await SiteCrawler(site, extract_fn=_extract, config=_config()).run()
    knowledge = sp.create_knowledge_json(site, scraped, [], sp.derive_site_name(scraped))
    path = sp.save_knowledge_json(knowledge, site)
    stats = {}
    sp._export_knowledge_best_effort(scraped, knowledge, path, stats)

    saved_text = open(path, encoding="utf-8").read()
    saved = json.loads(saved_text)
    assert not any(key.startswith("_") for page in saved["pages"] for key in page)
    assert '"_blocks"' not in saved_text and '"_markdown"' not in saved_text
    assert saved["metadata"]["name"] == "Test Store"  # repeated title segment, no LLM call
    assert saved["metadata"]["crawl"]["pages_stored"] == scraped["crawl_report"]["pages_stored"]
    assert {page["page_type"] for page in saved["pages"]} >= {"homepage", "document", "site_wide"}

    exports = stats["knowledge_exports"]
    assert exports["markdown_pages"] == len(scraped["pages"])
    first_line = json.loads(open(exports["jsonl_path"], encoding="utf-8").readline())
    assert first_line["markdown"].startswith("---")

    chunks = build_chunks_from_knowledge(saved)
    chunk_text = "\n".join(chunk["text"] for chunk in chunks)
    assert "Helpline 0311-1115262" in chunk_text  # kept once via site_wide
    helpline_chunks = [chunk for chunk in chunks if "Helpline 0311-1115262" in chunk["text"]]
    assert helpline_chunks and all(chunk["page_type"] == "site_wide" for chunk in helpline_chunks)
    assert "Waist 30 fits 76 cm." in chunk_text  # from the linked PDF
