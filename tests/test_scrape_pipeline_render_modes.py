import pytest

from backend.app.core.config import get_settings
from backend.app.services import scrape_pipeline as sp


RICH_HTML = """
<!doctype html>
<html>
  <head><title>Static Site</title></head>
  <body>
    <main>
      <h1>Static Site</h1>
      <p>This static page has enough meaningful text for the extractor to keep it
      without opening a browser in auto mode.</p>
      <p>It includes a second useful paragraph so paragraph count and content length
      both pass the browser rendering detector.</p>
      <p>More useful public page content about services, pricing, and contact details
      gives the static scraper a sufficiently rich record.</p>
      <p>The page also includes stable public information about support channels,
      implementation options, product categories, and frequently asked questions.
      This extra body copy represents a normal server-rendered marketing page where
      static HTML already contains enough reliable content for the knowledge file.</p>
      <p>Because all of this content is present before JavaScript runs, opening a
      browser would add cost without improving the extracted result.</p>
    </main>
  </body>
</html>
"""

WEAK_HTML = """
<!doctype html>
<html>
  <head><title>Client App</title><script>window.__NEXT_DATA__ = {}</script></head>
  <body><div id="root"></div><script src="/app.js"></script></body>
</html>
"""

RENDERED_HTML = """
<!doctype html>
<html>
  <head><title>Rendered App</title></head>
  <body>
    <main>
      <h1>Rendered App</h1>
      <p>Client-rendered content is now visible after Playwright rendering.</p>
    </main>
  </body>
</html>
"""


async def _fake_fetch_factory(html: str, error: str = ""):
    async def fake_fetch_page_with_retry(session, url, retries=sp.MAX_RETRIES):
        return url, html, error

    return fake_fetch_page_with_retry


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_static_mode_never_calls_playwright(monkeypatch):
    monkeypatch.setenv("SCRAPER_RENDER_MODE", "static")
    monkeypatch.setattr(sp, "fetch_page_with_retry", await _fake_fetch_factory(RICH_HTML))

    async def fail_if_called(url):
        raise AssertionError("Playwright should not be called in static mode")

    monkeypatch.setattr(sp, "render_page_with_playwright", fail_if_called)

    page, html, effective_url, error = await sp.extract_page_with_render_mode(
        session=object(),
        page_url="https://example.com",
        page_type="homepage",
    )

    assert error == ""
    assert html == RICH_HTML
    assert effective_url == "https://example.com"
    assert page["extraction_method"] == "static"


@pytest.mark.asyncio
async def test_browser_mode_attempts_playwright(monkeypatch):
    monkeypatch.setenv("SCRAPER_RENDER_MODE", "browser")

    async def fail_if_called(session, url, retries=sp.MAX_RETRIES):
        raise AssertionError("Static fetch should not run before Playwright in browser mode")

    async def fake_render(url):
        return {
            "success": True,
            "html": RENDERED_HTML,
            "final_url": "https://example.com/rendered",
            "error": "",
            "render_time_ms": 123,
        }

    monkeypatch.setattr(sp, "fetch_page_with_retry", fail_if_called)
    monkeypatch.setattr(sp, "render_page_with_playwright", fake_render)

    page, html, effective_url, error = await sp.extract_page_with_render_mode(
        session=object(),
        page_url="https://example.com",
        page_type="homepage",
    )

    assert error == ""
    assert html == RENDERED_HTML
    assert effective_url == "https://example.com/rendered"
    assert page["extraction_method"] == "playwright"
    assert page["quality"]["render_time_ms"] == 123
    assert page["quality"]["final_url"] == "https://example.com/rendered"


@pytest.mark.asyncio
async def test_auto_mode_keeps_rich_static_html_without_playwright(monkeypatch):
    monkeypatch.setenv("SCRAPER_RENDER_MODE", "auto")
    monkeypatch.setattr(sp, "fetch_page_with_retry", await _fake_fetch_factory(RICH_HTML))

    async def fail_if_called(url):
        raise AssertionError("Playwright should not be called for rich static HTML")

    monkeypatch.setattr(sp, "render_page_with_playwright", fail_if_called)

    page, _, _, error = await sp.extract_page_with_render_mode(
        session=object(),
        page_url="https://example.com",
        page_type="homepage",
    )

    assert error == ""
    assert page["extraction_method"] == "static"


@pytest.mark.asyncio
async def test_auto_mode_uses_playwright_for_weak_static_html(monkeypatch):
    monkeypatch.setenv("SCRAPER_RENDER_MODE", "auto")
    monkeypatch.setattr(sp, "fetch_page_with_retry", await _fake_fetch_factory(WEAK_HTML))

    async def fake_render(url):
        return {
            "success": True,
            "html": RENDERED_HTML,
            "final_url": url,
            "error": "",
            "render_time_ms": 456,
        }

    monkeypatch.setattr(sp, "render_page_with_playwright", fake_render)

    page, html, effective_url, error = await sp.extract_page_with_render_mode(
        session=object(),
        page_url="https://example.com",
        page_type="homepage",
    )

    assert error == ""
    assert html == RENDERED_HTML
    assert effective_url == "https://example.com"
    assert page["extraction_method"] == "playwright"
    assert page["quality"]["render_time_ms"] == 456


@pytest.mark.asyncio
async def test_auto_mode_playwright_failure_keeps_static_page_with_error(monkeypatch):
    monkeypatch.setenv("SCRAPER_RENDER_MODE", "auto")
    monkeypatch.setattr(sp, "fetch_page_with_retry", await _fake_fetch_factory(WEAK_HTML))

    async def fake_render(url):
        return {
            "success": False,
            "html": "",
            "final_url": url,
            "error": "Playwright render failed: browser missing",
            "render_time_ms": 0,
        }

    monkeypatch.setattr(sp, "render_page_with_playwright", fake_render)

    page, html, effective_url, error = await sp.extract_page_with_render_mode(
        session=object(),
        page_url="https://example.com",
        page_type="homepage",
    )

    assert error == ""
    assert html == WEAK_HTML
    assert effective_url == "https://example.com"
    assert page["extraction_method"] == "static_with_playwright_failed"
    assert "Playwright render failed: browser missing" in page["quality"]["errors"]
    assert page["quality"]["render_time_ms"] == 0


@pytest.mark.asyncio
async def test_browser_mode_failure_falls_back_to_static(monkeypatch):
    monkeypatch.setenv("SCRAPER_RENDER_MODE", "browser")
    monkeypatch.setattr(sp, "fetch_page_with_retry", await _fake_fetch_factory(RICH_HTML))

    async def fake_render(url):
        return {
            "success": False,
            "html": "",
            "final_url": url,
            "error": "Playwright render failed: timeout",
            "render_time_ms": 0,
        }

    monkeypatch.setattr(sp, "render_page_with_playwright", fake_render)

    page, html, effective_url, error = await sp.extract_page_with_render_mode(
        session=object(),
        page_url="https://example.com",
        page_type="homepage",
    )

    assert error == ""
    assert html == RICH_HTML
    assert effective_url == "https://example.com"
    assert page["extraction_method"] == "static_after_playwright_failed"
    assert "Playwright render failed: timeout" in page["quality"]["errors"]
