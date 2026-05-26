from backend.app.services.browser_renderer import (
    get_scraper_render_mode,
    should_render_with_browser,
)
from backend.app.services.static_extractor import extract_static_page


RICH_HTML = """
<!doctype html>
<html>
  <head><title>Rich Site</title></head>
  <body>
    <main>
      <h1>Rich Site</h1>
      <p>This public static page has enough useful text to extract without a browser.
      It explains the company, services, and contact details in plain HTML.</p>
      <p>This second paragraph gives the extractor enough paragraph structure and
      enough content length that browser rendering should not be needed in auto mode.</p>
      <p>Additional descriptive text keeps this fixture above the richness threshold
      for normal static websites that do not require JavaScript hydration.</p>
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

SPA_HTML = """
<!doctype html>
<html>
  <head>
    <title>Client App</title>
    <script>window.__NEXT_DATA__ = {"props": {}}</script>
  </head>
  <body>
    <div id="__next"></div>
    <script src="/bundle.js"></script>
    <script>hydrateRoot(document.getElementById("__next"))</script>
  </body>
</html>
"""


def test_get_scraper_render_mode_defaults_and_accepts_valid_modes(monkeypatch):
    monkeypatch.delenv("SCRAPER_RENDER_MODE", raising=False)
    assert get_scraper_render_mode() == "auto"

    monkeypatch.setenv("SCRAPER_RENDER_MODE", "browser")
    assert get_scraper_render_mode() == "browser"

    monkeypatch.setenv("SCRAPER_RENDER_MODE", "static")
    assert get_scraper_render_mode() == "static"


def test_get_scraper_render_mode_invalid_value_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("SCRAPER_RENDER_MODE", "invalid")

    assert get_scraper_render_mode() == "auto"


def test_should_render_with_browser_false_for_rich_static_html():
    page = extract_static_page(RICH_HTML, page_url="https://example.com", page_type="homepage")

    assert should_render_with_browser(page, RICH_HTML) is False


def test_should_render_with_browser_true_for_spa_like_empty_root_html():
    page = extract_static_page(SPA_HTML, page_url="https://example.com", page_type="homepage")

    assert should_render_with_browser(page, SPA_HTML) is True
