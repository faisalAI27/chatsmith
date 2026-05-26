import os
import re
import time
from typing import Any, Dict

from bs4 import BeautifulSoup


VALID_RENDER_MODES = {"auto", "browser", "static"}
DEFAULT_RENDER_MODE = "auto"
DEFAULT_PLAYWRIGHT_TIMEOUT_MS = 15000
DEFAULT_PLAYWRIGHT_WAIT_MS = 1000
DEFAULT_BLOCK_HEAVY_RESOURCES = True
PLAYWRIGHT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

SPA_ROOT_IDS = {"root", "app", "__next", "__nuxt"}
HYDRATION_MARKERS = (
    "__NEXT_DATA__",
    "__NUXT__",
    "data-reactroot",
    "hydrateRoot",
    "createRoot(",
    "ng-version",
    "data-v-app",
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_scraper_render_mode() -> str:
    """Return the configured scraper render mode, falling back safely to auto."""
    mode = (os.getenv("SCRAPER_RENDER_MODE", DEFAULT_RENDER_MODE) or "").strip().lower()
    if mode not in VALID_RENDER_MODES:
        return DEFAULT_RENDER_MODE
    return mode


def get_playwright_timeout_ms() -> int:
    return _env_int("PLAYWRIGHT_TIMEOUT_MS", DEFAULT_PLAYWRIGHT_TIMEOUT_MS)


def get_playwright_wait_ms() -> int:
    return _env_int("PLAYWRIGHT_WAIT_MS", DEFAULT_PLAYWRIGHT_WAIT_MS)


def should_block_heavy_resources() -> bool:
    return _env_bool("PLAYWRIGHT_BLOCK_HEAVY_RESOURCES", DEFAULT_BLOCK_HEAVY_RESOURCES)


def browser_rendering_available() -> bool:
    """Return True when the Playwright Python package is importable."""
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except Exception:
        return False
    return True


def should_render_with_browser(static_page_record: dict, raw_html: str) -> bool:
    """Decide whether static extraction looks weak enough to need browser rendering."""
    if not raw_html:
        return True

    page = static_page_record if isinstance(static_page_record, dict) else {}
    quality = page.get("quality") if isinstance(page.get("quality"), dict) else {}
    content = page.get("content", "") or ""
    content_length = int(quality.get("content_length") or len(content))
    paragraph_count = len(page.get("paragraphs") or [])

    soup = BeautifulSoup(raw_html or "", "lxml")
    body = soup.find("body")
    body_text = _clean_text(body.get_text(" ")) if body else ""
    script_count = len(soup.find_all("script"))
    has_title = bool(soup.title and _clean_text(soup.title.get_text()))
    has_spa_root = any(soup.find(id=root_id) is not None for root_id in SPA_ROOT_IDS)
    has_react_root = soup.find(attrs={"data-reactroot": True}) is not None
    has_hydration_marker = any(marker in raw_html for marker in HYDRATION_MARKERS)

    if content_length >= 800 and paragraph_count >= 2:
        return False
    if content_length < 500:
        return True
    if paragraph_count == 0 and has_title and content_length < 1200:
        return True
    if (has_spa_root or has_react_root or has_hydration_marker) and content_length < 1500:
        return True
    if script_count >= 8 and len(body_text) < 800:
        return True

    return False


def _render_result(
    success: bool,
    html: str,
    final_url: str,
    error: str,
    render_time_ms: int,
) -> Dict[str, Any]:
    return {
        "success": success,
        "html": html,
        "final_url": final_url,
        "error": error,
        "render_time_ms": render_time_ms,
    }


async def render_page_with_playwright(
    url: str,
    timeout_ms: int | None = None,
    wait_ms: int | None = None,
) -> Dict[str, Any]:
    """Render a public page with Chromium and return its final DOM HTML."""
    timeout = timeout_ms if timeout_ms is not None else get_playwright_timeout_ms()
    wait = wait_ms if wait_ms is not None else get_playwright_wait_ms()
    block_heavy_resources = should_block_heavy_resources()
    start = time.perf_counter()

    try:
        from playwright.async_api import async_playwright
    except Exception as exc:
        return _render_result(
            success=False,
            html="",
            final_url=url,
            error=f"Playwright is not installed or importable: {exc}",
            render_time_ms=0,
        )

    browser = None
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=PLAYWRIGHT_USER_AGENT)

            if block_heavy_resources:

                async def route_handler(route):
                    resource_type = route.request.resource_type
                    if resource_type in {"font", "media"}:
                        await route.abort()
                    else:
                        await route.continue_()

                await page.route("**/*", route_handler)

            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            if wait > 0:
                await page.wait_for_timeout(wait)
            html = await page.content()
            final_url = page.url or url
            render_time_ms = int((time.perf_counter() - start) * 1000)
            return _render_result(
                success=True,
                html=html,
                final_url=final_url,
                error="",
                render_time_ms=render_time_ms,
            )
    except Exception as exc:
        return _render_result(
            success=False,
            html="",
            final_url=url,
            error=f"Playwright render failed: {exc}",
            render_time_ms=0,
        )
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass
