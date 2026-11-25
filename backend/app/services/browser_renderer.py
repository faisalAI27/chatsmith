"""Playwright rendering for JavaScript-heavy public pages.

A crawl opens one ``BrowserSession`` and every render reuses its Chromium
instance (launching a browser per page cost seconds per URL). Rendering is
tolerant: if navigation times out but the DOM already has content, that
content is kept instead of discarding the whole render.
"""

import asyncio
import contextvars
import re
import time
from typing import Any, Dict

from bs4 import BeautifulSoup

from ..core.config import get_settings


VALID_RENDER_MODES = {"auto", "browser", "static"}
DEFAULT_RENDER_MODE = "auto"
DEFAULT_PLAYWRIGHT_TIMEOUT_MS = 30000
DEFAULT_PLAYWRIGHT_WAIT_MS = 1000
DEFAULT_BLOCK_HEAVY_RESOURCES = True
DEFAULT_NETWORK_IDLE_MS = 3000
DEFAULT_SCROLL_STEPS = 4
DEFAULT_STABILIZE_MS = 3000
DEFAULT_MAX_CONCURRENT_PAGES = 2
MIN_TEXT_AFTER_TIMEOUT = 50
MIN_DOM_GRACE_MS = 8000
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

_SCROLL_JS = """
async (steps) => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  for (let i = 0; i < steps; i++) {
    const height = document.body ? document.body.scrollHeight : 0;
    window.scrollTo(0, height);
    await sleep(400);
    if (!document.body || document.body.scrollHeight === height) break;
  }
  window.scrollTo(0, 0);
}
"""
_EXPAND_JS = """
() => {
  document.querySelectorAll('details:not([open])').forEach((d) => d.setAttribute('open', ''));
  for (const host of document.querySelectorAll('*')) {
    if (!host.shadowRoot) continue;
    try {
      const copy = document.createElement('div');
      copy.setAttribute('data-chatsmith-shadow-root', '1');
      copy.innerHTML = host.shadowRoot.innerHTML;
      host.appendChild(copy);
    } catch (e) {}
  }
}
"""
_TEXT_LENGTH_JS = "() => (document.body ? document.body.innerText.length : 0)"

_active_session: contextvars.ContextVar["BrowserSession | None"] = contextvars.ContextVar(
    "chatsmith_browser_session",
    default=None,
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _settings_value(name: str, default: Any) -> Any:
    try:
        return getattr(get_settings(), name, default)
    except Exception:
        return default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def get_scraper_render_mode() -> str:
    """Return the configured scraper render mode from backend settings."""
    mode = str(_settings_value("scraper_render_mode", DEFAULT_RENDER_MODE) or "").strip().lower()
    if mode not in VALID_RENDER_MODES:
        return DEFAULT_RENDER_MODE
    return mode


def get_playwright_timeout_ms() -> int:
    return _positive_int(
        _settings_value("playwright_timeout_ms", DEFAULT_PLAYWRIGHT_TIMEOUT_MS),
        DEFAULT_PLAYWRIGHT_TIMEOUT_MS,
    )


def get_playwright_wait_ms() -> int:
    return _positive_int(
        _settings_value("playwright_wait_ms", DEFAULT_PLAYWRIGHT_WAIT_MS),
        DEFAULT_PLAYWRIGHT_WAIT_MS,
    )


def should_block_heavy_resources() -> bool:
    return bool(
        _settings_value(
            "playwright_block_heavy_resources",
            DEFAULT_BLOCK_HEAVY_RESOURCES,
        )
    )


def get_playwright_scroll_steps() -> int:
    return _non_negative_int(_settings_value("playwright_scroll_steps", DEFAULT_SCROLL_STEPS), DEFAULT_SCROLL_STEPS)


def get_playwright_max_concurrent_pages() -> int:
    return _positive_int(
        _settings_value("playwright_max_concurrent_pages", DEFAULT_MAX_CONCURRENT_PAGES),
        DEFAULT_MAX_CONCURRENT_PAGES,
    )


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
    if quality.get("requires_javascript"):
        return True
    content = page.get("content", "") or ""
    # Prefer the page's own text (main content, not header/footer) so a
    # server-rendered menu and footer cannot hide an empty client-side body.
    content_length = int(quality.get("core_text_length") or quality.get("content_length") or len(content))
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
    status_code: int | None = None,
    warnings: list[str] | None = None,
) -> Dict[str, Any]:
    return {
        "success": success,
        "html": html,
        "final_url": final_url,
        "error": error,
        "render_time_ms": render_time_ms,
        "status_code": status_code,
        "warnings": list(warnings or []),
    }


async def _wait_for_dom_stable(page, max_ms: int, interval_ms: int = 300) -> None:
    """Poll visible text length until it stops changing (client data has loaded)."""
    previous = -1
    waited = 0
    while waited <= max_ms:
        try:
            current = int(await page.evaluate(_TEXT_LENGTH_JS) or 0)
        except Exception:
            return
        if current == previous and current > 0:
            return
        previous = current
        await page.wait_for_timeout(interval_ms)
        waited += interval_ms


async def _render_on_page(page, url: str, timeout: int, wait: int, scroll_steps: int) -> Dict[str, Any]:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    start = time.perf_counter()
    warnings: list[str] = []
    status_code = None
    navigation_timed_out = False

    # Two-step navigation: "commit" returns as soon as the server answers, then
    # DOMContentLoaded gets its own grace period. Slow sites that blow the
    # budget still yield whatever DOM has been built.
    try:
        response = await page.goto(url, wait_until="commit", timeout=timeout)
        status_code = response.status if response is not None else None
    except PlaywrightTimeoutError:
        return _render_result(
            False, "", url, f"Playwright render failed: no response within {timeout}ms",
            int((time.perf_counter() - start) * 1000),
        )
    except Exception as exc:
        return _render_result(False, "", url, f"Playwright render failed: {exc}", int((time.perf_counter() - start) * 1000))

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=max(MIN_DOM_GRACE_MS, timeout - elapsed_ms))
    except PlaywrightTimeoutError:
        navigation_timed_out = True
        warnings.append(f"DOMContentLoaded not reached within {timeout}ms; using the DOM rendered so far")
    except Exception:
        pass

    try:
        await page.wait_for_load_state("networkidle", timeout=DEFAULT_NETWORK_IDLE_MS)
    except Exception:
        pass  # long-polling/analytics sites never go idle; that is fine
    if wait > 0:
        await page.wait_for_timeout(wait)
    try:
        if scroll_steps:
            await page.evaluate(_SCROLL_JS, scroll_steps)
        await _wait_for_dom_stable(page, DEFAULT_STABILIZE_MS)
        await page.evaluate(_EXPAND_JS)
    except Exception as exc:
        warnings.append(f"post-load scripts failed: {str(exc)[:120]}")

    try:
        html = await page.content()
        text_length = int(await page.evaluate(_TEXT_LENGTH_JS) or 0)
    except Exception as exc:
        return _render_result(False, "", url, f"Playwright render failed: {exc}", int((time.perf_counter() - start) * 1000))

    final_url = page.url or url
    render_time_ms = int((time.perf_counter() - start) * 1000)
    if navigation_timed_out and text_length < MIN_TEXT_AFTER_TIMEOUT:
        return _render_result(
            False, "", final_url,
            f"Playwright render failed: Page.goto timeout {timeout}ms and the page stayed empty",
            render_time_ms, status_code, warnings,
        )
    if status_code is not None and status_code >= 400:
        return _render_result(False, "", final_url, f"Playwright render failed: http_{status_code}", render_time_ms, status_code, warnings)
    return _render_result(True, html, final_url, "", render_time_ms, status_code, warnings)


class BrowserSession:
    """One Chromium instance shared by every render inside ``async with``.

    The browser starts lazily on the first render, so static-only crawls never
    launch Chromium. While the session is open, ``render_page_with_playwright``
    uses it automatically (through a context variable), which keeps the simple
    ``render_page_with_playwright(url)`` call signature used across the code.
    """

    def __init__(
        self,
        max_concurrent_pages: int | None = None,
        block_heavy_resources: bool | None = None,
        user_agent: str = PLAYWRIGHT_USER_AGENT,
    ):
        self.max_concurrent_pages = max_concurrent_pages or get_playwright_max_concurrent_pages()
        self.block_heavy_resources = should_block_heavy_resources() if block_heavy_resources is None else block_heavy_resources
        self.user_agent = user_agent
        self.launch_error = ""
        self.pages_rendered = 0
        self._playwright_cm = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.max_concurrent_pages)
        self._token = None

    async def __aenter__(self) -> "BrowserSession":
        self._token = _active_session.set(self)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._token is not None:
            _active_session.reset(self._token)
            self._token = None
        await self.close()

    async def _ensure_started(self) -> bool:
        if self._context is not None:
            return True
        if self.launch_error:
            return False
        async with self._lock:
            if self._context is not None:
                return True
            if self.launch_error:
                return False
            try:
                from playwright.async_api import async_playwright
            except Exception as exc:
                self.launch_error = f"Playwright is not installed or importable: {exc}"
                return False
            try:
                self._playwright_cm = async_playwright()
                self._playwright = await self._playwright_cm.__aenter__()
                launch_args = ["--disable-dev-shm-usage", "--mute-audio", "--autoplay-policy=user-gesture-required"]
                if self.block_heavy_resources:
                    # Native image blocking: no per-request Python round trip
                    # (request interception made slow sites miss the timeout).
                    # <img src> attributes stay in the DOM, so image metadata
                    # is still extracted.
                    launch_args.append("--blink-settings=imagesEnabled=false")
                self._browser = await self._playwright.chromium.launch(headless=True, args=launch_args)
                self._context = await self._browser.new_context(
                    user_agent=self.user_agent,
                    locale="en-US",
                    viewport={"width": 1366, "height": 900},
                    ignore_https_errors=True,
                )
            except Exception as exc:
                self.launch_error = f"Playwright browser launch failed: {exc}"
                await self.close()
                return False
        return True

    async def render(self, url: str, timeout_ms: int | None = None, wait_ms: int | None = None) -> Dict[str, Any]:
        timeout = timeout_ms if timeout_ms is not None else get_playwright_timeout_ms()
        wait = wait_ms if wait_ms is not None else get_playwright_wait_ms()
        if not await self._ensure_started():
            return _render_result(False, "", url, self.launch_error, 0)
        async with self._semaphore:
            page = None
            try:
                page = await self._context.new_page()
                result = await _render_on_page(page, url, timeout, wait, get_playwright_scroll_steps())
                if result["success"]:
                    self.pages_rendered += 1
                return result
            except Exception as exc:
                return _render_result(False, "", url, f"Playwright render failed: {exc}", 0)
            finally:
                if page is not None:
                    try:
                        await page.close()
                    except Exception:
                        pass

    async def close(self) -> None:
        for closer in (
            getattr(self._context, "close", None),
            getattr(self._browser, "close", None),
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:
                pass
        if self._playwright_cm is not None:
            try:
                await self._playwright_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._context = None
        self._browser = None
        self._playwright = None
        self._playwright_cm = None


def get_active_browser_session() -> BrowserSession | None:
    return _active_session.get()


async def render_page_with_playwright(
    url: str,
    timeout_ms: int | None = None,
    wait_ms: int | None = None,
) -> Dict[str, Any]:
    """Render a public page with Chromium and return its final DOM HTML.

    Uses the active ``BrowserSession`` when one is open (normal crawls);
    otherwise starts a one-off browser for this single call.
    """
    session = _active_session.get()
    if session is not None:
        return await session.render(url, timeout_ms=timeout_ms, wait_ms=wait_ms)
    async with BrowserSession() as one_off:
        return await one_off.render(url, timeout_ms=timeout_ms, wait_ms=wait_ms)
