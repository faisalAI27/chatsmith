"""HTTP fetching and render-mode selection for a single URL.

``fetch_page_with_retry`` returns a ``FetchResult``: a 3-tuple
``(url, html, error)`` (the historical return value, so existing callers and
tests keep working) that also carries the final URL, status code, content
type, and raw bytes for PDF/DOCX documents.
"""

import asyncio
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Dict, Tuple
from urllib.parse import urlparse

import aiohttp

from .browser_renderer import get_scraper_render_mode, should_render_with_browser
from .document_extractor import detect_document_kind, extract_document_page
from .static_extractor import extract_static_page

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_DELAY = 1.0
MAX_RETRY_AFTER_SECONDS = 30
MAX_HTML_BYTES = 8 * 1024 * 1024
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/xml", "text/xml")
NO_RETRY_STATUSES = {400, 401, 403, 404, 405, 410, 451}
NO_RENDER_STATUSES = {404, 410}
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.I)


class FetchResult(tuple):
    """``(url, html, error)`` plus fetch details as attributes."""

    def __new__(
        cls,
        url: str,
        html: str,
        error: str,
        *,
        final_url: str = "",
        status: int = 0,
        content_type: str = "",
        kind: str = "",
        body: bytes = b"",
        elapsed_ms: int = 0,
    ):
        obj = super().__new__(cls, (url, html, error))
        obj.final_url = final_url or url
        obj.status = status
        obj.content_type = content_type
        obj.kind = kind
        obj.body = body
        obj.elapsed_ms = elapsed_ms
        return obj

    @property
    def url(self) -> str:
        return self[0]

    @property
    def html(self) -> str:
        return self[1]

    @property
    def error(self) -> str:
        return self[2]


def as_fetch_result(value: Any, url: str) -> FetchResult:
    """Accept a FetchResult or a plain (url, html, error) tuple from a stub."""
    if isinstance(value, FetchResult):
        return value
    fetched_url, html, error = value
    return FetchResult(
        fetched_url or url,
        html or "",
        error or "",
        status=200 if html else 0,
        content_type="text/html" if html else "",
        kind="html" if html else "",
    )


def decode_html(body: bytes, header_charset: str | None = None) -> str:
    """Decode bytes using the HTTP charset, then <meta charset>, then UTF-8."""
    candidates = []
    if header_charset:
        candidates.append(header_charset)
    match = _META_CHARSET_RE.search(body[:4096])
    if match:
        candidates.append(match.group(1).decode("ascii", "ignore"))
    candidates.extend(["utf-8", "cp1252"])
    for encoding in candidates:
        try:
            return body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def parse_retry_after(value: str | None, default: float = 5.0) -> float:
    """Parse Retry-After as seconds or an HTTP date, capped to a sane range."""
    if not value:
        return default
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, IndexError, OverflowError):
            seconds = default
    return max(0.0, min(seconds, MAX_RETRY_AFTER_SECONDS))


def _content_kind(content_type: str, url: str, body: bytes) -> str:
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    document_kind = detect_document_kind(content_type, url, body[:8])
    if document_kind:
        return document_kind
    if not content_type or content_type.startswith(HTML_CONTENT_TYPES):
        return "html"
    return "other"


async def fetch_page_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    retries: int = MAX_RETRIES,
    headers: Dict[str, str] | None = None,
) -> FetchResult:
    """Fetch one URL with retries, safe decoding, and content-type checks."""
    last_error = ""
    last_status = 0
    request_headers = dict(DEFAULT_HEADERS)
    if headers:
        request_headers.update(headers)
    start = time.perf_counter()

    for attempt in range(max(1, retries)):
        try:
            async with session.get(
                url,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, sock_connect=10),
                allow_redirects=True,
            ) as response:
                last_status = response.status
                final_url = str(response.url)
                content_type = response.headers.get("Content-Type", "")

                if 200 <= response.status < 300:
                    is_document = bool(detect_document_kind(content_type.split(";")[0].strip().lower(), final_url, b""))
                    limit = MAX_DOCUMENT_BYTES if is_document else MAX_HTML_BYTES
                    body = await response.content.read(limit + 1)
                    truncated = len(body) > limit
                    body = body[:limit]
                    kind = _content_kind(content_type, final_url, body)
                    elapsed_ms = int((time.perf_counter() - start) * 1000)
                    if kind == "html":
                        html = decode_html(body, response.charset)
                        return FetchResult(
                            url, html, "",
                            final_url=final_url, status=response.status, content_type=content_type,
                            kind="html", elapsed_ms=elapsed_ms,
                        )
                    if kind in {"pdf", "docx"}:
                        if truncated:
                            return FetchResult(url, "", "document_too_large", final_url=final_url, status=response.status,
                                               content_type=content_type, kind=kind, elapsed_ms=elapsed_ms)
                        return FetchResult(
                            url, "", "",
                            final_url=final_url, status=response.status, content_type=content_type,
                            kind=kind, body=body, elapsed_ms=elapsed_ms,
                        )
                    return FetchResult(
                        url, "", f"unsupported_content_type: {content_type.split(';')[0] or 'unknown'}",
                        final_url=final_url, status=response.status, content_type=content_type,
                        kind="other", elapsed_ms=elapsed_ms,
                    )

                if response.status in {429, 503}:
                    wait_time = parse_retry_after(response.headers.get("Retry-After"))
                    last_error = "rate_limited" if response.status == 429 else "server_error_503"
                    if attempt < retries - 1:
                        print(f"  ⏳ HTTP {response.status} for {url[:60]}, waiting {wait_time:.0f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    break
                if response.status in {401, 403}:
                    last_error = f"access_denied_{response.status}"
                    break
                if response.status in {404, 410}:
                    last_error = "not_found"
                    break
                if response.status >= 500:
                    last_error = f"server_error_{response.status}"
                    if attempt < retries - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    break
                last_error = f"http_{response.status}"
                break

        except asyncio.TimeoutError:
            last_error = "timeout"
            if attempt < retries - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
        except aiohttp.ClientError as exc:
            last_error = f"client_error: {str(exc)[:80]}"
            if attempt < retries - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
        except Exception as exc:
            last_error = f"error: {str(exc)[:80]}"
            break

    if last_error:
        print(f"  ❌ Failed {url[:60]}: {last_error}")
    return FetchResult(url, "", last_error, status=last_status, elapsed_ms=int((time.perf_counter() - start) * 1000))


FetchFn = Callable[..., Awaitable[Any]]
RenderFn = Callable[[str], Awaitable[Dict[str, Any]]]


def _render_quality_extra(render_result: Dict) -> Dict:
    quality_extra = {"render_time_ms": int(render_result.get("render_time_ms") or 0)}
    final_url = render_result.get("final_url")
    if final_url:
        quality_extra["final_url"] = final_url
    if render_result.get("warnings"):
        quality_extra["render_warnings"] = list(render_result["warnings"])
    return quality_extra


def _add_page_quality_error(page_data: Dict, error_message: str) -> None:
    quality = page_data.setdefault("quality", {})
    errors = quality.setdefault("errors", [])
    if not isinstance(errors, list):
        errors = [str(errors)]
        quality["errors"] = errors
    if error_message:
        errors.append(error_message)


async def fetch_static_html_with_fallback(
    session: aiohttp.ClientSession,
    page_url: str,
    allow_http_fallback: bool = False,
    fetch_fn: FetchFn | None = None,
) -> Tuple[str, str, str, FetchResult]:
    """Fetch static HTML, optionally falling back from HTTPS to HTTP for the homepage.

    Returns (effective_url, html, error, fetch_result).
    """
    fetch = fetch_fn or fetch_page_with_retry
    result = as_fetch_result(await fetch(session, page_url), page_url)
    effective_url = page_url

    if allow_http_fallback and not result.html and not result.body and page_url.startswith("https://") and result.status == 0:
        fallback_url = "http://" + page_url[len("https://"):]
        print(f"  🔁 HTTPS fetch failed, retrying with HTTP: {fallback_url}")
        fallback = as_fetch_result(await fetch(session, fallback_url), fallback_url)
        if fallback.html or fallback.body:
            result = fallback
            effective_url = fallback_url.rstrip("/")

    if result.html or result.body:
        final_url = getattr(result, "final_url", "") or effective_url
        if final_url and urlparse(final_url).netloc and final_url.rstrip("/") != effective_url.rstrip("/"):
            effective_url = final_url
    return effective_url, result.html, result.error, result


def _status_code(result: FetchResult | None) -> int:
    status = getattr(result, "status", 0) if result is not None else 0
    return int(status or 200)


async def extract_page_with_render_mode(
    session: aiohttp.ClientSession,
    page_url: str,
    page_type: str = "other",
    allow_http_fallback: bool = False,
    fetch_fn: FetchFn | None = None,
    render_fn: RenderFn | None = None,
) -> Tuple[Dict | None, str, str, str]:
    """Extract one page according to SCRAPER_RENDER_MODE.

    Returns (page_record, html_used_for_discovery, effective_url, error_message).
    PDF/DOCX responses are turned into document page records.
    """
    if render_fn is None:
        from .browser_renderer import render_page_with_playwright

        render_fn = render_page_with_playwright
    render_mode = get_scraper_render_mode()
    looks_like_document = bool(detect_document_kind("", page_url, b""))

    if render_mode == "browser" and not looks_like_document:
        render_result = await render_fn(page_url)
        if render_result.get("success"):
            effective_url = render_result.get("final_url") or page_url
            html = render_result.get("html", "")
            return (
                extract_static_page(
                    html,
                    page_url=effective_url,
                    page_type=page_type,
                    status_code=int(render_result.get("status_code") or 200),
                    extraction_method="playwright",
                    quality_extra=_render_quality_extra(render_result),
                ),
                html,
                effective_url,
                "",
            )

        render_error = render_result.get("error", "Playwright render failed")
        effective_url, static_html, static_error, fetch_result = await fetch_static_html_with_fallback(
            session, page_url, allow_http_fallback=allow_http_fallback, fetch_fn=fetch_fn,
        )
        if static_html:
            return (
                extract_static_page(
                    static_html,
                    page_url=effective_url,
                    page_type=page_type,
                    status_code=_status_code(fetch_result),
                    errors=[render_error],
                    extraction_method="static_after_playwright_failed",
                    quality_extra=_render_quality_extra(render_result),
                ),
                static_html,
                effective_url,
                "",
            )
        return (
            None,
            "",
            effective_url,
            f"{render_error}; static fallback failed: {static_error}",
        )

    effective_url, static_html, static_error, fetch_result = await fetch_static_html_with_fallback(
        session, page_url, allow_http_fallback=allow_http_fallback, fetch_fn=fetch_fn,
    )

    if getattr(fetch_result, "kind", "") in {"pdf", "docx"} and getattr(fetch_result, "body", b""):
        page = extract_document_page(
            fetch_result.body,
            kind=fetch_result.kind,
            page_url=effective_url,
            status_code=_status_code(fetch_result),
        )
        if page is None:
            return None, "", effective_url, f"{fetch_result.kind}_extraction_failed"
        return page, "", effective_url, ""

    if not static_html:
        status = getattr(fetch_result, "status", 0)
        if render_mode == "auto" and status not in NO_RENDER_STATUSES and not looks_like_document and not static_error.startswith("unsupported_content_type"):
            # Bot-protected sites often refuse plain HTTP clients but serve a real browser.
            render_result = await render_fn(page_url)
            if render_result.get("success"):
                effective_url = render_result.get("final_url") or page_url
                html = render_result.get("html", "")
                return (
                    extract_static_page(
                        html,
                        page_url=effective_url,
                        page_type=page_type,
                        status_code=int(render_result.get("status_code") or 200),
                        extraction_method="playwright",
                        quality_extra=_render_quality_extra(render_result),
                    ),
                    html,
                    effective_url,
                    "",
                )
            render_error = render_result.get("error", "Playwright render failed")
            return None, "", effective_url, f"Static fetch failed: {static_error}; {render_error}"
        return None, "", effective_url, static_error

    static_page = extract_static_page(
        static_html,
        page_url=effective_url,
        page_type=page_type,
        status_code=_status_code(fetch_result),
        extraction_method="static",
    )

    if render_mode == "static" or not should_render_with_browser(static_page, static_html):
        return static_page, static_html, effective_url, ""

    render_result = await render_fn(effective_url)
    if render_result.get("success"):
        rendered_url = render_result.get("final_url") or effective_url
        rendered_html = render_result.get("html", "")
        rendered_page = extract_static_page(
            rendered_html,
            page_url=rendered_url,
            page_type=page_type,
            status_code=int(render_result.get("status_code") or 200),
            extraction_method="playwright",
            quality_extra=_render_quality_extra(render_result),
        )
        # Keep whichever version actually holds more of the page's own text.
        static_core = int(static_page["quality"].get("core_text_length") or 0)
        rendered_core = int(rendered_page["quality"].get("core_text_length") or 0)
        if rendered_core >= static_core * 0.8:
            return rendered_page, rendered_html, rendered_url, ""
        static_page["extraction_method"] = "static_preferred_over_playwright"
        static_page["quality"].update(_render_quality_extra(render_result))
        return static_page, static_html, effective_url, ""

    render_error = render_result.get("error", "Playwright render failed")
    static_page["extraction_method"] = "static_with_playwright_failed"
    static_page["quality"].update(_render_quality_extra(render_result))
    _add_page_quality_error(static_page, render_error)
    return static_page, static_html, effective_url, ""
