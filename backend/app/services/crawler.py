"""Frontier-based website crawler.

Starting from an owner-submitted URL the crawler:

1. reads robots.txt (rules, crawl-delay, sitemaps) and the site's sitemaps;
2. fetches pages from a priority frontier, following links found on *every*
   page (not only the homepage) up to ``max_pages`` / ``max_depth``;
3. renders JavaScript pages through one shared Playwright browser when the
   static HTML is weak (``SCRAPER_RENDER_MODE=auto``);
4. skips exact, canonical and near-duplicate pages (e.g. one product in
   several colours), and pages outside the site or blocked by robots.txt;
5. removes cross-page boilerplate and keeps it once as a ``site_wide`` page;
6. returns pages plus a crawl report explaining every skipped/failed URL.

Crawl scope comes from the submitted URL:

* ``https://site.com``            -> ``site``: the whole site
* ``https://site.com/docs/``      -> ``section``: only URLs under ``/docs/``
* ``https://site.com/blog/post``  -> ``page``: that page plus a few related pages
"""

import asyncio
import heapq
import ipaddress
import itertools
import socket
import ssl
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Tuple
from urllib.parse import urlparse

import aiohttp
import certifi

from ..core.config import get_settings
from .boilerplate import build_site_wide_page, remove_cross_page_boilerplate
from .browser_renderer import BrowserSession, get_scraper_render_mode
from .knowledge_export import page_to_markdown
from .page_fetcher import DEFAULT_HEADERS, as_fetch_result, fetch_page_with_retry
from .robots import RobotsRules, parse_robots_txt
from .text_fingerprint import hamming_distance
from .url_discovery import (
    CRAWL_INTENT_SPECIFIC,
    _canonical_domain,
    classify_page_type,
    extract_homepage_link_records,
    fetch_sitemap_urls,
    is_document_url,
    is_utility_or_meta_url,
    is_valid_crawl_url,
    normalize_discovered_url,
    score_specific_url_priority,
    score_url_priority,
    url_key,
)

SCOPE_SITE = "site"
SCOPE_SECTION = "section"
SCOPE_PAGE = "page"
MAX_REPORTED_URLS = 50
MAX_CRAWL_DELAY_SECONDS = 10.0
MIN_TEXT_FOR_NEAR_DUPLICATE = 300

ExtractFn = Callable[..., Awaitable[Tuple[Dict | None, str, str, str]]]
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


@dataclass
class CrawlConfig:
    max_pages: int = 100
    max_depth: int = 5
    concurrency: int = 4
    politeness_delay: float = 0.25
    respect_robots: bool = True
    include_subdomains: bool = False
    include_documents: bool = True
    max_documents: int = 10
    specific_page_max_pages: int = 5
    near_duplicate_distance: int = 3
    boilerplate_min_pages: int = 4
    boilerplate_ratio: float = 0.5
    time_budget_seconds: float = 900.0
    allow_private_hosts: bool = False
    max_sitemap_urls: int = 500
    max_path_depth: int = 8

    @classmethod
    def from_settings(cls, **overrides: Any) -> "CrawlConfig":
        settings = get_settings()
        config = cls(
            max_pages=settings.crawler_max_pages,
            max_depth=settings.crawler_max_depth,
            concurrency=settings.crawler_concurrency,
            politeness_delay=max(0, settings.crawler_politeness_delay_ms) / 1000,
            respect_robots=settings.crawler_respect_robots,
            include_subdomains=settings.crawler_include_subdomains,
            include_documents=settings.crawler_include_documents,
            max_documents=settings.crawler_max_documents,
            specific_page_max_pages=settings.crawler_specific_page_max_pages,
            near_duplicate_distance=settings.crawler_near_duplicate_distance,
            boilerplate_min_pages=settings.crawler_boilerplate_min_pages,
            boilerplate_ratio=settings.crawler_boilerplate_ratio,
            time_budget_seconds=settings.crawler_time_budget_seconds,
            allow_private_hosts=settings.crawler_allow_private_hosts,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        config.max_pages = max(1, int(config.max_pages))
        config.max_depth = max(0, int(config.max_depth))
        config.concurrency = max(1, int(config.concurrency))
        return config


@dataclass(order=True)
class _FrontierItem:
    sort_key: Tuple[int, int, int]
    url: str = field(compare=False)
    depth: int = field(compare=False)
    source: str = field(compare=False)


def detect_crawl_scope(raw_url: str) -> Tuple[str, str]:
    """Return (scope, path_prefix) for a submitted URL."""
    parsed = urlparse(raw_url if "://" in raw_url else f"https://{raw_url}")
    path = parsed.path or "/"
    if path == "/" and not parsed.query:
        return SCOPE_SITE, ""
    if path.endswith("/") and not parsed.query:
        return SCOPE_SECTION, path.rstrip("/")
    return SCOPE_PAGE, ""


def _site_domain(url: str) -> str:
    """Site identity used for scope checks: host[:port] without www/default ports."""
    return _canonical_domain(urlparse(url).netloc.split("@")[-1])


async def is_public_host(hostname: str) -> bool:
    """Reject localhost, private, link-local and reserved addresses (SSRF guard)."""
    host = (hostname or "").strip("[]").lower()
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except OSError:
            return True  # unresolvable: let the fetch fail with a normal error
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue
    return all(
        not (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved
             or address.is_multicast or address.is_unspecified)
        for address in addresses
    )


class SiteCrawler:
    def __init__(
        self,
        start_url: str,
        extract_fn: ExtractFn,
        config: CrawlConfig | None = None,
        fetch_fn: Callable[..., Awaitable[Any]] | None = None,
        sitemap_fn: Callable[..., Awaitable[List[str]]] | None = None,
    ):
        raw = (start_url or "").strip()
        if not raw.startswith(("http://", "https://")):
            raw = "https://" + raw
        self.raw_start_url = raw
        self.config = config or CrawlConfig.from_settings()
        self.scope, self.path_prefix = detect_crawl_scope(raw)
        self.start_url = normalize_discovered_url(raw) or raw
        self.base_domain = _site_domain(self.start_url)
        self.extract_fn = extract_fn
        self.fetch_fn = fetch_fn or fetch_page_with_retry
        self.sitemap_fn = sitemap_fn or fetch_sitemap_urls
        self.robots = RobotsRules()
        self.robots_text = ""

        if self.scope == SCOPE_PAGE:
            self.max_pages = min(self.config.max_pages, self.config.specific_page_max_pages)
            self.max_depth = min(self.config.max_depth, 1)
        else:
            self.max_pages = self.config.max_pages
            self.max_depth = self.config.max_depth

        self._frontier: List[_FrontierItem] = []
        self._counter = itertools.count()
        self._seen_keys: set[str] = set()
        self._stored_keys: Dict[str, int] = {}
        self._content_hashes: Dict[str, str] = {}
        self._simhashes: List[Tuple[str, str]] = []
        self._public_hosts: Dict[str, bool] = {}
        self._last_request = 0.0
        self._throttle_lock = asyncio.Lock()
        self.pages: List[Dict[str, Any]] = []
        self.report: Dict[str, Any] = {
            "scope": self.scope,
            "path_prefix": self.path_prefix,
            "start_url": self.start_url,
            "render_mode": get_scraper_render_mode(),
            "pages_stored": 0,
            "documents_stored": 0,
            "pages_attempted": 0,
            "sitemap_urls": 0,
            "discovered_urls": 0,
            "duplicates": [],
            "failed": [],
            "skipped": {
                "robots_txt": 0,
                "out_of_scope": 0,
                "page_budget": 0,
                "depth_limit": 0,
                "document_limit": 0,
                "private_host": 0,
                "time_budget": 0,
            },
        }
        self.errors: List[str] = []

    # -- frontier -----------------------------------------------------------

    def _score(self, url: str, depth: int, anchor_text: str = "") -> int:
        if self.scope == SCOPE_PAGE:
            score = score_specific_url_priority(url, self.start_url, anchor_text=anchor_text)
        else:
            score = score_url_priority(url)
        if is_document_url(url):
            score -= 150
        return score - depth * 60

    def _in_scope(self, url: str) -> bool:
        if not is_valid_crawl_url(
            url,
            self.base_domain,
            allow_documents=self.config.include_documents,
            include_subdomains=self.config.include_subdomains,
            max_path_depth=self.config.max_path_depth,
        ):
            return False
        if self.scope == SCOPE_SECTION:
            path = urlparse(url).path.rstrip("/")
            if path != self.path_prefix and not path.startswith(self.path_prefix + "/"):
                return False
        if self.scope == SCOPE_PAGE and is_utility_or_meta_url(url, self.start_url, CRAWL_INTENT_SPECIFIC):
            return False
        return True

    def _enqueue(self, url: str, depth: int, source: str, anchor_text: str = "") -> None:
        normalized = normalize_discovered_url(url)
        key = url_key(normalized)
        if not key or key in self._seen_keys:
            return
        self._seen_keys.add(key)
        if depth > self.max_depth:
            self.report["skipped"]["depth_limit"] += 1
            return
        if not self._in_scope(normalized):
            self.report["skipped"]["out_of_scope"] += 1
            return
        if self.config.respect_robots and not self.robots.can_fetch(normalized):
            self.report["skipped"]["robots_txt"] += 1
            return
        score = self._score(normalized, depth, anchor_text)
        heapq.heappush(
            self._frontier,
            _FrontierItem(sort_key=(-score, depth, next(self._counter)), url=normalized, depth=depth, source=source),
        )

    def _enqueue_links(self, html: str, page_url: str, depth: int) -> None:
        if not html or depth + 1 > self.max_depth:
            return
        for record in extract_homepage_link_records(html, page_url):
            self._enqueue(record["url"], depth + 1, page_url, record.get("anchor_text", ""))

    # -- politeness & safety --------------------------------------------------

    async def _throttle(self) -> None:
        delay = self.config.politeness_delay
        if self.robots.crawl_delay:
            delay = max(delay, min(self.robots.crawl_delay, MAX_CRAWL_DELAY_SECONDS))
        if delay <= 0:
            return
        async with self._throttle_lock:
            wait = self._last_request + delay - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

    async def _host_allowed(self, url: str) -> bool:
        if self.config.allow_private_hosts:
            return True
        host = urlparse(url).hostname or ""
        if host not in self._public_hosts:
            self._public_hosts[host] = await is_public_host(host)
        return self._public_hosts[host]

    async def _load_robots(self, session) -> None:
        parsed = urlparse(self.start_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            result = as_fetch_result(await self.fetch_fn(session, robots_url, retries=1), robots_url)
            self.robots_text = result.html if 200 <= int(getattr(result, "status", 200) or 200) < 300 else ""
        except Exception:
            self.robots_text = ""
        self.robots = parse_robots_txt(self.robots_text) if self.robots_text else RobotsRules()
        self.report["robots"] = {
            "found": bool(self.robots_text),
            "disallow_rules": self.robots.disallow_count,
            "crawl_delay": self.robots.crawl_delay,
            "sitemaps": self.robots.sitemaps[:10],
        }

    # -- page handling ------------------------------------------------------

    def _record_failure(self, url: str, error: str) -> None:
        self.errors.append(f"{url}: {error}")
        if len(self.report["failed"]) < MAX_REPORTED_URLS:
            self.report["failed"].append({"url": url, "error": error})

    def _record_duplicate(self, url: str, duplicate_of: str, reason: str) -> None:
        if len(self.report["duplicates"]) < MAX_REPORTED_URLS:
            self.report["duplicates"].append({"url": url, "duplicate_of": duplicate_of, "reason": reason})
        self.report.setdefault("duplicate_count", 0)
        self.report["duplicate_count"] += 1

    def _duplicate_of(self, page: Dict[str, Any], effective_url: str) -> Tuple[str, str]:
        key = url_key(effective_url)
        if key in self._stored_keys:
            return self.pages[self._stored_keys[key]]["page_url"], "same_url_after_redirect"
        canonical = ((page.get("metadata") or {}).get("canonical") or "").strip()
        if canonical:
            canonical_key = url_key(canonical)
            if canonical_key != key and canonical_key in self._stored_keys:
                return self.pages[self._stored_keys[canonical_key]]["page_url"], "canonical_duplicate"
        quality = page.get("quality") or {}
        digest = quality.get("content_hash") or ""
        core_length = int(quality.get("core_text_length") or 0)
        if digest and core_length >= 50 and digest in self._content_hashes:
            return self._content_hashes[digest], "identical_content"
        fingerprint = quality.get("simhash") or ""
        if fingerprint and core_length >= MIN_TEXT_FOR_NEAR_DUPLICATE:
            for stored_url, stored_fingerprint in self._simhashes:
                if hamming_distance(fingerprint, stored_fingerprint) <= self.config.near_duplicate_distance:
                    return stored_url, "near_duplicate_content"
        return "", ""

    def _store(self, page: Dict[str, Any], effective_url: str, depth: int, source: str) -> None:
        page["crawl"] = {"depth": depth, "discovered_from": source}
        index = len(self.pages)
        self.pages.append(page)
        for candidate in {url_key(effective_url), url_key(page.get("page_url", ""))}:
            if candidate:
                self._stored_keys[candidate] = index
                self._seen_keys.add(candidate)
        canonical = ((page.get("metadata") or {}).get("canonical") or "").strip()
        if canonical and _site_domain(canonical) == self.base_domain:
            self._seen_keys.add(url_key(canonical))
        quality = page.get("quality") or {}
        if quality.get("content_hash"):
            self._content_hashes.setdefault(quality["content_hash"], page["page_url"])
        if quality.get("simhash") and int(quality.get("core_text_length") or 0) >= MIN_TEXT_FOR_NEAR_DUPLICATE:
            self._simhashes.append((page["page_url"], quality["simhash"]))
        if page.get("page_type") == "document":
            self.report["documents_stored"] += 1
        self.report["pages_stored"] = len(self.pages)

    async def _fetch_one(self, session, item: _FrontierItem):
        await self._throttle()
        page_type = "homepage" if item.depth == 0 and self.scope == SCOPE_SITE else classify_page_type(item.url)
        try:
            return item, await self.extract_fn(
                session,
                item.url,
                page_type=page_type,
                allow_http_fallback=item.depth == 0,
            )
        except Exception as exc:  # one bad page must not stop the crawl
            return item, (None, "", item.url, f"error: {str(exc)[:120]}")

    def _handle_result(self, item: _FrontierItem, result) -> None:
        page, html, effective_url, error = result
        effective_url = effective_url or item.url
        if not page:
            self._record_failure(item.url, error or "no content extracted")
            return
        if item.depth == 0:
            # Follow the homepage's redirect (http->https, www, regional domain).
            self.base_domain = _site_domain(effective_url)
            self.start_url = normalize_discovered_url(effective_url) or self.start_url
            self.report["final_base_url"] = self.start_url
        elif not is_valid_crawl_url(
            effective_url, self.base_domain,
            allow_documents=True, include_subdomains=self.config.include_subdomains, max_path_depth=99,
        ):
            self._record_failure(item.url, f"redirected off-site to {effective_url}")
            return

        duplicate_of, reason = self._duplicate_of(page, effective_url)
        if duplicate_of:
            self._record_duplicate(item.url, duplicate_of, reason)
        else:
            self._store(page, effective_url, item.depth, item.source)
        # Duplicates can still link to new pages.
        if self.scope != SCOPE_PAGE or item.depth == 0:
            self._enqueue_links(html, effective_url, item.depth)

    # -- main loop ----------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        started = time.monotonic()
        scraped_at = datetime.now().isoformat()
        print(f"🌐 Crawling {self.start_url} (scope={self.scope}, max_pages={self.max_pages}, max_depth={self.max_depth})")

        if not await self._host_allowed(self.start_url):
            self.report["skipped"]["private_host"] += 1
            self._record_failure(self.start_url, "refusing to crawl a private/local network address")
            return self._result(scraped_at, started)

        connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT, limit_per_host=max(2, self.config.concurrency))
        async with aiohttp.ClientSession(connector=connector, headers=DEFAULT_HEADERS) as session:
            async with BrowserSession() as browser:
                if self.config.respect_robots:
                    await self._load_robots(session)
                    if not self.robots.can_fetch(self.start_url):
                        self.report["skipped"]["robots_txt"] += 1
                        self._record_failure(self.start_url, "blocked by robots.txt (set CRAWLER_RESPECT_ROBOTS=false only for sites you own)")
                        return self._result(scraped_at, started, browser)

                # The submitted URL always goes first.
                start_key = url_key(self.start_url)
                self._seen_keys.add(start_key)
                start_item = _FrontierItem(sort_key=(-10**9, 0, next(self._counter)), url=self.start_url, depth=0, source="start_url")
                self.report["pages_attempted"] += 1
                self._handle_result(*await self._fetch_one(session, start_item))
                if not self.pages:
                    return self._result(scraped_at, started, browser)

                if self.scope != SCOPE_PAGE and self.max_depth >= 1:
                    try:
                        sitemap_urls = await self.sitemap_fn(
                            session,
                            self.start_url,
                            robots_text=self.robots_text,
                            max_urls=self.config.max_sitemap_urls,
                            headers=DEFAULT_HEADERS,
                        )
                    except Exception as exc:
                        sitemap_urls = []
                        self.errors.append(f"sitemap discovery failed: {str(exc)[:80]}")
                    self.report["sitemap_urls"] = len(sitemap_urls)
                    for sitemap_url in sitemap_urls:
                        self._enqueue(sitemap_url, 1, "sitemap")

                await self._crawl_frontier(session, started)
                self.report["render"] = {
                    "pages_rendered": browser.pages_rendered,
                    "browser_error": browser.launch_error,
                }
                return self._result(scraped_at, started, browser)

    async def _crawl_frontier(self, session, started: float) -> None:
        pending: set[asyncio.Task] = set()
        max_attempts = self.max_pages * 3
        while True:
            while self._frontier and len(pending) < self.config.concurrency:
                if len(self.pages) + len(pending) >= self.max_pages:
                    break
                if self.report["pages_attempted"] >= max_attempts:
                    break
                if time.monotonic() - started > self.config.time_budget_seconds:
                    self.report["skipped"]["time_budget"] += len(self._frontier)
                    self._frontier.clear()
                    break
                item = heapq.heappop(self._frontier)
                if is_document_url(item.url):
                    documents_in_flight = sum(1 for page in self.pages if page.get("page_type") == "document")
                    if documents_in_flight >= self.config.max_documents:
                        self.report["skipped"]["document_limit"] += 1
                        continue
                if not await self._host_allowed(item.url):
                    self.report["skipped"]["private_host"] += 1
                    continue
                self.report["pages_attempted"] += 1
                pending.add(asyncio.create_task(self._fetch_one(session, item)))
            if not pending:
                break
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                self._handle_result(*task.result())
        self.report["skipped"]["page_budget"] += len(self._frontier)

    def _result(self, scraped_at: str, started: float, browser: BrowserSession | None = None) -> Dict[str, Any]:
        boilerplate = remove_cross_page_boilerplate(
            self.pages,
            min_pages=self.config.boilerplate_min_pages,
            ratio=self.config.boilerplate_ratio,
        )
        site_wide_blocks = boilerplate.pop("site_wide_blocks", [])
        # Pages whose only text was template (e.g. a JS page whose render
        # failed and whose static HTML is just the menu) carry no knowledge.
        empty_pages = [
            page for page in self.pages
            if not (page.get("content") or "").strip() and not page.get("faqs") and not page.get("tables")
        ]
        if empty_pages and len(empty_pages) < len(self.pages):
            for page in empty_pages:
                self.errors.append(f"{page.get('page_url')}: no page-specific text after boilerplate removal")
            self.report["empty_pages_dropped"] = [page.get("page_url") for page in empty_pages][:MAX_REPORTED_URLS]
            empty_ids = {id(page) for page in empty_pages}
            self.pages = [page for page in self.pages if id(page) not in empty_ids]
            self.report["pages_stored"] = len(self.pages)
            self.report["documents_stored"] = sum(1 for page in self.pages if page.get("page_type") == "document")
        pages = list(self.pages)
        if pages:
            site_wide = build_site_wide_page(pages, site_wide_blocks, self.start_url)
            if site_wide is not None:
                pages.append(site_wide)
        for page in pages:
            page["_markdown"] = page_to_markdown(page)

        self.report["boilerplate"] = boilerplate
        self.report["discovered_urls"] = len(self._seen_keys)
        self.report["frontier_remaining"] = len(self._frontier)
        self.report["duration_seconds"] = round(time.monotonic() - started, 2)
        self.report["config"] = asdict(self.config)
        if browser is not None and "render" not in self.report:
            self.report["render"] = {"pages_rendered": browser.pages_rendered, "browser_error": browser.launch_error}
        print(
            f"  🎉 Crawl done: {self.report['pages_stored']} pages stored, "
            f"{self.report.get('duplicate_count', 0)} duplicates, {len(self.report['failed'])} failed, "
            f"{self.report['skipped']['page_budget']} left in frontier ({self.report['duration_seconds']}s)"
        )
        return {
            "source_url": self.start_url,
            "scraped_at": scraped_at,
            "pages": pages,
            "total_pages": len(self.pages),  # real pages; the site_wide record is extra
            "success": bool(self.pages),
            "errors": self.errors[:MAX_REPORTED_URLS],
            "crawl_report": self.report,
        }


async def crawl_website(
    url: str,
    extract_fn: ExtractFn,
    config: CrawlConfig | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return await SiteCrawler(url, extract_fn=extract_fn, config=config, **kwargs).run()
