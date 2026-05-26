import re
import xml.etree.ElementTree as ET
from collections import deque
from typing import Dict, Iterable, List
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup


HIGH_PRIORITY_KEYWORDS = {
    "about",
    "contact",
    "docs",
    "documentation",
    "faq",
    "features",
    "pricing",
    "products",
    "services",
    "solutions",
}
MEDIUM_PRIORITY_KEYWORDS = {
    "blog",
    "careers",
    "case-studies",
    "customers",
    "help",
    "news",
    "resources",
    "team",
}
LOW_PRIORITY_KEYWORDS = {
    "archive",
    "category",
    "legal",
    "privacy",
    "tag",
    "terms",
}
SEARCH_QUERY_KEYS = {
    "keyword",
    "q",
    "query",
    "s",
    "search",
    "term",
}
SEARCH_PATH_SEGMENTS = {
    "find",
    "lookup",
    "results",
    "search",
    "search-results",
    "search_results",
}
SKIP_PATH_KEYWORDS = {
    "account",
    "cart",
    "checkout",
    "login",
    "password",
    "register",
    "signin",
    "signup",
}
DOWNLOAD_EXTENSIONS = {
    ".avi",
    ".doc",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".svg",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}
COMMON_SITEMAP_PATHS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
)
MAX_SITEMAP_FILES = 5
MAX_SITEMAP_URLS = 100


def _canonical_domain(netloc: str) -> str:
    domain = (netloc or "").lower()
    if domain.endswith(":443"):
        domain = domain[:-4]
    elif domain.endswith(":80"):
        domain = domain[:-3]
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _path_tokens(url: str) -> List[str]:
    parsed = urlparse(url)
    return [token for token in re.split(r"[/_-]+", parsed.path.lower()) if token]


def _path_segments(path: str) -> List[str]:
    return [segment for segment in (path or "").lower().split("/") if segment]


def _has_any_keyword(url: str, keywords: Iterable[str]) -> bool:
    path_text = re.sub(r"[/_-]+", " ", urlparse(url).path.lower())
    compact_path = re.sub(r"[^a-z0-9]+", "", path_text)
    for keyword in keywords:
        keyword_text = re.sub(r"[-_]+", " ", keyword.lower())
        compact_keyword = re.sub(r"[^a-z0-9]+", "", keyword_text)
        if keyword_text in path_text or compact_keyword in compact_path:
            return True
    return False


def _query_keys(query: str) -> set[str]:
    return {key.lower() for key, _ in parse_qsl(query or "", keep_blank_values=True)}


def _has_search_query(parsed_url) -> bool:
    return bool(_query_keys(parsed_url.query) & SEARCH_QUERY_KEYS)


def _is_search_result_path(parsed_url) -> bool:
    return any(segment in SEARCH_PATH_SEGMENTS for segment in _path_segments(parsed_url.path))


def _is_root_path(parsed_url) -> bool:
    return not parsed_url.path or parsed_url.path == "/"


def normalize_discovered_url(url: str) -> str:
    """Normalize a crawl candidate while preserving meaningful path/query identity."""
    raw_url = (url or "").strip()
    if not raw_url:
        return ""
    if not urlparse(raw_url).scheme:
        raw_url = f"https://{raw_url}"

    parsed = urlparse(raw_url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if netloc.endswith(":443"):
        netloc = netloc[:-4]
    elif netloc.endswith(":80"):
        netloc = netloc[:-3]

    path = parsed.path or ""
    if path == "/":
        path = ""
    else:
        path = path.rstrip("/")

    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_lower = key.lower()
        if key_lower.startswith("utm_") or key_lower in TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items), doseq=True)

    return urlunparse((scheme, netloc, path, "", query, ""))


def is_valid_crawl_url(url: str, base_domain: str) -> bool:
    """Return True when a URL is useful and safe enough for same-site crawling."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if _canonical_domain(parsed.netloc) != _canonical_domain(base_domain):
        return False

    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in DOWNLOAD_EXTENSIONS):
        return False
    if any(keyword in path_lower for keyword in SKIP_PATH_KEYWORDS):
        return False
    if _is_search_result_path(parsed) or _has_search_query(parsed):
        return False

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    if parsed.query and _is_root_path(parsed):
        return False

    depth = len([part for part in parsed.path.split("/") if part])
    is_priority = _has_any_keyword(url, HIGH_PRIORITY_KEYWORDS | MEDIUM_PRIORITY_KEYWORDS)
    if len(query_items) > 2 and not is_priority:
        return False
    if depth > 4 and not is_priority:
        return False

    return True


def classify_page_type(url: str) -> str:
    """Classify a URL into a coarse page type for extraction metadata."""
    parsed = urlparse(url)
    tokens = set(_path_tokens(url))
    if _is_search_result_path(parsed) or _has_search_query(parsed):
        return "search_result"
    if _is_root_path(parsed) and parsed.query:
        return "query"
    if _is_root_path(parsed):
        return "homepage"
    if {"about", "about-us", "who-we-are"} & tokens:
        return "about"
    if {"service", "services"} & tokens:
        return "services"
    if {"product", "products"} & tokens:
        return "products"
    if "pricing" in tokens or "plans" in tokens:
        return "pricing"
    if {"doc", "docs", "documentation"} & tokens:
        return "docs"
    if {"faq", "faqs"} & tokens:
        return "faq"
    if "contact" in tokens:
        return "contact"
    if "blog" in tokens:
        return "blog"
    if "team" in tokens:
        return "team"
    if "careers" in tokens or "jobs" in tokens:
        return "careers"
    if {"privacy", "terms", "legal"} & tokens:
        return "legal"
    return "other"


def score_url_priority(url: str) -> int:
    """Score higher-value pages ahead of noisy or low-value pages."""
    parsed = urlparse(url)
    page_type = classify_page_type(url)
    if page_type == "homepage":
        score = 1000
    elif page_type in {"about", "services", "products", "pricing", "faq", "contact", "docs"}:
        score = 900
    elif page_type in {"blog", "team", "careers"} or _has_any_keyword(url, MEDIUM_PRIORITY_KEYWORDS):
        score = 600
    elif page_type == "legal" or _has_any_keyword(url, LOW_PRIORITY_KEYWORDS):
        score = 100
    elif page_type in {"search_result", "query"}:
        score = 10
    else:
        score = 300

    depth = len([part for part in parsed.path.split("/") if part])
    score += max(0, 30 - depth * 5)
    if _has_any_keyword(url, HIGH_PRIORITY_KEYWORDS):
        score += 50
    if parsed.query:
        score -= 200
    if page_type == "search_result":
        score -= 300
    return max(0, score)


def extract_sitemap_urls_from_robots(robots_text: str) -> List[str]:
    """Extract Sitemap directives from robots.txt text."""
    sitemap_urls = []
    for line in (robots_text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip().lower() == "sitemap":
            sitemap_url = value.strip()
            if sitemap_url:
                sitemap_urls.append(sitemap_url)
    return sitemap_urls


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_sitemap_xml(xml_text: str) -> Dict[str, List[str]]:
    """Parse a sitemap or sitemap index XML string."""
    try:
        root = ET.fromstring((xml_text or "").strip())
    except ET.ParseError:
        return {"urls": [], "sitemaps": []}

    urls = []
    sitemaps = []
    for child in root:
        child_type = _xml_local_name(child.tag)
        loc = ""
        for node in child:
            if _xml_local_name(node.tag) == "loc" and node.text:
                loc = node.text.strip()
                break
        if not loc:
            continue
        if child_type == "url":
            urls.append(loc)
        elif child_type == "sitemap":
            sitemaps.append(loc)
    return {"urls": urls, "sitemaps": sitemaps}


async def fetch_sitemap_urls(
    session,
    base_url: str,
    robots_text: str = "",
    max_sitemap_files: int = MAX_SITEMAP_FILES,
    max_urls: int = MAX_SITEMAP_URLS,
    headers: Dict[str, str] | None = None,
) -> List[str]:
    """Fetch sitemap URLs from robots.txt directives and common sitemap paths."""
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    initial_sitemaps = extract_sitemap_urls_from_robots(robots_text)
    initial_sitemaps.extend(urljoin(origin, path) for path in COMMON_SITEMAP_PATHS)

    queue = deque(normalize_discovered_url(url) for url in initial_sitemaps if url)
    seen_sitemaps = set()
    seen_urls = set()
    urls = []

    while queue and len(seen_sitemaps) < max_sitemap_files and len(urls) < max_urls:
        sitemap_url = queue.popleft()
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)
        try:
            async with session.get(
                sitemap_url,
                headers=headers or {},
                timeout=aiohttp.ClientTimeout(total=10),
                allow_redirects=True,
            ) as response:
                if response.status != 200:
                    continue
                parsed_xml = parse_sitemap_xml(await response.text())
        except Exception:
            continue

        for nested_sitemap in parsed_xml["sitemaps"]:
            normalized_sitemap = normalize_discovered_url(nested_sitemap)
            if normalized_sitemap and normalized_sitemap not in seen_sitemaps:
                queue.append(normalized_sitemap)

        for page_url in parsed_xml["urls"]:
            normalized_url = normalize_discovered_url(page_url)
            if normalized_url and normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                urls.append(normalized_url)
                if len(urls) >= max_urls:
                    break

    return urls


def extract_homepage_links(homepage_html: str, base_url: str) -> List[str]:
    """Extract normalized same-page candidates from homepage anchors."""
    if not homepage_html:
        return []
    soup = BeautifulSoup(homepage_html, "lxml")
    links = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        normalized_url = normalize_discovered_url(urljoin(base_url, href))
        if normalized_url and normalized_url not in seen:
            seen.add(normalized_url)
            links.append(normalized_url)
    return links


def select_candidate_urls(
    base_url: str,
    urls: Iterable[str],
    max_pages: int,
    include_homepage: bool = False,
) -> List[str]:
    """Normalize, filter, deduplicate, rank, and limit crawl candidates."""
    base = normalize_discovered_url(base_url)
    base_domain = urlparse(base).netloc
    seen = set()
    scored = []

    for url in urls:
        normalized_url = normalize_discovered_url(url)
        if not normalized_url:
            continue
        if normalized_url == base and not include_homepage:
            continue
        if not is_valid_crawl_url(normalized_url, base_domain):
            continue
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        depth = len([part for part in urlparse(normalized_url).path.split("/") if part])
        scored.append((score_url_priority(normalized_url), -depth, normalized_url))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [url for _, _, url in scored[:max_pages]]


def discover_candidate_urls(
    homepage_html: str,
    base_url: str,
    sitemap_urls: Iterable[str] | None = None,
    max_pages: int = 10,
    include_homepage: bool = False,
) -> List[str]:
    """Discover final candidate URLs from homepage links plus sitemap URLs."""
    candidates = extract_homepage_links(homepage_html, base_url)
    candidates.extend(sitemap_urls or [])
    return select_candidate_urls(
        base_url=base_url,
        urls=candidates,
        max_pages=max_pages,
        include_homepage=include_homepage,
    )
