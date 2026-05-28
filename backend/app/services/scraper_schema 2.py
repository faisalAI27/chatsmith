import hashlib
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}


def normalize_url_for_cache(url: str) -> str:
    """Normalize a URL into a stable cache and website identity."""
    raw_url = (url or "").strip()
    if not raw_url:
        return ""

    if not urlparse(raw_url).scheme:
        raw_url = f"https://{raw_url}"

    parsed = urlparse(raw_url)
    scheme = (parsed.scheme or "https").lower()
    if scheme in {"http", "https"}:
        scheme = "https"

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


def make_website_id(normalized_url: str) -> str:
    """Create the stable website id shared by JSON, chunks, and future stores."""
    return hashlib.md5((normalized_url or "").encode("utf-8")).hexdigest()[:12]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Optional[List[Any]]) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _page_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "canonical": "",
        "open_graph": {},
        "twitter": {},
    }
    if isinstance(metadata, dict):
        base.update(metadata)
    if not isinstance(base.get("open_graph"), dict):
        base["open_graph"] = {}
    if not isinstance(base.get("twitter"), dict):
        base["twitter"] = {}
    return base


def create_quality_record(
    status_code: int = 200,
    content_length: int = 0,
    useful_text_score: float = 0.0,
    image_count: int = 0,
    link_count: int = 0,
    scraped_at: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a per-page extraction quality record."""
    return {
        "status_code": int(status_code or 0),
        "content_length": int(content_length or 0),
        "useful_text_score": float(useful_text_score or 0.0),
        "image_count": int(image_count or 0),
        "link_count": int(link_count or 0),
        "scraped_at": scraped_at or _utc_now(),
        "errors": _as_list(errors),
    }


def create_page_record(
    page_url: str,
    page_type: str = "other",
    title: str = "",
    description: str = "",
    extraction_method: str = "static",
    headings: Optional[List[Any]] = None,
    sections: Optional[List[Any]] = None,
    paragraphs: Optional[List[Any]] = None,
    lists: Optional[List[Any]] = None,
    faqs: Optional[List[Any]] = None,
    tables: Optional[List[Any]] = None,
    images: Optional[List[Any]] = None,
    links: Optional[List[Any]] = None,
    structured_data: Optional[List[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    quality: Optional[Dict[str, Any]] = None,
    content: str = "",
) -> Dict[str, Any]:
    """Create a v2 page record without performing extraction or crawling."""
    image_records = _as_list(images)
    link_records = _as_list(links)
    quality_record = create_quality_record(
        content_length=len(content or ""),
        image_count=len(image_records),
        link_count=len(link_records),
    )
    if isinstance(quality, dict):
        quality_record.update(quality)
        quality_record["errors"] = _as_list(quality_record.get("errors"))

    return {
        "page_url": page_url or "",
        "normalized_page_url": normalize_url_for_cache(page_url),
        "page_type": page_type or "other",
        "title": title or "",
        "description": description or "",
        "extraction_method": extraction_method or "static",
        "headings": _as_list(headings),
        "sections": _as_list(sections),
        "paragraphs": _as_list(paragraphs),
        "lists": _as_list(lists),
        "faqs": _as_list(faqs),
        "tables": _as_list(tables),
        "images": image_records,
        "links": link_records,
        "structured_data": _as_list(structured_data),
        "metadata": _page_metadata(metadata),
        "quality": quality_record,
        "content": content or "",
    }


def convert_v2_page_to_legacy_page(page: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a v2 page record into the legacy page shape used by chat context."""
    if not isinstance(page, dict):
        page = {}
    return {
        "title": page.get("title", "") or "",
        "description": page.get("description", "") or "",
        "sections": _as_list(page.get("sections")),
        "content": page.get("content", "") or "",
        "url": page.get("page_url") or page.get("url", "") or "",
        "page_type": page.get("page_type", "") or "other",
    }


def _legacy_page_to_v2_page(page: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(page, dict):
        page = {}
    return create_page_record(
        page_url=page.get("url", "") or page.get("page_url", ""),
        page_type=page.get("page_type", "other") or "other",
        title=page.get("title", "") or "",
        description=page.get("description", "") or "",
        sections=_as_list(page.get("sections")),
        content=page.get("content", "") or "",
    )


def ensure_v2_knowledge_shape(
    knowledge: Dict[str, Any],
    fallback_url: str = "",
) -> Dict[str, Any]:
    """
    Return a v2-compatible knowledge object while preserving legacy fields.

    Old knowledge files can be missing website_id, normalized_url, scraping_version,
    or top-level pages. The current chatbot still reads primary_content.pages, so
    that legacy path is always preserved or rebuilt from v2 pages.
    """
    if not isinstance(knowledge, dict):
        raise ValueError("Knowledge JSON must be an object")

    result = deepcopy(knowledge)
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    result["metadata"] = metadata

    primary_content = result.get("primary_content")
    if not isinstance(primary_content, dict):
        primary_content = {}
    primary_content.setdefault("source", "website_scraping")
    primary_content.setdefault("reliability", "high")

    legacy_pages = primary_content.get("pages")
    if not isinstance(legacy_pages, list):
        legacy_pages = []
    primary_content["pages"] = legacy_pages
    result["primary_content"] = primary_content

    secondary_content = result.get("secondary_content")
    if not isinstance(secondary_content, dict):
        secondary_content = {}
    secondary_content.setdefault("source", "web_search")
    secondary_content.setdefault("reliability", "medium")
    searches = secondary_content.get("searches")
    if not isinstance(searches, list):
        searches = []
    secondary_content["searches"] = searches
    result["secondary_content"] = secondary_content

    pages = result.get("pages")
    if isinstance(pages, list) and pages:
        v2_pages = [
            page if isinstance(page, dict) and "page_url" in page else _legacy_page_to_v2_page(page)
            for page in pages
        ]
    else:
        v2_pages = [_legacy_page_to_v2_page(page) for page in legacy_pages]
    result["pages"] = v2_pages

    if not legacy_pages and v2_pages:
        primary_content["pages"] = [convert_v2_page_to_legacy_page(page) for page in v2_pages]
        legacy_pages = primary_content["pages"]

    original_url = metadata.get("url") or fallback_url or result.get("source_url", "") or ""
    normalized_url = metadata.get("normalized_url") or normalize_url_for_cache(original_url)
    metadata["url"] = original_url
    metadata["normalized_url"] = normalized_url
    metadata["website_id"] = metadata.get("website_id") or make_website_id(normalized_url)
    metadata["name"] = metadata.get("name", "") or ""
    metadata["created_at"] = metadata.get("created_at") or _utc_now()
    metadata["scraping_version"] = metadata.get("scraping_version") or "v2"
    metadata["pages_scraped"] = int(metadata.get("pages_scraped") or len(legacy_pages) or len(v2_pages))
    metadata["has_web_search_supplement"] = bool(
        metadata.get("has_web_search_supplement", False) or searches
    )

    return result
