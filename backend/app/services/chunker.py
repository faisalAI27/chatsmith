import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

from .scraper_schema import ensure_v2_knowledge_shape, make_website_id, normalize_url_for_cache


MAX_CHUNK_WORDS = 1000
OVERLAP_WORDS = 120
MIN_CHUNK_WORDS = 4

SOCIAL_PLATFORMS = {
    "instagram": ("instagram.com",),
    "facebook": ("facebook.com", "fb.com"),
    "youtube": ("youtube.com", "youtu.be"),
    "tiktok": ("tiktok.com",),
    "twitter": ("twitter.com", "x.com"),
    "linkedin": ("linkedin.com",),
    "pinterest": ("pinterest.com",),
    "whatsapp": ("wa.me", "whatsapp.com"),
}
SOCIAL_PLATFORM_LABELS = {
    "instagram": "Instagram",
    "facebook": "Facebook",
    "youtube": "YouTube",
    "tiktok": "TikTok",
    "twitter": "Twitter/X",
    "linkedin": "LinkedIn",
    "pinterest": "Pinterest",
    "whatsapp": "WhatsApp",
}

URL_RE = re.compile(r"https?://[^\s\"'<>)}\]]+", re.IGNORECASE)


def build_chunks_from_knowledge(knowledge: dict) -> list[dict]:
    """Convert scraper knowledge JSON into RAG-ready chunk records."""
    shaped = ensure_v2_knowledge_shape(knowledge)
    metadata = shaped.get("metadata", {})
    website_url = _clean_text(metadata.get("url", ""))
    normalized_website_url = _clean_text(metadata.get("normalized_url", ""))
    if not normalized_website_url and website_url:
        normalized_website_url = normalize_url_for_cache(website_url)
    website_id = _clean_text(metadata.get("website_id", ""))
    if not website_id:
        website_id = make_website_id(normalized_website_url)
    primary_content = shaped.get("primary_content", {})
    source_reliability = "high"
    if isinstance(primary_content, dict):
        source_reliability = _clean_text(primary_content.get("reliability")) or "high"

    chunks: List[Dict[str, Any]] = []
    seen_texts: set[str] = set()
    seen_social_links: set[str] = set()

    for page in _pages_from_knowledge(shaped):
        if not isinstance(page, dict):
            continue

        source_url = _clean_text(page.get("page_url") or page.get("url") or website_url)
        normalized_page_url = _clean_text(page.get("normalized_page_url"))
        if not normalized_page_url and source_url:
            normalized_page_url = normalize_url_for_cache(source_url)

        page_title = _clean_text(page.get("title"))
        page_type = _clean_text(page.get("page_type")) or "other"
        extraction_method = _clean_text(page.get("extraction_method")) or "static"
        image_count = len(_as_list(page.get("images")))
        table_count = len(_as_list(page.get("tables")))

        page_context = {
            "website_id": website_id,
            "website_url": website_url,
            "normalized_website_url": normalized_website_url,
            "source_url": source_url,
            "normalized_page_url": normalized_page_url,
            "page_title": page_title,
            "page_type": page_type,
            "extraction_method": extraction_method,
            "source_reliability": source_reliability,
            "image_count": image_count,
            "table_count": table_count,
        }

        _add_page_summary(chunks, seen_texts, page, page_context)
        _add_section_chunks(chunks, seen_texts, page, page_context)
        _add_paragraph_chunks(chunks, seen_texts, page, page_context)
        _add_faq_chunks(chunks, seen_texts, page, page_context)
        _add_table_chunks(chunks, seen_texts, page, page_context)
        _add_image_context_chunks(chunks, seen_texts, page, page_context)
        _add_social_link_chunks(chunks, seen_texts, seen_social_links, page, page_context)
        _add_structured_data_chunks(chunks, seen_texts, page, page_context)

    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index

    return chunks


def summarize_chunks(chunks: list[dict]) -> dict:
    """Return a small debug summary for generated chunks."""
    by_type = Counter(chunk.get("chunk_type", "unknown") for chunk in chunks if isinstance(chunk, dict))
    pages = {
        chunk.get("source_url")
        for chunk in chunks
        if isinstance(chunk, dict) and chunk.get("source_url")
    }
    return {
        "total_chunks": len(chunks),
        "by_type": dict(sorted(by_type.items())),
        "pages_covered": len(pages),
    }


def _pages_from_knowledge(knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = knowledge.get("pages")
    if isinstance(pages, list) and pages:
        return [page for page in pages if isinstance(page, dict)]

    primary_content = knowledge.get("primary_content", {})
    legacy_pages = primary_content.get("pages") if isinstance(primary_content, dict) else []
    return [page for page in legacy_pages if isinstance(page, dict)]


def _add_page_summary(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    page: Dict[str, Any],
    page_context: Dict[str, Any],
) -> None:
    parts = []
    title = _clean_text(page.get("title"))
    description = _clean_text(page.get("description"))

    if title:
        parts.append(f"Page title: {title}")
    if description:
        parts.append(f"Description: {description}")

    text = "\n\n".join(parts)
    _add_text_chunks(
        chunks,
        seen_texts,
        text=text,
        chunk_type="page_summary",
        heading=title or "Page summary",
        page_context=page_context,
    )


def _add_section_chunks(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    page: Dict[str, Any],
    page_context: Dict[str, Any],
) -> None:
    for section in _as_list(page.get("sections")):
        heading = ""
        content = ""
        if isinstance(section, dict):
            heading = _clean_text(section.get("heading") or section.get("title") or section.get("text"))
            content = _clean_text(section.get("content") or section.get("body") or section.get("text"))
        else:
            content = _clean_text(section)

        text = _join_labeled_text([("Section", heading), ("Content", content)])
        _add_text_chunks(
            chunks,
            seen_texts,
            text=text,
            chunk_type="section",
            heading=heading,
            page_context=page_context,
            dedupe_text=content,
        )


def _add_paragraph_chunks(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    page: Dict[str, Any],
    page_context: Dict[str, Any],
) -> None:
    paragraphs = [_clean_text(paragraph) for paragraph in _as_list(page.get("paragraphs"))]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    text = "\n\n".join(paragraphs)
    if not text:
        text = _clean_text(page.get("content"))

    _add_text_chunks(
        chunks,
        seen_texts,
        text=text,
        chunk_type="paragraph_group",
        heading="",
        page_context=page_context,
    )


def _add_faq_chunks(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    page: Dict[str, Any],
    page_context: Dict[str, Any],
) -> None:
    for faq in _as_list(page.get("faqs")):
        if not isinstance(faq, dict):
            continue
        question = _clean_text(faq.get("question") or faq.get("q"))
        answer = _clean_text(faq.get("answer") or faq.get("a"))
        text = _join_labeled_text([("Question", question), ("Answer", answer)])
        _add_text_chunks(
            chunks,
            seen_texts,
            text=text,
            chunk_type="faq",
            heading=question,
            page_context=page_context,
        )


def _add_table_chunks(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    page: Dict[str, Any],
    page_context: Dict[str, Any],
) -> None:
    for table in _as_list(page.get("tables")):
        text = _table_to_text(table)
        _add_text_chunks(
            chunks,
            seen_texts,
            text=text,
            chunk_type="table",
            heading="Table",
            page_context=page_context,
        )


def _add_image_context_chunks(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    page: Dict[str, Any],
    page_context: Dict[str, Any],
) -> None:
    for image in _as_list(page.get("images")):
        text = _image_to_text(image, page_context.get("page_title", ""))
        extra_metadata = {}
        if isinstance(image, dict):
            extra_metadata = {
                "image_url": _clean_text(image.get("absolute_url") or image.get("src")),
                "image_type": _clean_text(image.get("image_type")) or "unknown",
            }
        _add_text_chunks(
            chunks,
            seen_texts,
            text=text,
            chunk_type="image_context",
            heading="Image context",
            page_context=page_context,
            extra_metadata=extra_metadata,
        )


def _add_structured_data_chunks(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    page: Dict[str, Any],
    page_context: Dict[str, Any],
) -> None:
    for record in _as_list(page.get("structured_data")):
        text = _structured_data_to_text(record)
        _add_text_chunks(
            chunks,
            seen_texts,
            text=text,
            chunk_type="structured_data",
            heading="Structured data",
            page_context=page_context,
        )


def _add_social_link_chunks(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    seen_social_links: set[str],
    page: Dict[str, Any],
    page_context: Dict[str, Any],
) -> None:
    for social_link in _extract_social_links(page):
        platform = social_link["platform"]
        url = social_link["url"]
        source_url = page_context.get("source_url", "")
        dedupe_key = _dedupe_key("|".join([platform, url, source_url]))
        if not dedupe_key or dedupe_key in seen_social_links:
            continue
        seen_social_links.add(dedupe_key)

        link_text = social_link.get("link_text", "")
        source_kind = social_link.get("source_kind", "")
        platform_label = SOCIAL_PLATFORM_LABELS.get(platform, platform.title())
        text = _join_labeled_text(
            [
                ("Social platform", platform_label),
                ("URL", url),
                ("Link text", link_text),
                ("Source page", page_context.get("page_title")),
                ("Source kind", source_kind),
            ]
        )
        _add_text_chunks(
            chunks,
            seen_texts,
            text=text,
            chunk_type="social_link",
            heading=f"{platform_label} social link",
            page_context=page_context,
            extra_metadata={
                "social_platform": platform,
                "social_url": url,
                "link_text": link_text,
                "source_kind": source_kind,
            },
            dedupe_text="|".join([platform, url, source_url]),
        )


def _add_text_chunks(
    chunks: List[Dict[str, Any]],
    seen_texts: set[str],
    text: str,
    chunk_type: str,
    heading: str,
    page_context: Dict[str, Any],
    extra_metadata: Dict[str, Any] | None = None,
    dedupe_text: str | None = None,
) -> None:
    chunk_texts = _split_text(_clean_text(text))
    dedupe_chunks = _split_text(_clean_text(dedupe_text)) if dedupe_text else []
    for index, chunk_text in enumerate(chunk_texts):
        dedupe_source = dedupe_chunks[index] if index < len(dedupe_chunks) else chunk_text
        dedupe_key = _dedupe_key(dedupe_source)
        if not dedupe_key or dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)
        chunks.append(
            _create_chunk_record(
                text=chunk_text,
                chunk_type=chunk_type,
                heading=heading,
                page_context=page_context,
                extra_metadata=extra_metadata or {},
            )
        )


def _create_chunk_record(
    text: str,
    chunk_type: str,
    heading: str,
    page_context: Dict[str, Any],
    extra_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    website_id = page_context["website_id"]
    source_url = page_context["source_url"]
    chunk_id = _make_chunk_id(
        website_id=website_id,
        source_url=source_url,
        chunk_type=chunk_type,
        text=text,
    )
    metadata = {
        "website_url": page_context.get("website_url", ""),
        "normalized_website_url": page_context.get("normalized_website_url", ""),
        "normalized_url": page_context.get("normalized_page_url", ""),
        "page_url": source_url,
        "source_type": "primary",
        "reliability": page_context.get("source_reliability", "high"),
        "extraction_method": page_context.get("extraction_method", "static"),
        "heading": _clean_text(heading),
        "image_count": int(page_context.get("image_count", 0) or 0),
        "table_count": int(page_context.get("table_count", 0) or 0),
    }
    metadata.update(extra_metadata)
    return {
        "chunk_id": chunk_id,
        "website_id": website_id,
        "source_url": source_url,
        "page_title": page_context.get("page_title", ""),
        "page_type": page_context.get("page_type", "other"),
        "chunk_type": chunk_type,
        "chunk_index": 0,
        "text": text,
        "metadata": metadata,
    }


def _split_text(text: str, max_words: int = MAX_CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if len(words) < MIN_CHUNK_WORDS:
        return []
    if len(words) <= max_words:
        return [text]

    chunks = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def _make_chunk_id(website_id: str, source_url: str, chunk_type: str, text: str) -> str:
    stable_input = "|".join([website_id, source_url, chunk_type, _dedupe_key(text)])
    digest = hashlib.md5(stable_input.encode("utf-8")).hexdigest()[:16]
    return f"{website_id}_{digest}" if website_id else digest


def _table_to_text(table: Any) -> str:
    if not isinstance(table, dict):
        return ""

    headers = [_clean_text(header) for header in _as_list(table.get("headers"))]
    headers = [header for header in headers if header]
    rows = []
    for row in _as_list(table.get("rows")):
        if isinstance(row, dict):
            values = [_clean_text(value) for value in row.values()]
        else:
            values = [_clean_text(value) for value in _as_list(row)]
        values = [value for value in values if value]
        if values:
            rows.append(values)

    if not headers and not rows:
        return ""

    parts = []
    if headers:
        parts.append("Columns: " + " | ".join(headers))
    for row in rows:
        parts.append("Row: " + " | ".join(row))
    return "\n".join(parts)


def _image_to_text(image: Any, page_title: str) -> str:
    if not isinstance(image, dict):
        return ""

    parts = []
    if page_title:
        parts.append(f"Page: {page_title}")
    parts.extend(
        _join_labeled_text(
            [
                ("Alt text", image.get("alt")),
                ("Title", image.get("title")),
                ("Caption", image.get("figcaption")),
                ("Nearby text", image.get("nearby_text")),
                ("Image type", image.get("image_type")),
            ]
        ).splitlines()
    )
    useful_parts = [part for part in parts if _clean_text(part)]
    useful_text = "\n".join(useful_parts)

    if not any(
        _clean_text(image.get(field))
        for field in ("alt", "title", "figcaption", "nearby_text")
    ):
        return ""
    return useful_text


def _structured_data_to_text(record: Any) -> str:
    if isinstance(record, str):
        return _clean_text(record)
    if isinstance(record, (dict, list)):
        return _clean_text(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return ""


def _extract_social_links(page: Dict[str, Any]) -> List[Dict[str, str]]:
    links = []

    for link in _as_list(page.get("links")):
        if not isinstance(link, dict):
            continue
        url = _clean_social_url(link.get("url") or link.get("href") or link.get("absolute_url"))
        platform = _detect_social_platform(url, link.get("text"))
        if not platform or not url:
            continue
        links.append(
            {
                "platform": platform,
                "url": url,
                "link_text": _clean_text(link.get("text") or link.get("title") or platform.title()),
                "source_kind": "page_link",
            }
        )

    for record in _as_list(page.get("structured_data")):
        for url in _extract_urls_from_value(record):
            cleaned_url = _clean_social_url(url)
            platform = _detect_social_platform(cleaned_url)
            if not platform:
                continue
            links.append(
                {
                    "platform": platform,
                    "url": cleaned_url,
                    "link_text": platform.title(),
                    "source_kind": "structured_data",
                }
            )

    return links


def _extract_urls_from_value(value: Any) -> List[str]:
    urls = []
    if isinstance(value, str):
        urls.extend(URL_RE.findall(value))
        if value.strip().startswith(("{", "[")):
            try:
                urls.extend(_extract_urls_from_value(json.loads(value)))
            except (TypeError, json.JSONDecodeError):
                pass
    elif isinstance(value, dict):
        for nested_value in value.values():
            urls.extend(_extract_urls_from_value(nested_value))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_extract_urls_from_value(item))
    return urls


def _detect_social_platform(url: Any, link_text: Any = "") -> str:
    cleaned_url = _clean_social_url(url)
    lowered_url = cleaned_url.lower()
    lowered_text = _clean_text(link_text).lower()
    for platform, domains in SOCIAL_PLATFORMS.items():
        if any(_domain_matches(lowered_url, domain) for domain in domains):
            return platform
        if lowered_text == platform or f" {platform} " in f" {lowered_text} ":
            return platform
    if lowered_text in {"x", "twitter/x", "x twitter"}:
        return "twitter"
    return ""


def _domain_matches(url: str, domain: str) -> bool:
    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        hostname = ""
    hostname = hostname.lower()
    return hostname == domain or hostname.endswith(f".{domain}")


def _clean_social_url(value: Any) -> str:
    text = _clean_text(value)
    return text.rstrip(".,;)")


def _join_labeled_text(parts: Iterable[tuple[str, Any]]) -> str:
    lines = []
    for label, value in parts:
        cleaned = _clean_text(value)
        if cleaned:
            lines.append(f"{label}: {cleaned}")
    return "\n".join(lines)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []
