import json
import re
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .scraper_schema import create_page_record, create_quality_record


NOISE_PATTERNS = (
    "ad-",
    "ads",
    "advertisement",
    "banner",
    "cookie",
    "modal",
    "newsletter",
    "overlay",
    "popup",
    "promo",
    "subscribe",
)

SKIP_LINK_SCHEMES = ("javascript:", "mailto:", "tel:")
MAX_CLEAN_TEXT_CHARS = 12000


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _copy_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "lxml")


def remove_noise(soup: BeautifulSoup) -> None:
    """Remove obvious non-content elements while preserving possible footer/contact text."""
    for element in soup.find_all(["script", "style", "noscript", "iframe", "svg"]):
        element.decompose()

    for pattern in NOISE_PATTERNS:
        for element in soup.find_all(class_=lambda value: value and pattern in str(value).lower()):
            element.decompose()
        for element in soup.find_all(id=lambda value: value and pattern in str(value).lower()):
            element.decompose()

    for element in soup.find_all():
        if element.name in {"img", "table", "thead", "tbody", "tr", "th", "td"}:
            continue
        if not element.get_text(strip=True) and not element.find(["img", "table"]):
            element.decompose()


def extract_page_metadata(soup: BeautifulSoup, page_url: str) -> Dict[str, Any]:
    title = ""
    if soup.title:
        title = _clean_text(soup.title.get_text())
    if not title and soup.find("h1"):
        title = _clean_text(soup.find("h1").get_text())

    description = ""
    meta_desc = soup.find("meta", attrs={"name": lambda value: value and value.lower() == "description"})
    if meta_desc and meta_desc.get("content"):
        description = _clean_text(meta_desc["content"])

    canonical = ""
    canonical_link = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical_link and canonical_link.get("href"):
        canonical = urljoin(page_url, canonical_link["href"])

    open_graph = {}
    for tag in soup.find_all("meta", property=lambda value: value and value.lower().startswith("og:")):
        key = tag.get("property", "")[3:]
        if key and tag.get("content"):
            open_graph[key] = _clean_text(tag["content"])

    twitter = {}
    for tag in soup.find_all("meta", attrs={"name": lambda value: value and value.lower().startswith("twitter:")}):
        key = tag.get("name", "")[8:]
        if key and tag.get("content"):
            twitter[key] = _clean_text(tag["content"])

    return {
        "title": title,
        "description": description,
        "metadata": {
            "canonical": canonical,
            "open_graph": open_graph,
            "twitter": twitter,
        },
    }


def extract_headings(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    headings = []
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        text = _clean_text(heading.get_text(" "))
        if text:
            headings.append({"level": int(heading.name[1]), "text": text})
    return headings


def extract_paragraphs(soup: BeautifulSoup) -> List[str]:
    paragraphs = []
    seen = set()
    for paragraph in soup.find_all("p"):
        text = _clean_text(paragraph.get_text(" "))
        if text and text not in seen:
            paragraphs.append(text)
            seen.add(text)
    return paragraphs


def extract_lists(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    lists = []
    for list_node in soup.find_all(["ul", "ol"]):
        items = []
        for item in list_node.find_all("li", recursive=False):
            text = _clean_text(item.get_text(" "))
            if text:
                items.append(text)
        if items:
            lists.append({"type": list_node.name, "items": items})
    return lists


def extract_tables(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    tables = []
    for table in soup.find_all("table"):
        rows = []
        headers = []
        for row_index, tr in enumerate(table.find_all("tr")):
            cells = [_clean_text(cell.get_text(" ")) for cell in tr.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            has_header_cells = bool(tr.find_all("th"))
            if has_header_cells and not headers:
                headers = cells
                continue
            if row_index == 0 and not headers and len(cells) > 1 and tr.find_all("th"):
                headers = cells
                continue
            rows.append(cells)
        if headers or rows:
            tables.append({"headers": headers, "rows": rows})
    return tables


def extract_links(soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc.lower()
    links = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href or href.startswith(SKIP_LINK_SCHEMES):
            continue
        absolute_url = urljoin(base_url, href)
        parsed_url = urlparse(absolute_url)
        if parsed_url.scheme not in {"http", "https"}:
            continue
        text = _clean_text(anchor.get_text(" ") or anchor.get("aria-label", "") or anchor.get("title", ""))
        if not text:
            continue
        key = (text, absolute_url)
        if key in seen:
            continue
        seen.add(key)
        links.append({
            "text": text,
            "url": absolute_url,
            "is_internal": parsed_url.netloc.lower() == base_domain,
        })
    return links


def extract_faqs(soup: BeautifulSoup) -> List[Dict[str, str]]:
    faqs = []
    seen = set()

    for details in soup.find_all("details"):
        summary = details.find("summary")
        question = _clean_text(summary.get_text(" ")) if summary else ""
        answer_parts = [
            _clean_text(child.get_text(" ") if hasattr(child, "get_text") else str(child))
            for child in details.children
            if child != summary
        ]
        answer = _clean_text(" ".join(part for part in answer_parts if part))
        if question and answer and question not in seen:
            faqs.append({"question": question, "answer": answer})
            seen.add(question)

    question_pattern = re.compile(r"\?$|^(q:|question:)|\b(what|why|how|when|where|who|can|do|does|is|are)\b", re.I)
    for heading in soup.find_all(["h2", "h3", "h4"]):
        question = _clean_text(heading.get_text(" "))
        if not question or question in seen or not question_pattern.search(question):
            continue
        answer_parts = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3", "h4"]:
                break
            text = _clean_text(sibling.get_text(" "))
            if text:
                answer_parts.append(text)
            if len(" ".join(answer_parts)) > 600:
                break
        answer = _clean_text(" ".join(answer_parts))
        if answer:
            faqs.append({"question": question, "answer": answer[:1000]})
            seen.add(question)

    return faqs


def extract_json_ld(soup: BeautifulSoup) -> List[Any]:
    records = []
    for script in soup.find_all("script", type=lambda value: value and "ld+json" in value.lower()):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return records


def extract_sections(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    sections = []
    for heading in soup.find_all(["h1", "h2", "h3"]):
        heading_text = _clean_text(heading.get_text(" "))
        if not heading_text:
            continue
        content_parts = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ["h1", "h2", "h3"]:
                break
            text = _clean_text(sibling.get_text(" "))
            if text:
                content_parts.append(text)
            if len(" ".join(content_parts)) > 2000:
                break
        content = _clean_text(" ".join(content_parts))
        if content:
            sections.append({
                "heading": heading_text,
                "level": int(heading.name[1]),
                "content": content[:2000],
            })
    return sections[:20]


def build_clean_text(soup: BeautifulSoup) -> str:
    main_element = soup.find("main") or soup.find("article") or soup.find("body") or soup
    return _clean_text(main_element.get_text(" "))[:MAX_CLEAN_TEXT_CHARS]


def calculate_page_quality(
    content: str,
    paragraphs: List[str],
    headings: List[Dict[str, Any]],
    image_count: int,
    link_count: int,
    status_code: int = 200,
    errors: List[str] | None = None,
) -> Dict[str, Any]:
    content_length = len(content or "")
    useful_text_score = min(
        1.0,
        (content_length / 3000 * 0.5)
        + (len(paragraphs) / 12 * 0.3)
        + (len(headings) / 8 * 0.2),
    )
    return create_quality_record(
        status_code=status_code,
        content_length=content_length,
        useful_text_score=round(useful_text_score, 3),
        image_count=image_count,
        link_count=link_count,
        errors=errors or [],
    )


def extract_static_page(
    html: str,
    page_url: str = "",
    page_type: str = "other",
    status_code: int = 200,
    errors: List[str] | None = None,
) -> Dict[str, Any]:
    """Extract a static HTML page into the scraper v2 page schema."""
    if not html:
        return create_page_record(
            page_url=page_url,
            page_type=page_type,
            quality=create_quality_record(status_code=status_code, errors=errors or []),
        )

    soup = _copy_soup(html)
    page_metadata = extract_page_metadata(soup, page_url)
    structured_data = extract_json_ld(soup)
    remove_noise(soup)

    headings = extract_headings(soup)
    paragraphs = extract_paragraphs(soup)
    lists = extract_lists(soup)
    tables = extract_tables(soup)
    links = extract_links(soup, page_url)
    faqs = extract_faqs(soup)
    sections = extract_sections(soup)
    content = build_clean_text(soup)
    image_count = len(soup.find_all("img"))
    quality = calculate_page_quality(
        content=content,
        paragraphs=paragraphs,
        headings=headings,
        image_count=image_count,
        link_count=len(links),
        status_code=status_code,
        errors=errors or [],
    )

    return create_page_record(
        page_url=page_url,
        page_type=page_type,
        title=page_metadata["title"],
        description=page_metadata["description"],
        extraction_method="static",
        headings=headings,
        sections=sections,
        paragraphs=paragraphs,
        lists=lists,
        faqs=faqs,
        tables=tables,
        links=links,
        structured_data=structured_data,
        metadata=page_metadata["metadata"],
        quality=quality,
        content=content,
    )
