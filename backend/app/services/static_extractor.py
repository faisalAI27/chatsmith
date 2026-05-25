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
SKIP_IMAGE_SCHEMES = ("data:", "blob:", "javascript:", "mailto:", "tel:")
MAX_CLEAN_TEXT_CHARS = 12000
MAX_NEARBY_IMAGE_TEXT_CHARS = 300

IMAGE_SOURCE_ATTRS = (
    "src",
    "data-src",
    "data-lazy-src",
    "data-original",
    "data-srcset",
    "srcset",
)
PLACEHOLDER_IMAGE_PATTERNS = (
    "blank",
    "pixel",
    "placeholder",
    "spacer",
    "transparent",
    "tracking",
)
ICON_IMAGE_PATTERNS = (
    "avatar-icon",
    "facebook",
    "favicon",
    "icon",
    "instagram",
    "linkedin",
    "social",
    "sprite",
    "twitter",
    "youtube",
)


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


def _first_srcset_candidate(srcset: str) -> str:
    """Return the first URL candidate from a simple srcset string."""
    for candidate in (srcset or "").split(","):
        candidate_url = candidate.strip().split(" ")[0].strip()
        if candidate_url:
            return candidate_url
    return ""


def _is_placeholder_image_src(src: str) -> bool:
    src_lower = (src or "").lower()
    if not src_lower:
        return True
    if src_lower.startswith(SKIP_IMAGE_SCHEMES):
        return True
    return any(pattern in src_lower for pattern in PLACEHOLDER_IMAGE_PATTERNS)


def normalize_image_url(src: str, page_url: str) -> str:
    """Normalize an image source to an absolute URL, or empty string if unusable."""
    raw_src = _first_srcset_candidate(src) if "," in (src or "") else (src or "")
    raw_src = raw_src.strip().strip("'\"")
    if not raw_src or _is_placeholder_image_src(raw_src):
        return ""
    absolute_url = urljoin(page_url, raw_src)
    parsed_url = urlparse(absolute_url)
    if parsed_url.scheme not in {"http", "https"}:
        return ""
    return absolute_url


def _extract_image_source(img_tag) -> str:
    for attr in IMAGE_SOURCE_ATTRS:
        value = img_tag.get(attr)
        if not value:
            continue
        source = _first_srcset_candidate(value) if "srcset" in attr else value
        if source and not _is_placeholder_image_src(source):
            return source.strip()

    picture = img_tag.find_parent("picture")
    if picture:
        for source_tag in picture.find_all("source"):
            for attr in ("srcset", "data-srcset", "src", "data-src"):
                value = source_tag.get(attr)
                if not value:
                    continue
                source = _first_srcset_candidate(value) if "srcset" in attr else value
                if source and not _is_placeholder_image_src(source):
                    return source.strip()
    return ""


def _parse_dimension(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    if not match:
        return None
    return int(match.group(0))


def _extract_style_dimension(style: str, dimension: str) -> int | None:
    pattern = rf"{dimension}\s*:\s*(\d+(?:\.\d+)?)px"
    match = re.search(pattern, style or "", re.I)
    if not match:
        return None
    return int(float(match.group(1)))


def _image_dimensions(img_tag) -> tuple[int | None, int | None]:
    width = _parse_dimension(img_tag.get("width"))
    height = _parse_dimension(img_tag.get("height"))
    style = img_tag.get("style", "")
    if width is None:
        width = _extract_style_dimension(style, "width")
    if height is None:
        height = _extract_style_dimension(style, "height")
    return width, height


def get_image_figcaption(img_tag) -> str:
    figure = img_tag.find_parent("figure")
    if not figure:
        return ""
    figcaption = figure.find("figcaption")
    return _clean_text(figcaption.get_text(" ")) if figcaption else ""


def _unique_text_parts(parts: List[str]) -> List[str]:
    unique_parts = []
    seen = set()
    for part in parts:
        cleaned = _clean_text(part)
        key = cleaned.lower()
        if cleaned and key not in seen:
            unique_parts.append(cleaned)
            seen.add(key)
    return unique_parts


def get_nearby_image_text(img_tag) -> str:
    """Collect short text near an image without pulling in the full page body."""
    text_parts = []
    aria_label = img_tag.get("aria-label")
    if aria_label:
        text_parts.append(aria_label)

    context_nodes = [img_tag]
    parent = img_tag.parent
    figure = img_tag.find_parent("figure")
    if parent:
        context_nodes.append(parent)
    if figure and figure not in context_nodes:
        context_nodes.append(figure)

    for node in context_nodes:
        for sibling in list(node.find_previous_siblings(limit=2)) + list(node.find_next_siblings(limit=2)):
            if getattr(sibling, "name", None) in {"script", "style", "noscript", "svg"}:
                continue
            if hasattr(sibling, "get_text"):
                text_parts.append(sibling.get_text(" "))

    for ancestor in img_tag.parents:
        if getattr(ancestor, "name", None) in {"body", "html", "main"}:
            break
        text = _clean_text(ancestor.get_text(" "))
        if 10 <= len(text) <= 500:
            text_parts.append(text)
            break

    nearby_text = " ".join(_unique_text_parts(text_parts))
    return _clean_text(nearby_text)[:MAX_NEARBY_IMAGE_TEXT_CHARS]


def classify_image_type(image_record: Dict[str, Any]) -> str:
    combined_text = " ".join(
        str(image_record.get(key, ""))
        for key in ("src", "absolute_url", "alt", "title", "figcaption", "nearby_text")
    ).lower()
    if "logo" in combined_text:
        return "logo"
    if any(pattern in combined_text for pattern in ICON_IMAGE_PATTERNS):
        return "icon"
    if any(keyword in combined_text for keyword in ("background", "decorative")):
        return "decorative"
    if any(keyword in combined_text for keyword in ("hero", "banner", "cover")):
        return "hero"
    if any(keyword in combined_text for keyword in ("product", "catalog", "sku", "shop")):
        return "product"
    if any(keyword in combined_text for keyword in ("team", "staff", "founder", "leader", "people")):
        return "team"
    if any(keyword in combined_text for keyword in ("diagram", "chart", "graph", "flow", "infographic")):
        return "diagram"
    if any(keyword in combined_text for keyword in ("screenshot", "screen shot", "dashboard", "interface")):
        return "screenshot"
    if not any(
        image_record.get(key)
        for key in ("alt", "title", "figcaption", "nearby_text")
    ):
        return "decorative"
    return "content"


def _descriptive_text_score(image_record: Dict[str, Any]) -> int:
    return sum(
        len(str(image_record.get(key, "") or ""))
        for key in ("alt", "title", "figcaption", "nearby_text")
    )


def is_useful_image(image_record: Dict[str, Any]) -> bool:
    absolute_url = image_record.get("absolute_url", "")
    if not absolute_url:
        return False

    width = image_record.get("width")
    height = image_record.get("height")
    if width and height and width <= 1 and height <= 1:
        return False

    image_type = image_record.get("image_type") or classify_image_type(image_record)
    has_context = any(
        image_record.get(key)
        for key in ("alt", "title", "figcaption", "nearby_text")
    )
    if not has_context:
        return False

    if image_type == "decorative":
        return False

    small_image = bool(width and height and (width <= 64 or height <= 64))
    if image_type == "icon" and (
        small_image
        or not (image_record.get("figcaption") or image_record.get("nearby_text"))
        or _descriptive_text_score(image_record) < 50
    ):
        return False
    if image_type == "logo" and not image_record.get("figcaption"):
        return False
    if small_image and _descriptive_text_score(image_record) < 25:
        return False

    source_text = f"{image_record.get('src', '')} {absolute_url}".lower()
    social_names = ("facebook", "instagram", "linkedin", "twitter", "youtube")
    if any(name in source_text for name in social_names):
        return False
    if any(pattern in source_text for pattern in ICON_IMAGE_PATTERNS) and small_image:
        return False

    return True


def extract_images(soup: BeautifulSoup, page_url: str) -> List[Dict[str, Any]]:
    """Extract useful image metadata without downloading image binaries."""
    images_by_url: Dict[str, Dict[str, Any]] = {}
    for img_tag in soup.find_all("img"):
        src = _extract_image_source(img_tag)
        absolute_url = normalize_image_url(src, page_url)
        width, height = _image_dimensions(img_tag)
        image_record = {
            "src": src,
            "absolute_url": absolute_url,
            "alt": _clean_text(img_tag.get("alt", "")),
            "title": _clean_text(img_tag.get("title", "")),
            "figcaption": get_image_figcaption(img_tag),
            "nearby_text": get_nearby_image_text(img_tag),
            "width": width,
            "height": height,
            "loading": _clean_text(img_tag.get("loading", "")),
            "page_url": page_url,
            "image_type": "unknown",
        }
        image_record["image_type"] = classify_image_type(image_record)

        if not is_useful_image(image_record):
            continue

        existing = images_by_url.get(absolute_url)
        if not existing or _descriptive_text_score(image_record) > _descriptive_text_score(existing):
            images_by_url[absolute_url] = image_record

    return list(images_by_url.values())


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
    images = extract_images(soup, page_url)
    remove_noise(soup)

    headings = extract_headings(soup)
    paragraphs = extract_paragraphs(soup)
    lists = extract_lists(soup)
    tables = extract_tables(soup)
    links = extract_links(soup, page_url)
    faqs = extract_faqs(soup)
    sections = extract_sections(soup)
    content = build_clean_text(soup)
    quality = calculate_page_quality(
        content=content,
        paragraphs=paragraphs,
        headings=headings,
        image_count=len(images),
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
        images=images,
        links=links,
        structured_data=structured_data,
        metadata=page_metadata["metadata"],
        quality=quality,
        content=content,
    )
