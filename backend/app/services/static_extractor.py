"""Structured extraction of a single HTML page into the scraper v2 page schema.

The extractor works on any HTML (static or Playwright-rendered). Text is read
as an ordered list of *blocks* (paragraphs, list items, table rows, headings,
and text that sits directly inside div/span containers), so it works for
classic article markup and for component-based layouts (React, Vue,
Tailwind cards, page builders) where headings and text are not siblings.
"""

import json
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from bs4.element import CData, Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag

from .scraper_schema import create_page_record, create_quality_record
from .text_fingerprint import content_hash, simhash_hex


# Class/id *tokens* (whole words after splitting on - _ and camelCase) that mark
# overlays and ads. Matching whole tokens avoids the old substring bug where
# "read-more" matched "ad-" and "downloads" matched "ads".
NOISE_TOKENS = {
    "ad",
    "ads",
    "adsense",
    "adslot",
    "advert",
    "adverts",
    "advertisement",
    "advertising",
    "consent",
    "cookie",
    "cookies",
    "gdpr",
    "modal",
    "newsletter",
    "popup",
    "sponsored",
    "subscribe",
}
# Kept for backward compatibility with code that imported the old constant.
NOISE_PATTERNS = tuple(sorted(NOISE_TOKENS))
NAV_TOKENS = {
    "breadcrumb",
    "breadcrumbs",
    "megamenu",
    "menu",
    "nav",
    "navbar",
    "navigation",
    "submenu",
}
NOISE_ROLES = {"dialog", "alertdialog"}
NAV_ROLES = {"navigation", "menubar", "menu"}
PROTECTED_TAGS = {"html", "body", "main", "article"}
ALWAYS_REMOVE_TAGS = ["script", "style", "noscript", "iframe", "svg", "template", "canvas", "object", "embed"]
CHROME_TAGS = {"header", "footer", "aside", "nav"}

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "body", "caption", "dd", "details",
    "dialog", "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hgroup", "html", "legend", "li",
    "main", "nav", "ol", "p", "pre", "section", "summary", "table", "tbody", "tfoot",
    "thead", "tr", "ul",
}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SECTION_HEADING_MAX_LEVEL = 4

SKIP_LINK_SCHEMES = ("javascript:", "mailto:", "tel:")
SKIP_IMAGE_SCHEMES = ("data:", "blob:", "javascript:", "mailto:", "tel:")
# Safety limit only; the chunker splits long text. The old 12,000-character cap
# silently dropped most of long pages.
MAX_CLEAN_TEXT_CHARS = 300_000
MAX_SECTIONS = 300
MAX_NAVIGATION_ITEMS = 300
MAX_NEARBY_IMAGE_TEXT_CHARS = 300
MAX_FAQ_ANSWER_CHARS = 1500
LAYOUT_TABLE_CELL_CHARS = 1000

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
JS_REQUIRED_PATTERN = re.compile(r"(enable|requires?|turn on)\s+javascript|javascript\s+(is\s+)?(required|disabled)", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
QUESTION_PREFIX_RE = re.compile(r"^(q\s*[:.)-]|question\s*[:.)-]|q\d+\s*[:.)-])", re.I)
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SKIP_STRING_TYPES = (Comment, Declaration, Doctype, CData, ProcessingInstruction)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _smart_join(strings: Iterable[str]) -> str:
    """Join text nodes, adding a space only where two words would otherwise fuse."""
    result = ""
    for raw in strings:
        piece = str(raw)
        if not piece:
            continue
        if result and not result[-1].isspace() and not piece[0].isspace():
            if result[-1].isalnum() and piece[0].isalnum():
                result += " "
        result += piece
    return _clean_text(result)


def _visible_strings(element: Tag) -> Iterable[str]:
    for node in element.find_all(string=True):
        if isinstance(node, _SKIP_STRING_TYPES):
            continue
        yield str(node)


def _element_text(element: Tag | None) -> str:
    if element is None:
        return ""
    return _smart_join(_visible_strings(element))


def _copy_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "lxml")


def _attr_tokens(element: Tag) -> set[str]:
    """Split class and id values into lowercase whole-word tokens."""
    values: List[str] = []
    classes = element.get("class")
    if isinstance(classes, (list, tuple)):
        values.extend(str(value) for value in classes)
    elif classes:
        values.append(str(classes))
    if element.get("id"):
        values.append(str(element.get("id")))
    tokens: set[str] = set()
    for value in values:
        spaced = _CAMEL_RE.sub(" ", value)
        tokens.update(token for token in _TOKEN_SPLIT_RE.split(spaced.lower()) if token)
    return tokens


def _is_decomposed(element: Any) -> bool:
    return bool(getattr(element, "decomposed", False))


def _link_density(element: Tag) -> float:
    total = len(_element_text(element))
    if not total:
        return 0.0
    link_chars = sum(len(_element_text(anchor)) for anchor in element.find_all("a"))
    return min(1.0, link_chars / total)


def _is_protected(element: Tag, body_text_length: int) -> bool:
    """Never delete the page itself, the main content, or a large wrapper."""
    if element.name in PROTECTED_TAGS:
        return True
    if (element.get("role") or "").lower() == "main":
        return True
    if element.find(["main", "article", "h1"]) is not None:
        return True
    if element.find(attrs={"role": "main"}) is not None:
        return True
    if body_text_length >= 200 and len(_element_text(element)) > 0.3 * body_text_length:
        return True
    return False


# ---------------------------------------------------------------------------
# Noise and navigation removal
# ---------------------------------------------------------------------------

def remove_noise(soup: BeautifulSoup) -> None:
    """Remove scripts, overlays, cookie bars, and ads without touching main content."""
    for element in soup.find_all(ALWAYS_REMOVE_TAGS):
        element.decompose()

    body = soup.find("body") or soup
    body_text_length = len(_element_text(body))
    for element in soup.find_all(True):
        if _is_decomposed(element) or element.name in PROTECTED_TAGS:
            continue
        role = (element.get("role") or "").lower()
        is_dialog = role in NOISE_ROLES or (element.get("aria-modal") or "").lower() == "true"
        if not is_dialog and not (_attr_tokens(element) & NOISE_TOKENS):
            continue
        if _is_protected(element, body_text_length):
            continue
        element.decompose()


def extract_and_remove_navigation(soup: BeautifulSoup) -> List[str]:
    """Remove menus from the content tree and return their item labels.

    Menu labels are still useful once per site (for example the list of
    programmes in a university mega-menu), so they are returned instead of
    being thrown away.
    """
    body = soup.find("body") or soup
    body_text_length = len(_element_text(body))
    items: List[str] = []
    seen: set[str] = set()

    for element in soup.find_all(True):
        if _is_decomposed(element) or element.name in PROTECTED_TAGS:
            continue
        role = (element.get("role") or "").lower()
        is_semantic_nav = element.name == "nav" or role in NAV_ROLES
        is_menu_like = bool(_attr_tokens(element) & NAV_TOKENS) and element.name in {"div", "ul", "ol", "section", "aside", "header"}
        if not is_semantic_nav and not is_menu_like:
            continue
        if is_menu_like and not is_semantic_nav and _link_density(element) < 0.5:
            continue  # e.g. a restaurant "menu" with dishes and prices is content
        if _is_protected(element, body_text_length):
            continue
        for anchor in element.find_all(["a", "button", "li"]):
            if anchor.name == "li" and anchor.find(["a", "button"]):
                continue
            label = _element_text(anchor)
            key = label.lower()
            if label and len(label) <= 120 and key not in seen:
                seen.add(key)
                items.append(label)
        element.decompose()

    return items[:MAX_NAVIGATION_ITEMS]


# ---------------------------------------------------------------------------
# Block model
# ---------------------------------------------------------------------------

def _nearest_block(node: Any, root: Tag) -> Tag:
    parent = node.parent
    while parent is not None and parent is not root:
        if isinstance(parent, Tag) and parent.name in BLOCK_TAGS:
            return parent
        parent = parent.parent
    return root


def _has_ancestor(element: Tag, names: set[str], root: Tag) -> bool:
    current = element
    while current is not None and current is not root:
        if isinstance(current, Tag) and current.name in names:
            return True
        current = current.parent
    return isinstance(root, Tag) and root.name in names


def _row_text(row: Tag) -> str:
    cells = [_element_text(cell) for cell in row.find_all(["th", "td"], recursive=False)]
    cells = [cell for cell in cells if cell]
    return " | ".join(cells)


def collect_blocks(root: Tag, main_root: Tag | None = None) -> List[Dict[str, Any]]:
    """Return the page text as ordered blocks.

    Each block is a plain dict (JSON-serialisable): ``text``, ``tag``,
    ``level`` (headings only), ``chrome`` (inside header/footer/aside),
    ``in_main`` and ``group`` (table id for table rows).
    """
    blocks: List[Dict[str, Any]] = []
    table_ids: Dict[int, int] = {}
    cell_is_small: Dict[int, bool] = {}
    current: Tag | None = None
    parts: List[str] = []

    def row_for(block_element: Tag) -> Tag:
        """Map text inside a small data-table cell to its row, so a row is one block.

        Big cells (layout tables wrapping whole pages) keep their inner blocks.
        """
        if block_element.name == "tr":
            return block_element
        cell = block_element.find_parent(["td", "th"])
        if cell is None:
            return block_element
        key = id(cell)
        if key not in cell_is_small:
            cell_is_small[key] = len(_element_text(cell)) <= LAYOUT_TABLE_CELL_CHARS
        if not cell_is_small[key]:
            return block_element
        return cell.find_parent("tr") or block_element

    def flush() -> None:
        if current is None or not parts:
            return
        if current.name == "tr":
            text = _row_text(current) or _smart_join(parts)
        else:
            text = _smart_join(parts)
        if not text:
            return
        block: Dict[str, Any] = {
            "text": text[:MAX_CLEAN_TEXT_CHARS],
            "tag": current.name,
            "level": int(current.name[1]) if current.name in HEADING_TAGS else 0,
            "chrome": _has_ancestor(current, CHROME_TAGS, root),
            "in_main": bool(main_root is not None and (current is main_root or main_root in current.parents)),
            "group": 0,
        }
        if current.name == "tr":
            table = current.find_parent("table")
            if table is not None:
                block["group"] = table_ids.setdefault(id(table), len(table_ids) + 1)
        blocks.append(block)

    for node in root.find_all(string=True):
        if isinstance(node, _SKIP_STRING_TYPES):
            continue
        text = str(node)
        if not text.strip():
            continue
        block_element = _nearest_block(node, root)
        if block_element is not root and block_element.name not in HEADING_TAGS:
            block_element = row_for(block_element)
        if block_element is not current:
            flush()
            current = block_element
            parts = []
        parts.append(text)
    flush()

    # A <tr> appears once per string inside it; merge consecutive duplicates.
    merged: List[Dict[str, Any]] = []
    for block in blocks:
        if merged and block["tag"] == "tr" and merged[-1]["tag"] == "tr" and merged[-1]["text"] == block["text"]:
            continue
        merged.append(block)
    return merged


def build_sections_from_blocks(blocks: List[Dict[str, Any]], intro_heading: str = "") -> List[Dict[str, Any]]:
    """Group blocks under the nearest preceding heading (h1-h4)."""
    sections: List[Dict[str, Any]] = []
    heading = intro_heading
    level = 0
    parts: List[str] = []

    def flush() -> None:
        content = _clean_text(" ".join(parts))
        if content:
            sections.append({"heading": heading, "level": level, "content": content})

    for block in blocks:
        if block.get("tag") in HEADING_TAGS and int(block.get("level") or 0) <= SECTION_HEADING_MAX_LEVEL:
            flush()
            heading = block["text"]
            level = int(block["level"])
            parts = []
        else:
            parts.append(block.get("text", ""))
    flush()
    return sections[:MAX_SECTIONS]


def build_content_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    return _clean_text(" ".join(block.get("text", "") for block in blocks))[:MAX_CLEAN_TEXT_CHARS]


def core_text_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    """Text of the page's own content: <main> if present, else non-chrome blocks."""
    in_main = [block for block in blocks if block.get("in_main")]
    selected = in_main if len(build_content_from_blocks(in_main)) >= 200 else [
        block for block in blocks if not block.get("chrome")
    ]
    if not selected:
        selected = blocks
    return build_content_from_blocks(selected)


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def extract_page_metadata(soup: BeautifulSoup, page_url: str) -> Dict[str, Any]:
    title = ""
    if soup.title:
        title = _clean_text(soup.title.get_text())
    if not title and soup.find("h1"):
        title = _element_text(soup.find("h1"))

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
    if not description and open_graph.get("description"):
        description = open_graph["description"]

    twitter = {}
    for tag in soup.find_all("meta", attrs={"name": lambda value: value and value.lower().startswith("twitter:")}):
        key = tag.get("name", "")[8:]
        if key and tag.get("content"):
            twitter[key] = _clean_text(tag["content"])

    html_tag = soup.find("html")
    language = _clean_text(html_tag.get("lang", "")) if html_tag else ""

    metadata = {
        "canonical": canonical,
        "open_graph": open_graph,
        "twitter": twitter,
    }
    if language:
        metadata["language"] = language
    return {
        "title": title,
        "description": description,
        "metadata": metadata,
    }


def extract_headings(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    headings = []
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        text = _element_text(heading)
        if text:
            headings.append({"level": int(heading.name[1]), "text": text})
    return headings


def extract_paragraphs(soup: BeautifulSoup) -> List[str]:
    paragraphs = []
    seen = set()
    for paragraph in soup.find_all("p"):
        text = _element_text(paragraph)
        if text and text not in seen:
            paragraphs.append(text)
            seen.add(text)
    return paragraphs


def _li_own_text(item: Tag) -> str:
    """Text of a list item without the text of lists nested inside it."""
    strings = []
    for node in item.find_all(string=True):
        if isinstance(node, _SKIP_STRING_TYPES):
            continue
        nearest_list = node.find_parent(["ul", "ol"])
        if nearest_list is item.parent:
            strings.append(str(node))
    return _smart_join(strings)


def extract_lists(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    lists = []
    for list_node in soup.find_all(["ul", "ol"]):
        items = []
        for item in list_node.find_all("li", recursive=False):
            text = _li_own_text(item)
            if text:
                items.append(text)
        if items:
            lists.append({"type": list_node.name, "items": items})
    return lists


def _is_layout_table(table: Tag) -> bool:
    if table.find("table") is not None:
        return True
    return any(len(_element_text(cell)) > LAYOUT_TABLE_CELL_CHARS for cell in table.find_all(["td", "th"]))


def extract_tables(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    tables = []
    for table in soup.find_all("table"):
        if _is_layout_table(table):
            continue
        rows = []
        headers: List[str] = []
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            cells: List[str] = []
            for cell in tr.find_all(["th", "td"], recursive=False):
                text = _element_text(cell)
                try:
                    span = max(1, min(int(cell.get("colspan", 1)), 10))
                except (TypeError, ValueError):
                    span = 1
                cells.extend([text] * span)
            if not any(cells):
                continue
            in_thead = tr.find_parent("thead") is not None
            # Only an all-<th> (or <thead>) first row is a header row. Tables
            # with a <th> in the first column are row-labelled, not headed.
            is_header_row = all(cell.name == "th" for cell in tr.find_all(["th", "td"], recursive=False))
            if not headers and not rows and (in_thead or is_header_row):
                headers = cells
                continue
            rows.append(cells)
        while headers and not headers[-1]:
            headers.pop()
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
        if not href or href.lower().startswith(SKIP_LINK_SCHEMES):
            continue
        absolute_url = urljoin(base_url, href)
        parsed_url = urlparse(absolute_url)
        if parsed_url.scheme not in {"http", "https"}:
            continue
        text = _element_text(anchor) or _clean_text(anchor.get("aria-label", "") or anchor.get("title", ""))
        if not text:
            image = anchor.find("img", alt=True)
            text = _clean_text(image.get("alt", "")) if image else ""
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


def extract_contacts(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """Collect e-mail addresses, phone numbers and WhatsApp links from hrefs.

    Icon-only contact links have no visible text, so without this they never
    reach the knowledge base.
    """
    emails: List[str] = []
    phones: List[str] = []
    whatsapp: List[str] = []

    def add(values: List[str], value: str) -> None:
        value = value.strip()
        if value and value not in values:
            values.append(value)

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        lower = href.lower()
        if lower.startswith("mailto:"):
            address = href[7:].split("?", 1)[0]
            if EMAIL_RE.fullmatch(address):
                add(emails, address)
        elif lower.startswith("tel:"):
            number = re.sub(r"[^\d+]", "", href[4:])
            if len(re.sub(r"\D", "", number)) >= 6:
                add(phones, number)
        elif "wa.me/" in lower or "api.whatsapp.com" in lower or "whatsapp://" in lower:
            add(whatsapp, href)
    return {"emails": emails[:20], "phones": phones[:20], "whatsapp": whatsapp[:10]}


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


def _looks_like_question(text: str) -> bool:
    text = _clean_text(text)
    if not text or len(text) > 300:
        return False
    return text.endswith("?") or bool(QUESTION_PREFIX_RE.match(text))


def _strip_question_prefix(text: str) -> str:
    return _clean_text(QUESTION_PREFIX_RE.sub("", text, count=1))


def _iter_json_ld_nodes(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_json_ld_nodes(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_ld_nodes(item)


def _json_ld_types(node: Dict[str, Any]) -> set[str]:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return {str(value).lower() for value in values if value}


def _html_fragment_text(value: Any) -> str:
    text = str(value or "")
    if "<" in text and ">" in text:
        return _element_text(BeautifulSoup(text, "lxml"))
    return _clean_text(text)


def extract_faqs(
    soup: BeautifulSoup,
    blocks: List[Dict[str, Any]] | None = None,
    structured_data: List[Any] | None = None,
) -> List[Dict[str, str]]:
    """Extract question/answer pairs from common FAQ patterns.

    Supported: <details>/<summary>, schema.org FAQPage JSON-LD, ARIA accordions
    (aria-controls), <dt>/<dd> pairs, and question headings followed by text.
    Headings only count as questions when they end in "?" or start with "Q:".
    """
    faqs: List[Dict[str, str]] = []
    seen: set[str] = set()

    def add(question: str, answer: str) -> None:
        question = _clean_text(question)
        answer = _clean_text(answer)[:MAX_FAQ_ANSWER_CHARS]
        key = question.lower()
        if question and answer and key not in seen and answer.lower() != key:
            faqs.append({"question": question, "answer": answer})
            seen.add(key)

    for details in soup.find_all("details"):
        summary = details.find("summary")
        question = _element_text(summary) if summary else ""
        answer_parts = [
            _element_text(child) if isinstance(child, Tag) else _clean_text(str(child))
            for child in details.children
            if child is not summary and not isinstance(child, _SKIP_STRING_TYPES)
        ]
        add(question, " ".join(part for part in answer_parts if part))

    for node in _iter_json_ld_nodes(structured_data or []):
        if "question" not in _json_ld_types(node):
            continue
        answer = node.get("acceptedAnswer") or node.get("suggestedAnswer")
        if isinstance(answer, list):
            answer = answer[0] if answer else {}
        answer_text = answer.get("text") if isinstance(answer, dict) else answer
        add(_html_fragment_text(node.get("name")), _html_fragment_text(answer_text))

    for trigger in soup.find_all(attrs={"aria-controls": True}):
        question = _element_text(trigger)
        if not _looks_like_question(question):
            continue
        panel = soup.find(id=str(trigger.get("aria-controls")).split()[0])
        if panel is not None and panel is not trigger:
            add(question, _element_text(panel))

    for term in soup.find_all("dt"):
        question = _element_text(term)
        definition = term.find_next_sibling("dd")
        if definition is not None and _looks_like_question(question):
            add(question, _element_text(definition))

    if blocks is None:
        body = soup.find("body") or soup
        blocks = collect_blocks(body)
    for index, block in enumerate(blocks):
        tag = block.get("tag")
        question = block.get("text", "")
        is_heading = tag in HEADING_TAGS
        if is_heading or tag in {"dt", "summary", "button"}:
            if not _looks_like_question(question):
                continue
        elif tag in {"p", "div", "li"}:
            # Ordinary text only counts with an explicit "Q:" style prefix;
            # a sentence such as "Want to know more?" is not an FAQ.
            if not QUESTION_PREFIX_RE.match(question) or len(question) > 300:
                continue
        else:
            continue

        answer_parts: List[str] = []
        for following in blocks[index + 1:]:
            following_text = following.get("text", "")
            if following.get("tag") in HEADING_TAGS:
                break
            if not is_heading and (_looks_like_question(following_text) or QUESTION_PREFIX_RE.match(following_text)):
                break
            answer_parts.append(re.sub(r"^(a|answer)\s*[:.)-]\s*", "", following_text, flags=re.I))
            if len(" ".join(answer_parts)) > MAX_FAQ_ANSWER_CHARS:
                break
            if not is_heading:
                break  # non-heading question: the answer is the next block
        if answer_parts:
            add(_strip_question_prefix(question) if QUESTION_PREFIX_RE.match(question) else question, " ".join(answer_parts))

    return faqs


def extract_json_ld(soup: BeautifulSoup) -> List[Any]:
    records = []
    for script in soup.find_all("script", type=lambda value: value and "ld+json" in value.lower()):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        candidates = [raw, raw.replace("<!--", "").replace("-->", "").strip()]
        for candidate in candidates:
            try:
                records.append(json.loads(candidate, strict=False))
                break
            except json.JSONDecodeError:
                continue
    return records


def extract_sections(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Sections from heading to next heading, in document order (any nesting)."""
    body = soup.find("body") or soup
    return build_sections_from_blocks(collect_blocks(body))


def _find_main_root(soup: BeautifulSoup) -> Tag | None:
    main = soup.find("main") or soup.find(attrs={"role": "main"})
    if main is not None:
        return main
    articles = soup.find_all("article")
    if len(articles) == 1:
        return articles[0]
    return None


def build_clean_text(soup: BeautifulSoup) -> str:
    """Full page text. Uses <main> when it holds most of the page, else <body>.

    The old version used soup.find("article"), which on listing pages (product
    grids, blog indexes) returned only the first card.
    """
    body = soup.find("body") or soup
    return build_content_from_blocks(collect_blocks(body))


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
    extraction_method: str = "static",
    quality_extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Extract a static HTML page into the scraper v2 page schema."""
    if not html:
        quality = create_quality_record(status_code=status_code, errors=errors or [])
        if quality_extra:
            quality.update({key: value for key, value in quality_extra.items() if value not in (None, "")})
        return create_page_record(
            page_url=page_url,
            page_type=page_type,
            extraction_method=extraction_method,
            quality=quality,
        )

    soup = _copy_soup(html)
    requires_javascript = any(
        JS_REQUIRED_PATTERN.search(_element_text(noscript)) for noscript in soup.find_all("noscript")
    )
    page_metadata = extract_page_metadata(soup, page_url)
    structured_data = extract_json_ld(soup)
    images = extract_images(soup, page_url)
    links = extract_links(soup, page_url)
    contacts = extract_contacts(soup)

    remove_noise(soup)
    navigation = extract_and_remove_navigation(soup)

    body = soup.find("body") or soup
    main_root = _find_main_root(soup)
    blocks = collect_blocks(body, main_root=main_root)

    headings = extract_headings(soup)
    paragraphs = extract_paragraphs(soup)
    lists = extract_lists(soup)
    tables = extract_tables(soup)
    faqs = extract_faqs(soup, blocks=blocks, structured_data=structured_data)
    sections = build_sections_from_blocks(blocks)
    content = build_content_from_blocks(blocks)
    core_text = core_text_from_blocks(blocks)

    quality = calculate_page_quality(
        content=content,
        paragraphs=paragraphs,
        headings=headings,
        image_count=len(images),
        link_count=len(links),
        status_code=status_code,
        errors=errors or [],
    )
    quality["core_text_length"] = len(core_text)
    quality["content_hash"] = content_hash(core_text)
    quality["simhash"] = simhash_hex(core_text)
    quality["block_count"] = len(blocks)
    if requires_javascript:
        quality["requires_javascript"] = True
    if quality_extra:
        quality.update({key: value for key, value in quality_extra.items() if value not in (None, "")})

    page = create_page_record(
        page_url=page_url,
        page_type=page_type,
        title=page_metadata["title"],
        description=page_metadata["description"],
        extraction_method=extraction_method,
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
    page["navigation"] = navigation
    page["contacts"] = contacts
    # Temporary, JSON-safe block list used by cross-page processing
    # (boilerplate removal, Markdown export). Stripped before saving.
    page["_blocks"] = blocks
    return page
