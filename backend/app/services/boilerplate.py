"""Cross-page template (boilerplate) removal.

A site's header, footer, menus and repeated widgets appear on almost every
page. Any text block found on at least ``ratio`` of the crawled pages is
treated as template: it is removed from each page and kept *once* in a
``site_wide`` page record, so information such as footer contact details is
not lost but is no longer repeated in every page's knowledge.

This is site-agnostic: it needs no per-site selectors.
"""

import math
import re
from typing import Any, Dict, List, Tuple

from .scraper_schema import create_page_record, create_quality_record
from .static_extractor import (
    HEADING_TAGS,
    build_content_from_blocks,
    build_sections_from_blocks,
)

EXCLUDED_PAGE_TYPES = {"document", "site_wide"}


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _template_keys(pages: List[Dict[str, Any]], min_pages: int, ratio: float) -> Tuple[set, int]:
    eligible = [page for page in pages if isinstance(page.get("_blocks"), list)]
    if len(eligible) < max(2, min_pages):
        return set(), 0
    threshold = max(3, math.ceil(ratio * len(eligible)))
    frequency: Dict[str, int] = {}
    for page in eligible:
        for key in {_key(block.get("text", "")) for block in page["_blocks"]}:
            if key:
                frequency[key] = frequency.get(key, 0) + 1
    return {key for key, count in frequency.items() if count >= threshold}, threshold


def _template_flags(blocks: List[Dict[str, Any]], template: set) -> List[bool]:
    """Flag template blocks. A heading is template only if its whole section is."""
    flags = [_key(block.get("text", "")) in template for block in blocks]
    for index, block in enumerate(blocks):
        if block.get("tag") not in HEADING_TAGS or not flags[index]:
            continue
        section_flags = []
        for next_index in range(index + 1, len(blocks)):
            if blocks[next_index].get("tag") in HEADING_TAGS:
                break
            section_flags.append(flags[next_index])
        if section_flags and not all(section_flags):
            flags[index] = False  # e.g. a repeated "Description" heading over unique text
    return flags


def remove_cross_page_boilerplate(
    pages: List[Dict[str, Any]],
    min_pages: int = 4,
    ratio: float = 0.5,
) -> Dict[str, Any]:
    """Strip template blocks from pages in place.

    Returns stats plus ``site_wide_blocks`` (template blocks in first-seen order).
    """
    html_pages = [page for page in pages if page.get("page_type") not in EXCLUDED_PAGE_TYPES]
    template, threshold = _template_keys(html_pages, min_pages, ratio)
    stats: Dict[str, Any] = {
        "pages_considered": len(html_pages),
        "threshold_pages": threshold,
        "template_blocks": len(template),
        "blocks_removed": 0,
        "site_wide_blocks": [],
    }
    if not template:
        return stats

    site_wide_blocks: List[Dict[str, Any]] = []
    site_wide_seen: set = set()
    for page in html_pages:
        blocks = page.get("_blocks")
        if not isinstance(blocks, list):
            continue
        flags = _template_flags(blocks, template)
        kept = [block for block, is_template in zip(blocks, flags) if not is_template]
        removed = len(blocks) - len(kept)
        for block, is_template in zip(blocks, flags):
            key = _key(block.get("text", ""))
            if is_template and key not in site_wide_seen:
                site_wide_seen.add(key)
                site_wide_blocks.append(dict(block))
        if not removed:
            continue
        stats["blocks_removed"] += removed

        page["_blocks"] = kept
        page["sections"] = build_sections_from_blocks(kept)
        page["content"] = build_content_from_blocks(kept)
        page["paragraphs"] = [text for text in page.get("paragraphs", []) if _key(text) not in template]
        page["lists"] = [
            item for item in page.get("lists", [])
            if not all(_key(entry) in template for entry in item.get("items", []))
        ]
        quality = page.setdefault("quality", {})
        quality["content_length"] = len(page["content"])
        quality["boilerplate_blocks_removed"] = removed

    stats["site_wide_blocks"] = site_wide_blocks
    return stats


def _merge_contacts(pages: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {"emails": [], "phones": [], "whatsapp": []}
    for page in pages:
        contacts = page.get("contacts") or {}
        for key in merged:
            for value in contacts.get(key, []) or []:
                if value and value not in merged[key]:
                    merged[key].append(value)
    return merged


def build_site_wide_page(
    pages: List[Dict[str, Any]],
    site_wide_blocks: List[Dict[str, Any]],
    base_url: str,
    site_name: str = "",
) -> Dict[str, Any] | None:
    """Build one record for header/footer/navigation/contact information."""
    homepage = next((page for page in pages if page.get("page_type") == "homepage"), pages[0] if pages else {})
    navigation: List[str] = []
    for page in [homepage, *pages]:
        for item in page.get("navigation", []) or []:
            if item not in navigation:
                navigation.append(item)
    contacts = _merge_contacts(pages)

    blocks: List[Dict[str, Any]] = [dict(block) for block in site_wide_blocks]
    contact_lines = []
    if contacts["emails"]:
        contact_lines.append("Email: " + ", ".join(contacts["emails"]))
    if contacts["phones"]:
        contact_lines.append("Phone: " + ", ".join(contacts["phones"]))
    if contacts["whatsapp"]:
        contact_lines.append("WhatsApp: " + ", ".join(contacts["whatsapp"]))
    if contact_lines:
        blocks.append({"text": "Contact details", "tag": "h2", "level": 2, "chrome": True, "in_main": False, "group": 0})
        blocks.extend({"text": line, "tag": "p", "level": 0, "chrome": True, "in_main": False, "group": 0} for line in contact_lines)
    if navigation:
        blocks.append({"text": "Site navigation", "tag": "h2", "level": 2, "chrome": True, "in_main": False, "group": 0})
        blocks.extend({"text": item, "tag": "li", "level": 0, "chrome": True, "in_main": False, "group": 0} for item in navigation)

    if not blocks:
        return None
    content = build_content_from_blocks(blocks)
    label = site_name or "this website"
    page = create_page_record(
        page_url=base_url,
        page_type="site_wide",
        title=f"{label}: site-wide information (header, footer, navigation, contacts)",
        extraction_method="cross_page_template",
        sections=build_sections_from_blocks(blocks, intro_heading="Shown on every page"),
        lists=[{"type": "ul", "items": navigation}] if navigation else [],
        quality=create_quality_record(content_length=len(content), useful_text_score=1.0),
        content=content,
    )
    page["navigation"] = navigation
    page["contacts"] = contacts
    page["_blocks"] = blocks
    return page
