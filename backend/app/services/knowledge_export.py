"""Export crawled knowledge as JSONL and Markdown (with source references).

Layout, next to the knowledge JSON file::

    knowledge_files/exports/<knowledge-file-stem>/
        pages.jsonl          one JSON object per page (clean page record + markdown)
        pages/NNN-slug.md    one Markdown file per page with YAML front matter
        crawl_report.json    crawl statistics, skipped/failed URLs
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .static_extractor import HEADING_TAGS


def _slug(text: str, fallback: str = "page") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:60] or fallback


def _yaml_value(value: Any) -> str:
    return json.dumps("" if value is None else value, ensure_ascii=False)


def _table_markdown(rows: List[List[str]]) -> List[str]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    padded = [[cell.replace("|", "\\|") for cell in row] + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(padded[0]) + " |", "|" + " --- |" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return lines


def blocks_to_markdown(blocks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    table_rows: List[List[str]] = []
    table_group = None

    def flush_table() -> None:
        nonlocal table_rows, table_group
        if table_rows:
            lines.extend(_table_markdown(table_rows))
            lines.append("")
        table_rows = []
        table_group = None

    for block in blocks:
        tag = block.get("tag", "")
        text = (block.get("text") or "").strip()
        if not text:
            continue
        if tag == "tr":
            if table_group is not None and block.get("group") != table_group:
                flush_table()
            table_group = block.get("group")
            table_rows.append([cell.strip() for cell in text.split(" | ")])
            continue
        flush_table()
        if tag in HEADING_TAGS:
            level = min(6, max(2, int(block.get("level") or 2)))
            lines.extend([f"{'#' * level} {text}", ""])
        elif tag == "li":
            lines.append(f"- {text}")
        elif tag == "blockquote":
            lines.extend([f"> {text}", ""])
        elif tag == "pre":
            lines.extend(["```", text, "```", ""])
        else:
            if lines and lines[-1].startswith("- "):
                lines.append("")
            lines.extend([text, ""])
    flush_table()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def page_to_markdown(page: Dict[str, Any]) -> str:
    quality = page.get("quality") or {}
    front_matter = [
        "---",
        f"source_url: {_yaml_value(page.get('page_url'))}",
        f"title: {_yaml_value(page.get('title'))}",
        f"page_type: {_yaml_value(page.get('page_type'))}",
        f"extraction_method: {_yaml_value(page.get('extraction_method'))}",
        f"scraped_at: {_yaml_value(quality.get('scraped_at'))}",
        "---",
        "",
        f"# {page.get('title') or page.get('page_url') or 'Untitled page'}",
        "",
    ]
    if page.get("description"):
        front_matter.extend([f"> {page['description']}", ""])

    blocks = page.get("_blocks")
    if isinstance(blocks, list) and blocks:
        body = blocks_to_markdown(blocks)
    else:
        parts = []
        for section in page.get("sections") or []:
            if section.get("heading"):
                parts.append(f"## {section['heading']}\n")
            parts.append(f"{section.get('content', '')}\n")
        body = "\n".join(parts) or (page.get("content") or "")

    extras: List[str] = []
    faqs = page.get("faqs") or []
    if faqs:
        extras.extend(["", "## Frequently asked questions", ""])
        for faq in faqs:
            extras.extend([f"**Q: {faq.get('question', '')}**", "", f"{faq.get('answer', '')}", ""])
    return "\n".join(front_matter) + body + "\n".join(extras).rstrip() + "\n"


def _clean_page(page: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in page.items() if not str(key).startswith("_")}


def export_crawl_outputs(
    scraped_pages: List[Dict[str, Any]],
    knowledge: Dict[str, Any],
    knowledge_json_path: str | Path,
    crawl_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Write pages.jsonl, per-page Markdown and crawl_report.json. Best effort."""
    json_path = Path(knowledge_json_path)
    export_dir = json_path.parent / "exports" / json_path.stem
    pages_dir = export_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for old_file in pages_dir.glob("*.md"):
        old_file.unlink()

    metadata = (knowledge or {}).get("metadata", {}) if isinstance(knowledge, dict) else {}
    website_id = metadata.get("website_id", "")
    jsonl_path = export_dir / "pages.jsonl"
    written = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, page in enumerate(scraped_pages or []):
            if not isinstance(page, dict):
                continue
            markdown = page_to_markdown(page)
            record = {"website_id": website_id, "page_index": index, **_clean_page(page), "markdown": markdown}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            name = f"{index:03d}-{_slug(page.get('title') or page.get('page_url', ''))}.md"
            (pages_dir / name).write_text(markdown, encoding="utf-8")
            written += 1

    report_path = export_dir / "crawl_report.json"
    report_path.write_text(
        json.dumps({"website_id": website_id, "url": metadata.get("url", ""), **(crawl_report or {})}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "export_dir": str(export_dir),
        "jsonl_path": str(jsonl_path),
        "markdown_pages": written,
        "crawl_report_path": str(report_path),
    }
