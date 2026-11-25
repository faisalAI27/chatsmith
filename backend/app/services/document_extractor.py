"""Text extraction for linked public documents (PDF and Word .docx).

Uses pdfplumber / python-docx (the libraries named in the project scope) and
falls back to pypdf / raw DOCX XML so extraction still works when an optional
library is missing. Scanned PDFs without a text layer produce no text; OCR is
out of scope here.
"""

import io
import re
import zipfile
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from .scraper_schema import create_page_record, create_quality_record
from .text_fingerprint import content_hash, simhash_hex

PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
DOCX_CONTENT_TYPES = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
DOCUMENT_EXTENSIONS = {".pdf": "pdf", ".docx": "docx"}
MAX_PDF_PAGES = 60
MAX_DOCUMENT_CHARS = 300_000
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def detect_document_kind(content_type: str, url: str, head: bytes = b"") -> str:
    """Return "pdf", "docx", or "" from MIME type, magic bytes, or URL extension."""
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if content_type in PDF_CONTENT_TYPES or head.startswith(b"%PDF"):
        return "pdf"
    if content_type in DOCX_CONTENT_TYPES:
        return "docx"
    path = urlparse(url or "").path.lower()
    for extension, kind in DOCUMENT_EXTENSIONS.items():
        if path.endswith(extension):
            if kind == "docx" and head and not head.startswith(b"PK"):
                return ""
            if kind == "pdf" and head and not head.startswith(b"%PDF"):
                return ""
            return kind
    return ""


def _title_from_url(url: str) -> str:
    name = unquote(urlparse(url or "").path.rsplit("/", 1)[-1])
    name = re.sub(r"\.(pdf|docx)$", "", name, flags=re.I)
    return _clean_text(re.sub(r"[-_]+", " ", name))


def _paragraphs_from_pdf_text(text: str) -> List[str]:
    """Join wrapped PDF lines back into paragraphs."""
    paragraphs: List[str] = []
    current: List[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if current and current[-1].endswith("-") and line[:1].islower():
            current[-1] = current[-1][:-1] + line
            continue
        current.append(line)
        if line.endswith((".", ":", "?", "!")) and len(line) < 60:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return [_clean_text(paragraph) for paragraph in paragraphs if _clean_text(paragraph)]


def _extract_pdf(body: bytes) -> Dict[str, Any]:
    pages: List[str] = []
    tables: List[Dict[str, Any]] = []
    title = ""
    page_count = 0
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(body)) as pdf:
            page_count = len(pdf.pages)
            title = _clean_text(str((pdf.metadata or {}).get("Title") or ""))
            for page in pdf.pages[:MAX_PDF_PAGES]:
                pages.append(page.extract_text() or "")
                for raw_table in page.extract_tables() or []:
                    rows = [[_clean_text(cell or "") for cell in row] for row in raw_table if row and any(row)]
                    if len(rows) >= 2:
                        tables.append({"headers": rows[0], "rows": rows[1:]})
        method = "pdfplumber"
    except Exception:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(body))
        page_count = len(reader.pages)
        title = _clean_text(str((reader.metadata or {}).get("/Title") or ""))
        pages = [(page.extract_text() or "") for page in reader.pages[:MAX_PDF_PAGES]]
        method = "pypdf"
    return {"pages": pages, "tables": tables, "title": title, "page_count": page_count, "method": method}


def _extract_docx(body: bytes) -> Dict[str, Any]:
    """Return ordered (style, text) paragraphs and tables from a .docx file."""
    paragraphs: List[tuple[str, str]] = []
    tables: List[Dict[str, Any]] = []
    title = ""
    try:
        import docx

        document = docx.Document(io.BytesIO(body))
        title = _clean_text(document.core_properties.title or "")
        for paragraph in document.paragraphs:
            text = _clean_text(paragraph.text)
            if text:
                paragraphs.append(((paragraph.style.name if paragraph.style is not None else "") or "", text))
        for table in document.tables:
            rows = [[_clean_text(cell.text) for cell in row.cells] for row in table.rows]
            rows = [row for row in rows if any(row)]
            if rows:
                tables.append({"headers": rows[0], "rows": rows[1:]})
        method = "python-docx"
    except Exception:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        for node in root.iter(f"{_WORD_NS}p"):
            style_node = node.find(f"{_WORD_NS}pPr/{_WORD_NS}pStyle")
            style = style_node.get(f"{_WORD_NS}val", "") if style_node is not None else ""
            text = _clean_text("".join(t.text or "" for t in node.iter(f"{_WORD_NS}t")))
            if text:
                paragraphs.append((style, text))
        method = "docx-xml"
    return {"paragraphs": paragraphs, "tables": tables, "title": title, "method": method}


def _heading_level(style: str) -> int:
    match = re.search(r"heading\s*([1-6])", style or "", re.I)
    if match:
        return int(match.group(1))
    if (style or "").lower() == "title":
        return 1
    return 0


def extract_document_page(
    body: bytes,
    kind: str,
    page_url: str,
    status_code: int = 200,
) -> Dict[str, Any] | None:
    """Turn PDF/DOCX bytes into a v2 page record (page_type "document")."""
    try:
        if kind == "pdf":
            extracted = _extract_pdf(body)
            blocks: List[Dict[str, Any]] = []
            for number, page_text in enumerate(extracted["pages"], start=1):
                paragraphs = _paragraphs_from_pdf_text(page_text)
                if not paragraphs:
                    continue
                if len(extracted["pages"]) > 1:
                    blocks.append({"text": f"Page {number}", "tag": "h2", "level": 2, "chrome": False, "in_main": True, "group": 0})
                blocks.extend(
                    {"text": paragraph, "tag": "p", "level": 0, "chrome": False, "in_main": True, "group": 0}
                    for paragraph in paragraphs
                )
            document_info = {"kind": "pdf", "page_count": extracted["page_count"], "parser": extracted["method"]}
        elif kind == "docx":
            extracted = _extract_docx(body)
            blocks = []
            for style, text in extracted["paragraphs"]:
                level = _heading_level(style)
                is_list = "list" in (style or "").lower()
                blocks.append({
                    "text": text,
                    "tag": f"h{level}" if level else ("li" if is_list else "p"),
                    "level": level,
                    "chrome": False,
                    "in_main": True,
                    "group": 0,
                })
            document_info = {"kind": "docx", "parser": extracted["method"]}
        else:
            return None
    except Exception as exc:
        print(f"  ⚠️ Document extraction failed for {page_url[:60]}: {exc}")
        return None

    from .static_extractor import build_content_from_blocks, build_sections_from_blocks

    title = extracted.get("title") or ""
    if not title:
        first_heading = next((block["text"] for block in blocks if block["level"]), "")
        title = first_heading or _title_from_url(page_url)
    content = build_content_from_blocks(blocks)[:MAX_DOCUMENT_CHARS]
    paragraphs = [block["text"] for block in blocks if block["tag"] == "p"]
    headings = [{"level": block["level"], "text": block["text"]} for block in blocks if block["level"]]
    quality = create_quality_record(
        status_code=status_code,
        content_length=len(content),
        useful_text_score=round(min(1.0, len(content) / 3000), 3),
    )
    quality["core_text_length"] = len(content)
    quality["content_hash"] = content_hash(content)
    quality["simhash"] = simhash_hex(content)
    quality["block_count"] = len(blocks)
    if not content:
        quality["errors"].append("no_text_layer (scanned document or image-only PDF; OCR not enabled)")

    page = create_page_record(
        page_url=page_url,
        page_type="document",
        title=title,
        extraction_method=kind,
        headings=headings,
        sections=build_sections_from_blocks(blocks),
        paragraphs=paragraphs,
        tables=extracted.get("tables") or [],
        metadata={"document": document_info},
        quality=quality,
        content=content,
    )
    page["navigation"] = []
    page["contacts"] = {"emails": [], "phones": [], "whatsapp": []}
    page["_blocks"] = blocks
    return page
