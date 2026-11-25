import json

from backend.app.services.boilerplate import build_site_wide_page, remove_cross_page_boilerplate
from backend.app.services.document_extractor import detect_document_kind, extract_document_page
from backend.app.services.knowledge_export import export_crawl_outputs, page_to_markdown
from backend.app.services.page_fetcher import FetchResult, as_fetch_result, decode_html, parse_retry_after
from backend.app.services.static_extractor import extract_static_page
from tests.helpers import make_docx, make_pdf


def test_detect_document_kind_uses_mime_magic_and_extension():
    assert detect_document_kind("application/pdf", "https://x.com/a") == "pdf"
    assert detect_document_kind("application/octet-stream", "https://x.com/a", b"%PDF-1.7") == "pdf"
    assert detect_document_kind("", "https://x.com/Prospectus.DOCX") == "docx"
    assert detect_document_kind("text/html", "https://x.com/page") == ""


def test_pdf_and_docx_become_document_pages():
    pdf_page = extract_document_page(make_pdf("Admission test is on 12 July 2026."), "pdf", "https://x.com/notice.pdf")
    assert pdf_page["page_type"] == "document"
    assert pdf_page["extraction_method"] == "pdf"
    assert "Admission test is on 12 July 2026." in pdf_page["content"]

    docx_page = extract_document_page(make_docx(), "docx", "https://x.com/rules.docx")
    sections = {section["heading"]: section["content"] for section in docx_page["sections"]}
    assert sections["Visitors"] == "Visitors are allowed until 7 pm."
    assert docx_page["tables"] == [{"headers": ["Room", "Fee"], "rows": [["Single", "Rs 40,000"]]}]


def test_fetch_result_is_backward_compatible_tuple():
    result = FetchResult("https://x.com", "<p>hi</p>", "", final_url="https://www.x.com/", status=200)
    url, html, error = result
    assert (url, html, error) == ("https://x.com", "<p>hi</p>", "")
    assert result.final_url == "https://www.x.com/"
    stub = as_fetch_result(("https://x.com", "<p>hi</p>", ""), "https://x.com")
    assert stub.status == 200 and stub.kind == "html"


def test_decode_html_and_retry_after():
    assert decode_html('<meta charset="windows-1252"><p>caf\xe9</p>'.encode("cp1252")).endswith("café</p>")
    assert decode_html(b"<p>\xff\xfe broken</p>", "utf-8").endswith("broken</p>")  # never raises
    assert parse_retry_after("7") == 7
    assert parse_retry_after("not-a-date") == 5
    assert parse_retry_after("99999") == 30


def _site_page(url: str, body: str) -> dict:
    html = (
        "<html><body><header><p>Call 0311-1115262 · Free delivery over Rs 3000</p></header>"
        f"<main>{body}</main><footer><h2>Quick links</h2><p>© 2026 Example Store. All rights reserved.</p></footer></body></html>"
    )
    return extract_static_page(html, page_url=url, page_type="other")


def test_cross_page_boilerplate_is_removed_and_kept_once():
    pages = [
        _site_page(f"https://x.com/p{i}", f"<h1>Product {i}</h1><h2>Description</h2><p>Unique detail number {i} about fabric.</p>")
        for i in range(5)
    ]
    stats = remove_cross_page_boilerplate(pages, min_pages=4, ratio=0.5)
    assert stats["blocks_removed"] > 0
    for i, page in enumerate(pages):
        assert "0311-1115262" not in page["content"]
        assert "All rights reserved" not in page["content"]
        assert f"Unique detail number {i}" in page["content"]
        # The repeated "Description" heading stays because its text is unique.
        assert any(section["heading"] == "Description" for section in page["sections"])

    site_wide = build_site_wide_page(pages, stats["site_wide_blocks"], "https://x.com", site_name="Example Store")
    assert site_wide["page_type"] == "site_wide"
    assert "0311-1115262" in site_wide["content"]
    assert "All rights reserved" in site_wide["content"]


def test_markdown_and_jsonl_export(tmp_path):
    page = extract_static_page(
        "<html><head><title>Fees</title></head><body><h1>Fees</h1><p>Per semester fees.</p>"
        "<ul><li>Tuition</li><li>Hostel</li></ul>"
        "<table><tr><th>Item</th><th>Rs</th></tr><tr><td>Tuition</td><td>500,000</td></tr></table></body></html>",
        page_url="https://x.com/fees",
    )
    markdown = page_to_markdown(page)
    assert 'source_url: "https://x.com/fees"' in markdown
    assert "- Tuition" in markdown
    assert "| Item | Rs |" in markdown and "| Tuition | 500,000 |" in markdown

    json_path = tmp_path / "x_com_abc.json"
    result = export_crawl_outputs([page], {"metadata": {"website_id": "abc", "url": "https://x.com"}}, json_path, {"pages_stored": 1})
    lines = (tmp_path / "exports" / "x_com_abc" / "pages.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["website_id"] == "abc" and record["page_url"] == "https://x.com/fees"
    assert "_blocks" not in record and record["markdown"].startswith("---")
    assert result["markdown_pages"] == 1
