"""Regression tests for extraction bugs found in the November 2026 audit.

Each test reproduces a failure the old extractor had on real-world markup.
"""

from backend.app.services.static_extractor import extract_static_page

MAIN = "<main><h1>Admissions</h1><p>Fee is PKR 500,000 per semester. Apply before June 30.</p></main>"


def _page(html: str, url: str = "https://example.com/page") -> dict:
    return extract_static_page(html, page_url=url)


def test_noise_class_on_body_or_wrapper_does_not_wipe_the_page():
    wrappers = [
        f"<html><body class='elementor-default has-popup'>{MAIN}</body></html>",
        f"<html><body><div id='modal-root-wrapper'>{MAIN}</div></body></html>",
        f"<html><body><div class='site cookie-consent-active'>{MAIN}</div></body></html>",
    ]
    for html in wrappers:
        page = _page(html)
        assert "PKR 500,000" in page["content"], html


def test_class_names_containing_ad_substrings_are_not_noise():
    page = _page(
        "<html><body><main><div class='page-head-title'><h1>Fees</h1></div>"
        "<p class='read-more'>Read the full fee structure for 2026.</p>"
        "<section class='downloads'><p>Prospectus 2026 available for all applicants.</p></section>"
        "</main></body></html>"
    )
    assert "Read the full fee structure" in page["content"]
    assert "Prospectus 2026" in page["content"]
    assert page["sections"][0]["heading"] == "Fees"


def test_real_overlays_are_still_removed():
    page = _page(
        "<html><body><main><h1>Shop</h1><p>Summer collection now available in all stores.</p></main>"
        "<div class='newsletter-popup'><h2>Subscribe</h2><p>Get 10% off your first order.</p></div>"
        "<div role='dialog' aria-modal='true'><p>We use cookies.</p></div>"
        "<div class='ad-slot'>Sponsored link</div></body></html>"
    )
    assert "Summer collection" in page["content"]
    assert "10% off" not in page["content"]
    assert "We use cookies" not in page["content"]
    assert "Sponsored link" not in page["content"]


def test_sections_work_when_headings_and_text_are_not_siblings():
    page = _page(
        "<html><body><main><div class='card'><div><h2>Our Services</h2></div>"
        "<div><span>We build websites and mobile apps for SMEs across Pakistan.</span></div></div>"
        "<div class='card'><div><h3>Pricing</h3></div><div><div>Plans start at Rs 25,000.</div></div></div>"
        "</main></body></html>"
    )
    sections = {section["heading"]: section["content"] for section in page["sections"]}
    assert sections["Our Services"] == "We build websites and mobile apps for SMEs across Pakistan."
    assert sections["Pricing"] == "Plans start at Rs 25,000."


def test_listing_pages_keep_every_card_not_only_the_first_article():
    cards = "".join(f"<article><h3>Book {i}</h3><p>£{i}.00 In stock</p></article>" for i in range(20))
    page = _page(f"<html><body>{cards}</body></html>")
    assert all(f"Book {i}" in page["content"] for i in range(20))


def test_long_pages_are_not_truncated_at_12000_characters():
    text = " ".join(f"Sentence {i} states fact {i * 7}." for i in range(3000))
    page = _page(f"<html><body><main><p>{text}</p></main></body></html>")
    assert len(page["content"]) > 50_000
    assert "fact 20993" in page["content"]


def test_faq_requires_a_real_question():
    page = _page(
        "<html><body><main>"
        "<h2>Who We Are</h2><p>A software house in Lahore.</p>"
        "<h2>Our Products Are Great</h2><p>Fast and reliable.</p>"
        "<h2>Can I pay in installments?</h2><div><p>Yes, in 3 monthly installments.</p></div>"
        "<p>Want to know more?</p><p>Call us.</p>"
        "</main></body></html>"
    )
    assert [faq["question"] for faq in page["faqs"]] == ["Can I pay in installments?"]
    assert page["faqs"][0]["answer"].startswith("Yes, in 3 monthly installments.")


def test_faq_from_accordion_and_json_ld():
    page = _page(
        """<html><head><script type="application/ld+json">
        {"@context":"https://schema.org","@graph":[{"@type":"FAQPage","mainEntity":[
          {"@type":"Question","name":"Do you deliver to Quetta?",
           "acceptedAnswer":{"@type":"Answer","text":"<p>Yes, within 5 days.</p>"}}]}]}
        </script></head><body>
        <button aria-controls="faq-1" aria-expanded="false">What is the return period?</button>
        <div id="faq-1" hidden><p>14 days from delivery.</p></div>
        </body></html>"""
    )
    faqs = {faq["question"]: faq["answer"] for faq in page["faqs"]}
    assert faqs["Do you deliver to Quetta?"] == "Yes, within 5 days."
    assert faqs["What is the return period?"] == "14 days from delivery."


def test_navigation_is_removed_from_content_but_kept_as_labels():
    page = _page(
        "<html><body><nav><ul><li><a href='/a'>Admissions</a></li><li><a href='/cs'>BS Computer Science</a></li></ul></nav>"
        "<main><p>Real content about admission deadlines and fees.</p></main></body></html>"
    )
    assert "BS Computer Science" not in page["content"]
    assert page["navigation"] == ["Admissions", "BS Computer Science"]
    assert any(link["url"].endswith("/cs") for link in page["links"])  # still used for crawling


def test_restaurant_menu_is_content_not_navigation():
    page = _page(
        "<html><body><div class='menu'><h2>Lunch</h2><p>Chicken Karahi Rs 1200</p><p>Daal Rs 400</p></div></body></html>"
    )
    assert "Chicken Karahi Rs 1200" in page["content"]


def test_row_header_tables_and_empty_cells_keep_alignment():
    page = _page(
        "<html><body><table>"
        "<tr><th>Fee</th><td>500</td></tr>"
        "<tr><th>Hostel</th><td></td></tr>"
        "</table></body></html>"
    )
    assert page["tables"] == [{"headers": [], "rows": [["Fee", "500"], ["Hostel", ""]]}]


def test_contacts_from_icon_only_links():
    page = _page(
        "<html><body><p>Reach us</p><a href='mailto:info@example.edu.pk'></a>"
        "<a href='tel:+92-938-281026'><img src='/phone.png'></a>"
        "<a href='https://wa.me/923001234567'></a></body></html>"
    )
    assert page["contacts"] == {
        "emails": ["info@example.edu.pk"],
        "phones": ["+92938281026"],
        "whatsapp": ["https://wa.me/923001234567"],
    }


def test_inline_markup_does_not_split_or_fuse_words():
    page = _page("<html><body><p>Hello <b>world</b>, again.</p><div><span>Price</span><span>Rs 500</span></div></body></html>")
    assert "Hello world, again." in page["content"]
    assert "Price Rs 500" in page["content"]


def test_quality_has_fingerprints_and_js_hint():
    page = _page(
        "<html><body><noscript>You need to enable JavaScript to run this app.</noscript><div id='root'></div></body></html>"
    )
    assert page["quality"]["requires_javascript"] is True
    rich = _page(f"<html><body>{MAIN}</body></html>")
    assert rich["quality"]["content_hash"]
    assert len(rich["quality"]["simhash"]) == 16


def test_trafilatura_cross_check_records_coverage_and_restores_dropped_text():
    body = "".join(
        f"<p>The robotics lab section {i} describes twelve workstations, safety training and two industrial arms.</p>"
        for i in range(6)
    )
    page = _page(
        "<html><head><title>Committees</title></head><body><article><h1>Committees</h1>" + body +
        "<section class='ad-hoc-committee'><h2>Ad-hoc committee</h2><p>The ad-hoc committee reviews fee waiver "
        "requests every Friday at 2 pm in room B-12.</p></section></article></body></html>"
    )
    quality = page["quality"]
    # The structural pass reads class token "ad" as an advert and drops the
    # section; Trafilatura sees it as main text, so it is restored.
    assert "fee waiver requests every Friday" in page["content"]
    assert quality["main_text_recovered_paragraphs"] == 1
    assert 0.5 < quality["main_text_coverage"] < 1
    clean = _page(f"<html><body>{MAIN}</body></html>")
    assert clean["quality"].get("main_text_recovered_paragraphs", 0) == 0
