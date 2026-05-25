from backend.app.services.scrape_pipeline import create_knowledge_json
from backend.app.services.static_extractor import extract_static_page, normalize_image_url


SAMPLE_HTML = """
<!doctype html>
<html>
  <head>
    <title>Example Co</title>
    <meta name="description" content="Example Co builds useful tools.">
    <link rel="canonical" href="/canonical">
    <meta property="og:title" content="Example Open Graph">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary">
    <script type="application/ld+json">
      {"@context":"https://schema.org","@type":"Organization","name":"Example Co"}
    </script>
    <script>window.app = true;</script>
  </head>
  <body>
    <header>
      <nav><a href="/">Home</a></nav>
    </header>
    <div class="cookie-banner">Accept cookies</div>
    <main>
      <h1>Example Co</h1>
      <p>We build practical software for small teams.</p>
      <h2>Services</h2>
      <p>Our services include automation and support.</p>
      <ul>
        <li>Automation</li>
        <li>Support</li>
      </ul>
      <table>
        <tr><th>Plan</th><th>Price</th></tr>
        <tr><td>Basic</td><td>$10</td></tr>
      </table>
      <details>
        <summary>What do you offer?</summary>
        <p>We offer implementation help.</p>
      </details>
      <h3>How do I start?</h3>
      <p>Contact the team through the form.</p>
      <a href="/docs">Docs</a>
      <a href="https://external.example/resource">External Resource</a>
      <a href="mailto:hello@example.com">Email</a>
      <a href="javascript:void(0)">Do nothing</a>
      <img src="/team.jpg" alt="Team">
    </main>
  </body>
</html>
"""


def test_extract_static_page_metadata_and_structured_data():
    page = extract_static_page(SAMPLE_HTML, page_url="https://example.com/", page_type="homepage")

    assert page["title"] == "Example Co"
    assert page["description"] == "Example Co builds useful tools."
    assert page["metadata"]["canonical"] == "https://example.com/canonical"
    assert page["metadata"]["open_graph"]["title"] == "Example Open Graph"
    assert page["metadata"]["twitter"]["card"] == "summary"
    assert page["structured_data"][0]["@type"] == "Organization"


def test_extract_static_page_headings_paragraphs_lists_tables_and_links():
    page = extract_static_page(SAMPLE_HTML, page_url="https://example.com/", page_type="homepage")

    assert {"level": 1, "text": "Example Co"} in page["headings"]
    assert "We build practical software for small teams." in page["paragraphs"]
    assert page["lists"] == [{"type": "ul", "items": ["Automation", "Support"]}]
    assert page["tables"] == [{"headers": ["Plan", "Price"], "rows": [["Basic", "$10"]]}]

    docs_link = next(link for link in page["links"] if link["text"] == "Docs")
    external_link = next(link for link in page["links"] if link["text"] == "External Resource")
    assert docs_link == {"text": "Docs", "url": "https://example.com/docs", "is_internal": True}
    assert external_link["is_internal"] is False
    assert all(not link["url"].startswith("mailto:") for link in page["links"])
    assert all(not link["url"].startswith("javascript:") for link in page["links"])


def test_extract_static_page_faq_details_and_quality():
    page = extract_static_page(SAMPLE_HTML, page_url="https://example.com/", page_type="homepage")

    assert {"question": "What do you offer?", "answer": "We offer implementation help."} in page["faqs"]
    assert any(faq["question"] == "How do I start?" for faq in page["faqs"])
    assert "Accept cookies" not in page["content"]
    assert page["quality"]["status_code"] == 200
    assert page["quality"]["content_length"] == len(page["content"])
    assert page["quality"]["image_count"] == 1
    assert page["quality"]["link_count"] == len(page["links"])
    assert page["quality"]["useful_text_score"] > 0


IMAGE_HTML = """
<!doctype html>
<html>
  <body>
    <main>
      <section>
        <h2>Meet the team</h2>
        <p>Meet the team behind our company.</p>
        <figure>
          <img
            src="/images/team.jpg"
            alt="Company leadership team"
            title="Leadership"
            width="1200"
            height="800"
            loading="lazy"
          >
          <figcaption>Our executive team</figcaption>
        </figure>
      </section>
      <section>
        <p>Product overview screenshot for the dashboard.</p>
        <img data-src="/images/dashboard.png" alt="Product dashboard screenshot">
      </section>
      <picture>
        <source srcset="/images/hero-large.jpg 1200w, /images/hero-small.jpg 600w">
        <img alt="Hero product cover" title="Hero image">
      </picture>
      <img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" alt="Inline pixel">
      <img src="/pixel.gif" width="1" height="1" alt="">
      <img src="/icons/twitter.svg" width="24" height="24" alt="Twitter">
      <img src="/decorative.jpg">
    </main>
  </body>
</html>
"""


def test_normalize_image_url_supports_relative_and_rejects_inline_images():
    assert normalize_image_url("/images/team.jpg", "https://example.com/about") == (
        "https://example.com/images/team.jpg"
    )
    assert normalize_image_url("//cdn.example.com/team.jpg", "https://example.com/about") == (
        "https://cdn.example.com/team.jpg"
    )
    assert normalize_image_url("data:image/png;base64,abc", "https://example.com/about") == ""


def test_extract_static_page_image_metadata_sources_caption_and_nearby_text():
    page = extract_static_page(IMAGE_HTML, page_url="https://example.com/about", page_type="about")

    team_image = next(image for image in page["images"] if image["absolute_url"].endswith("/team.jpg"))
    assert team_image["src"] == "/images/team.jpg"
    assert team_image["absolute_url"] == "https://example.com/images/team.jpg"
    assert team_image["alt"] == "Company leadership team"
    assert team_image["title"] == "Leadership"
    assert team_image["figcaption"] == "Our executive team"
    assert team_image["width"] == 1200
    assert team_image["height"] == 800
    assert team_image["loading"] == "lazy"
    assert team_image["page_url"] == "https://example.com/about"
    assert team_image["image_type"] == "team"
    assert "Meet the team behind our company." in team_image["nearby_text"]

    dashboard_image = next(
        image for image in page["images"] if image["absolute_url"].endswith("/dashboard.png")
    )
    assert dashboard_image["src"] == "/images/dashboard.png"
    assert dashboard_image["image_type"] == "product"

    hero_image = next(image for image in page["images"] if image["absolute_url"].endswith("/hero-large.jpg"))
    assert hero_image["src"] == "/images/hero-large.jpg"
    assert hero_image["image_type"] == "hero"


def test_extract_static_page_filters_noisy_images_and_updates_quality_count():
    page = extract_static_page(IMAGE_HTML, page_url="https://example.com/about", page_type="about")

    image_urls = [image["absolute_url"] for image in page["images"]]
    assert len(page["images"]) == 3
    assert page["quality"]["image_count"] == 3
    assert all("pixel" not in image_url for image_url in image_urls)
    assert all("twitter" not in image_url for image_url in image_urls)
    assert all("decorative" not in image_url for image_url in image_urls)


def test_create_knowledge_json_keeps_v2_pages_and_legacy_primary_pages():
    page = extract_static_page(SAMPLE_HTML, page_url="https://example.com/", page_type="homepage")
    knowledge = create_knowledge_json(
        "https://example.com/",
        {"total_pages": 1, "pages": [page]},
        web_search_results=[],
        name="Example Co",
    )

    assert knowledge["metadata"]["scraping_version"] == "v2"
    assert knowledge["pages"][0]["page_url"] == "https://example.com/"
    assert "headings" in knowledge["pages"][0]
    assert knowledge["primary_content"]["pages"][0] == {
        "title": "Example Co",
        "description": "Example Co builds useful tools.",
        "sections": knowledge["pages"][0]["sections"],
        "content": knowledge["pages"][0]["content"],
        "url": "https://example.com/",
        "page_type": "homepage",
    }
