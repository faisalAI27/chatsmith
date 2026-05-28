import hashlib
import json
from pathlib import Path

from backend.app.services import scrape_pipeline as sp


def test_normalize_url_for_cache_groups_common_variants():
    expected = "https://example.com"

    assert sp.normalize_url_for_cache("https://Example.com/") == expected
    assert sp.normalize_url_for_cache("https://example.com") == expected
    assert sp.normalize_url_for_cache("https://example.com/?utm_source=x") == expected
    assert sp.normalize_url_for_cache("http://example.com#section") == expected
    assert sp.normalize_url_for_cache("example.com/path/?utm_source=x#section") == "https://example.com/path"


def test_cache_path_uses_official_dir_and_normalized_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "KNOWLEDGE_DIR", tmp_path)

    first = Path(sp.get_cache_path("https://Example.com/?utm_source=x"))
    second = Path(sp.get_cache_path("https://example.com/"))

    assert first == second
    assert first.parent == tmp_path
    assert first.name.startswith("example_com_")


def test_website_id_generation_uses_normalized_url():
    normalized_url = sp.normalize_url_for_cache("https://Example.com/?utm_source=x")
    expected = hashlib.md5(normalized_url.encode("utf-8")).hexdigest()[:12]

    assert sp.get_website_id("https://example.com/") == expected


def test_old_json_metadata_compatibility(tmp_path):
    cache_file = tmp_path / "old.json"
    old_knowledge = {
        "metadata": {
            "url": "https://Example.com/?utm_source=x",
            "name": "Example",
            "created_at": "2026-01-01T00:00:00",
            "pages_scraped": 1,
            "has_web_search_supplement": False,
        },
        "primary_content": {"pages": []},
        "secondary_content": {"searches": []},
    }
    cache_file.write_text(json.dumps(old_knowledge), encoding="utf-8")

    loaded = sp.load_knowledge_json(str(cache_file))
    metadata = loaded["metadata"]

    assert metadata["url"] == "https://Example.com/?utm_source=x"
    assert metadata["normalized_url"] == "https://example.com"
    assert metadata["website_id"] == sp.get_website_id("https://example.com")
