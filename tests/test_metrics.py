import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.services import metrics_logger as ml  # noqa: E402
from backend.app.services import scrape_pipeline as sp  # noqa: E402


class DummyProgress:
    def __call__(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_cache_hit_metrics(monkeypatch):
    monkeypatch.setattr(sp, "ENABLE_METRICS", True)
    monkeypatch.setattr(sp.gr, "update", lambda **kwargs: {"update": kwargs})
    monkeypatch.setattr(sp, "is_cached", lambda url: True)
    monkeypatch.setattr(
        sp,
        "get_cached_knowledge",
        lambda url: {"metadata": {"name": "CachedSite", "url": url, "pages_scraped": 2}},
    )
    monkeypatch.setattr(sp, "knowledge_to_chatbot_context", lambda knowledge: "ctx")
    monkeypatch.setattr(sp, "build_status_new", lambda *args, **kwargs: "status")

    result = await sp.run_full_research_new("https://example.com", progress=DummyProgress())
    _, _, _, _, _, _, stats = result

    assert stats["cache_hit"] is True
    assert "tcr_seconds" in stats and stats["tcr_seconds"] >= 0


@pytest.mark.asyncio
async def test_tcr_metrics_non_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "ENABLE_METRICS", True)
    monkeypatch.setattr(sp.gr, "update", lambda **kwargs: {"update": kwargs})
    monkeypatch.setattr(sp, "is_cached", lambda url: False)
    monkeypatch.setattr(
        sp,
        "scrape_website",
        lambda url: {
            "success": True,
            "total_pages": 1,
            "pages": [{"title": "Home", "description": "", "sections": [], "content": "", "url": url, "page_type": "homepage"}],
            "errors": [],
        },
    )
    monkeypatch.setattr(sp, "format_scraped_content_for_context", lambda scraped_data: "content")
    monkeypatch.setattr(
        sp,
        "analyze_content_gaps",
        lambda scraped_content, url: SimpleNamespace(has_gaps=False, gaps_found=[], confidence_score=10, recommended_searches=[]),
    )
    monkeypatch.setattr(sp, "knowledge_to_chatbot_context", lambda knowledge: "ctx")
    monkeypatch.setattr(sp, "extract_name_from_text", lambda text, url: "Site")
    monkeypatch.setattr(sp, "create_knowledge_json", lambda url, scraped_data, web_search_results, raw_name: {})
    monkeypatch.setattr(sp, "save_knowledge_json", lambda knowledge, url: tmp_path / "stub.json")
    monkeypatch.setattr(sp, "build_status_new", lambda *args, **kwargs: "status")

    result = await sp.run_full_research_new("https://example.com", progress=DummyProgress())
    _, _, _, _, _, _, stats = result

    assert stats["cache_hit"] is False
    assert "tcr_seconds" in stats and stats["tcr_seconds"] >= 0


def test_log_chat_answer(tmp_path):
    log_file = tmp_path / "chat.jsonl"
    ml.log_chat_answer(
        question="Q?",
        answer="A!",
        provenance="primary_only",
        user="user@example.com",
        log_path=log_file,
    )

    data = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    record = json.loads(data[0])
    assert record["question"] == "Q?"
    assert record["answer"] == "A!"
    assert record["provenance"] == "primary_only"
    assert record["user"] == "user@example.com"


def test_log_job_metrics(tmp_path):
    log_file = tmp_path / "jobs.jsonl"

    ml.log_job_metrics(
        "https://example.com",
        {"cache_hit": True, "tcr_seconds": 1.5, "searches_run": 2, "pages_scraped": 3, "gaps_found": 1},
        user_id="user-1",
        log_path=log_file,
    )

    data = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    record = json.loads(data[0])
    assert record["url"] == "https://example.com"
    assert record["cache_hit"] is True
    assert record["tcr_seconds"] == 1.5
    assert record["searches_run"] == 2
    assert record["pages_scraped"] == 3
    assert record["gaps_found"] == 1
    assert record["user_id"] == "user-1"
