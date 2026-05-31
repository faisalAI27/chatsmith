import csv
import json

from scripts.run_chatbot_eval import (
    build_result_record,
    compute_heuristics,
    evaluate_questions,
    load_questions,
    write_results,
)
from scripts.summarize_chatbot_eval import summarize_results


def _question(
    question_id="q1",
    category="contact_support",
    question="What is their contact number?",
    requires_source=True,
    hallucination_sensitive=True,
):
    return {
        "id": question_id,
        "category": category,
        "intent": "test_intent",
        "question": question,
        "expected_behavior": "Expected behavior.",
        "requires_source": requires_source,
        "hallucination_sensitive": hallucination_sensitive,
        "manual_review_required": True,
    }


def test_question_file_loads_with_category_filter_and_limit(tmp_path):
    question_file = tmp_path / "questions.json"
    question_file.write_text(
        json.dumps(
            [
                _question("contact_1", "contact_support"),
                _question("social_1", "social_links", "Give me Instagram link."),
                _question("contact_2", "contact_support", "How can I contact them?"),
            ]
        ),
        encoding="utf-8",
    )

    questions = load_questions(question_file, category="contact_support", limit=1)

    assert len(questions) == 1
    assert questions[0]["id"] == "contact_1"


def test_standard_and_smoke_question_files_are_valid():
    standard = load_questions("evaluation/questions/customer_service_standard.json")
    smoke = load_questions("evaluation/questions/customer_service_smoke.json")

    assert len(smoke) == 10
    assert any(question["category"] == "social_links" for question in standard)
    assert any(question["category"] == "hallucination_resistance" for question in standard)
    assert all({"id", "category", "intent", "question", "expected_behavior"}.issubset(question) for question in smoke)


def test_heuristic_flags_detect_missing_source_url_phone_and_hallucination():
    social_flags = compute_heuristics(
        _question("s1", "social_links", "Give me Instagram link."),
        "Yes, they have Instagram.",
        {"mode": "rag", "sources": []},
    )
    contact_flags = compute_heuristics(
        _question("c1", "contact_support", "What is their contact number?"),
        "You can contact customer service.",
        {"mode": "rag", "sources": [{"source_url": "/contact"}]},
    )
    hallucination_flags = compute_heuristics(
        _question("h1", "hallucination_resistance", "Who is the CEO?", False, True),
        "The CEO is Jane Doe.",
        {"mode": "rag", "sources": []},
    )
    legacy_flags = compute_heuristics(
        _question("b1", "business_overview", "What is this website about?", True, False),
        "A shop.",
        {"mode": "legacy_prompt", "sources": []},
    )

    assert {"missing_sources", "missing_url"}.issubset(social_flags["flags"])
    assert "missing_phone" in contact_flags["flags"]
    assert "possible_hallucination" in hallucination_flags["flags"]
    assert "non_rag_mode" in legacy_flags["flags"]


def test_build_result_record_preserves_response_fields_and_debug():
    result = build_result_record(
        website_id="site123",
        question_record=_question("q1", "business_overview", "What is this website about?", True, False),
        response={
            "answer": "This website sells clothes.",
            "mode": "rag",
            "sources": [{"source_url": "https://example.com", "page_title": "Home"}],
            "warnings": ["weak source"],
            "retrieval_debug": {"query_intent": "general"},
        },
        latency_ms=123,
        include_debug=True,
    )

    assert result["website_id"] == "site123"
    assert result["answer"] == "This website sells clothes."
    assert result["mode"] == "rag"
    assert result["source_count"] == 1
    assert result["warnings"] == ["weak source"]
    assert result["retrieval_debug"] == {"query_intent": "general"}
    assert result["manual_score"] == ""
    assert result["issue_type"] == ""


def test_evaluate_questions_uses_mock_chat_client_and_records_errors():
    questions = [
        _question("q1", "business_overview", "What is this website about?", True, False),
        _question("q2", "contact_support", "What is their contact number?", True, True),
    ]

    def fake_chat_client(base_url, website_id, question, timeout):
        if "contact number" in question:
            raise RuntimeError("backend unavailable")
        return {
            "answer": "This website sells clothes.",
            "mode": "rag",
            "sources": [{"source_url": "https://example.com"}],
            "warnings": [],
        }

    results = evaluate_questions(
        website_id="site123",
        questions=questions,
        base_url="http://testserver",
        delay_seconds=0,
        chat_client=fake_chat_client,
    )

    assert len(results) == 2
    assert results[0]["answer"] == "This website sells clothes."
    assert results[1]["error"] == "backend unavailable"


def test_jsonl_csv_and_markdown_result_writing(tmp_path):
    results = [
        build_result_record(
            website_id="site123",
            question_record=_question("q1", "social_links", "Give me Instagram link.", True, True),
            response={
                "answer": "Instagram: https://instagram.com/example",
                "mode": "rag",
                "sources": [{"source_url": "https://example.com"}],
                "warnings": [],
            },
            latency_ms=50,
        )
    ]

    paths = write_results(results, tmp_path / "eval")

    assert paths["jsonl"].exists()
    assert paths["csv"].exists()
    assert paths["md"].exists()
    assert json.loads(paths["jsonl"].read_text(encoding="utf-8").splitlines()[0])["question_id"] == "q1"

    with paths["csv"].open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["question_id"] == "q1"
    assert "issue_type" in rows[0]
    assert "## social_links" in paths["md"].read_text(encoding="utf-8")


def test_summarizer_calculates_totals_flags_and_manual_scores(tmp_path):
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text(
        "\n".join(
            [
                "question_id,category,question,answer,mode,source_count,warnings,flags,latency_ms,manual_score,manual_notes,issue_type",
                "q1,contact_support,Phone?,Call us,rag,0,,missing_sources; missing_phone,100,0,,retrieval_wrong_chunk",
                "q2,social_links,Instagram?,URL,rag,1,,missing_url,200,2,,correct",
                "q3,hallucination_resistance,CEO?,Jane,rag,0,,possible_hallucination,300,N/A,,hallucination",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_results(csv_path)

    assert summary["total_questions"] == 3
    assert summary["answer_count"] == 3
    assert summary["average_latency_ms"] == 200
    assert summary["flags"]["missing_sources"] == 1
    assert summary["flags"]["missing_url"] == 1
    assert summary["flags"]["missing_phone"] == 1
    assert summary["flags"]["possible_hallucination"] == 1
    assert summary["manual_scores"]["counts"] == {"0": 1, "2": 1, "N/A": 1}
    assert summary["by_category"]["contact_support"]["total"] == 1
