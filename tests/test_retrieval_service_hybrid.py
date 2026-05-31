from backend.app.core.config import get_settings
from backend.app.services.retrieval_service import retrieve_relevant_chunks


class FakeEmbeddingService:
    model = "fake"

    def embed_query(self, query):
        self.query = query
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, vector_candidates, lexical_candidates):
        self.vector_candidates = vector_candidates
        self.lexical_candidates = lexical_candidates
        self.query_top_k = None
        self.lexical_limit = None
        self.chunk_types = None

    def query_similar(self, query_embedding, website_id, top_k=5, filters=None):
        self.query_top_k = top_k
        self.filters = filters
        return self.vector_candidates[:top_k]

    def get_website_chunks(self, website_id, limit=2000, chunk_types=None):
        self.lexical_limit = limit
        self.chunk_types = chunk_types
        return self.lexical_candidates[:limit]


def _chunk(chunk_id, text, chunk_type="section", score=None, page_title="", source_url=""):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "chunk_type": chunk_type,
        "score": score,
        "page_title": page_title,
        "source_url": source_url,
        "metadata": {"website_id": "lama"},
    }


def _enable_hybrid(monkeypatch):
    monkeypatch.setenv("HYBRID_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("HYBRID_VECTOR_CANDIDATE_MULTIPLIER", "4")
    monkeypatch.setenv("HYBRID_LEXICAL_CANDIDATE_LIMIT", "2000")
    get_settings.cache_clear()


def test_hybrid_retrieval_combines_vector_and_lexical_candidates(monkeypatch):
    _enable_hybrid(monkeypatch)
    weak_vector = [
        _chunk("summary", "Page title: Contact - Lama Retail", "page_summary", 0.99, "Contact", "/pages/contact"),
        _chunk("image", "Alt text: contact banner", "image_context", 0.8, "Contact", "/pages/contact"),
    ]
    lexical = [
        _chunk(
            "contact-detail",
            "CUSTOMER SERVICE Email | Whatsapp Contact us at 0311-1115262 09:00 AM to 09:00 PM Monday - Saturday",
            "section",
            None,
            "Contact",
            "/pages/contact",
        )
    ]
    store = FakeVectorStore(weak_vector, lexical)

    results, debug = retrieve_relevant_chunks(
        "lama",
        "contact number of Lama Retail",
        top_k=2,
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        include_debug=True,
    )

    assert results[0]["chunk_id"] == "contact-detail"
    assert debug["candidate_count_vector"] == 2
    assert debug["candidate_count_lexical"] == 1
    assert debug["candidate_count_combined"] == 3
    assert debug["query_intent"] == "contact"
    assert store.query_top_k == 20


def test_hybrid_retrieval_deduplicates_candidates_and_respects_top_k(monkeypatch):
    _enable_hybrid(monkeypatch)
    duplicate_vector = _chunk(
        "contact-detail",
        "Contact us at 0311-1115262 for WhatsApp support.",
        "section",
        0.4,
        "Contact",
        "/pages/contact",
    )
    duplicate_lexical = dict(duplicate_vector)
    weak_summary = _chunk("summary", "Page title: Contact", "page_summary", 0.9, "Contact", "/pages/contact")
    store = FakeVectorStore([weak_summary, duplicate_vector], [duplicate_lexical])

    results, debug = retrieve_relevant_chunks(
        "lama",
        "phone whatsapp customer service",
        top_k=1,
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        include_debug=True,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "contact-detail"
    assert debug["final_count"] == 1
    assert debug["candidate_count_combined"] == 2


def test_hybrid_retrieval_passes_chunk_type_filter_to_vector_and_lexical_store(monkeypatch):
    _enable_hybrid(monkeypatch)
    store = FakeVectorStore(
        [_chunk("faq", "Support answer", "faq", 0.5, "FAQ", "/faqs")],
        [_chunk("faq", "Support answer", "faq", None, "FAQ", "/faqs")],
    )

    results = retrieve_relevant_chunks(
        "lama",
        "support question",
        top_k=3,
        chunk_types=["faq"],
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
    )

    assert results
    assert store.filters == {"chunk_type": ["faq"]}
    assert store.chunk_types == ["faq"]
    get_settings.cache_clear()
