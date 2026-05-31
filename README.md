# ChatSMITH - Website to Chatbot Generator

An intelligent AI system that automatically generates chatbots from any website URL using smart web scraping, gap detection, and multi-agent orchestration.

## ✨ Features (current stack)

- **Smart Website Scraping** - Directly extracts content from websites (PRIMARY SOURCE)
- **Browser Rendering Support** - Uses Playwright for public JavaScript-rendered websites when needed
- **Retrieval-Based Chat** - Uses `website_id` to retrieve indexed chunks and answer with sources
- **Intelligent Gap Detection** - Only runs web searches when necessary
- **JSON Knowledge Caching** - Instant load for previously processed websites
- **Polite Scraping** - Respects robots.txt, rate limiting, retry logic
- **React UI + FastAPI** - Progress and chat

## 🏗️ Architecture

### Multi-Agent System

1. **Smart Website Scraper (PRIMARY SOURCE)**
   - Parallel page discovery and fetching
   - Respects robots.txt and rate limits
   - Retry logic with exponential backoff
   - Extracts and cleans static or browser-rendered HTML content
   - Supports static, browser, and auto render modes

2. **Gap Detection Agent**
   - Analyzes extracted content completeness
   - Only triggers web search when confidence < 7/10
   - Recommends specific search queries

3. **Web Search Agent (SECONDARY SOURCE)**
   - Runs only when gaps are detected
   - Maximum 5 targeted searches (reduced from 15)
   - Results marked as secondary source

4. **Knowledge Storage System**
   - JSON files saved to `backend/knowledge_files/`
   - URL-based caching (instant reload)
   - Source attribution (primary vs secondary)

5. **Chunking + Vector Indexing**
   - Converts v2 knowledge JSON into RAG-ready chunks
   - Embeds chunks with OpenAI embeddings
   - Stores vectors in persistent local ChromaDB at `backend/vector_store/chroma/`
   - Retrieval is filtered by `website_id`

6. **RAG Chatbot Generator**
   - GPT-4o-mini powered responses
   - Retrieves relevant chunks by `website_id`
   - Builds a grounded prompt from retrieved website context
   - Returns answer sources separately
   - Falls back to the legacy generated system prompt if retrieval is unavailable

### Workflow

```
URL → Check Cache → [If cached: Load instantly]
                  → [If not cached:]
                     → Scrape Website (PRIMARY)
                     → Analyze Gaps
                     → Optional Web Search (SECONDARY)
                     → Save to JSON Cache
                     → Chunk + Vector Index (best effort)
                     → Generate Chatbot
                     → Chat: website_id → retrieve chunks → answer with sources
```

## 🚀 Quick Start (current stack)

### Backend (FastAPI)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

cp .env.example .env
# edit .env and set OPENAI_API_KEY

uvicorn backend.app.main:app --reload --port 8000
```

On Linux/CI, Playwright may need system browser dependencies:

```bash
python -m playwright install --with-deps chromium
```

Playwright/Chromium is required for JavaScript-rendered public websites. Static HTML websites can still be scraped through normal HTML extraction.

### Scraper render modes

- `SCRAPER_RENDER_MODE=auto` - default. Fetch static HTML first, then use Playwright only when static extraction looks weak.
- `SCRAPER_RENDER_MODE=browser` - use Playwright for every selected page, with static fallback if rendering fails.
- `SCRAPER_RENDER_MODE=static` - use only static HTML fetching; useful for tests/debugging.

Optional Playwright settings:

```dotenv
PLAYWRIGHT_TIMEOUT_MS=15000
PLAYWRIGHT_WAIT_MS=1000
PLAYWRIGHT_BLOCK_HEAVY_RESOURCES=true
```

Vector DB and embedding settings:

```dotenv
VECTOR_DB_PROVIDER=chroma
CHROMA_DB_DIR=backend/vector_store/chroma
CHROMA_COLLECTION_NAME=website_chunks

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BATCH_SIZE=64

RETRIEVAL_TOP_K=5
HYBRID_RETRIEVAL_ENABLED=true
HYBRID_LEXICAL_CANDIDATE_LIMIT=2000
HYBRID_VECTOR_CANDIDATE_MULTIPLIER=4
```

`backend/vector_store/` is generated runtime data and is ignored by Git. If vector indexing fails during development, website generation still completes through the legacy system-prompt fallback and exposes the indexing warning in job stats.

Limits: ChatSMITH is intended for public website content. It does not bypass login-protected pages, CAPTCHAs, private dashboards, paid content, or strong anti-bot protections.

### Frontend (Vite React)
Requires Node ^20.19.0 or >=22.12.0 for Vite 7.

```bash
cd frontend
cat > .env <<'EOF'
VITE_API_BASE_URL=http://127.0.0.1:8000/api
EOF
npm install
npm run dev   # opens on http://localhost:5173
```

### Optional metrics (feature-flagged)
- Set `ENABLE_METRICS_LOGGING=true` in your environment to capture Time-to-Chatbot-Ready (TCR), cache hit flags, and chat Q/A JSONL logs (`metrics_logs/chat_answers.jsonl`). Disabled by default to avoid any impact on existing flows.

### Manual vector indexing/retrieval check

Use this after a knowledge JSON exists in `backend/knowledge_files/`:

```bash
python - <<'PY'
from pathlib import Path
from backend.app.services.indexing_service import index_knowledge_file
from backend.app.services.retrieval_service import retrieve_relevant_chunks

path = Path("backend/knowledge_files/example.json")
summary = index_knowledge_file(path)
print(summary)

results = retrieve_relevant_chunks(
    summary["website_id"],
    "What services does this website offer?",
    top_k=5,
)
for result in results:
    print(result["chunk_type"], result["page_title"], result["source_url"])
    print(result["text"][:300])
PY
```

This command uses OpenAI embeddings, so `OPENAI_API_KEY` must be set in `.env`.

### RAG retrieval quality and hybrid reranking

ChatSMITH uses vector search for semantic similarity, then applies deterministic keyword/rule-based reranking before final chunks are sent to the chat prompt. This improves exact factual questions such as phone numbers, WhatsApp, email, addresses, store locations, timings, refund/return policies, and delivery questions.

The hybrid retrieval flow is:

1. fetch extra vector candidates,
2. fetch lexical candidates for the same `website_id`,
3. deduplicate candidates,
4. classify query intent,
5. boost answer-bearing chunks such as `section`, `paragraph_group`, `faq`, and `table`,
6. penalize weak `page_summary`, `image_context`, and unrelated product/footer chunks for contact/location/policy questions.

Reranking does not change stored chunks or embeddings. It only changes which chunks are returned for a query.

Manual retrieval ranking check:

```bash
python3 - <<'PY'
from backend.app.services.retrieval_service import retrieve_relevant_chunks

website_id = "PUT_WEBSITE_ID_HERE"
for q in [
    "contact number of Lama Retail",
    "Lama Retail phone number whatsapp customer service",
    "store locations of Lama Retail",
]:
    print("\nQUERY:", q)
    results = retrieve_relevant_chunks(website_id, q, top_k=5)
    for r in results:
        print(r.get("chunk_type"), r.get("page_title"), r.get("source_url"), r.get("rerank_score"))
        print(r.get("text", "")[:300])
PY
```

### RAG chat behavior

After `/api/jobs/run` completes, the job stats include `website_id`, `chunk_count`, and vector indexing status. The frontend sends `website_id` to `/api/chat`, and the backend:

1. embeds the latest user question,
2. retrieves top chunks filtered by `website_id`,
3. builds a grounded prompt from retrieved chunks,
4. calls the chat model,
5. returns the answer plus source previews.

If retrieval fails or no chunks are available, `/api/chat` uses the legacy `system_prompt` fallback when present. RAG answers are limited to scraped/indexed website context; the model should say when the retrieved context does not provide enough information.

### Manual RAG quality checks

After generating a chatbot for a site such as PakWheels, ask a mix of answerable and unanswerable questions:

- What services does this website offer?
- Does this website offer car inspection?
- How can I contact them?
- Who is the CEO of Tesla?

Expected behavior:

- website-related questions answer from retrieved chunks and show sources under the answer,
- unrelated questions say the website does not provide enough information,
- `/api/chat` returns `mode: "rag"` when `website_id` and vector index are available,
- response `warnings` and `retrieval_debug` explain weak or missing retrieval context without breaking the chat flow.

### Usage
- Generate chatbot: paste URL, optional Force refresh → Run. A brief summary (pages scraped, web searches) shows, then the chatbot appears.
- Ask questions in the chat panel after generation completes.

## 📁 Project Structure

```
backend/            # FastAPI app and pipeline copy
frontend/           # Vite React UI (run, chat)
backend/knowledge_files/ # Cached knowledge JSONs (used by backend pipeline)
backend/vector_store/    # Generated Chroma vector DB files (ignored)
requirements.txt    # Backend dependencies
README.md           # This file
```

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see IMPROVEMENT_PLAN.md for planned enhancements.
