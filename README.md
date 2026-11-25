# ChatSMITH

ChatSMITH is a website-to-chatbot generator. It takes a public website URL, scrapes the site, stores a structured JSON knowledge file, builds RAG-ready chunks, indexes those chunks into a local Chroma vector database, and serves a FastAPI + React chatbot that answers from the scraped website context with sources.

The current MVP is focused on public commercial, shopping, and customer-service websites. It does not use Supabase. It does not scrape login-protected pages, private dashboards, paid content, CAPTCHA-protected pages, or websites with strong anti-bot protection.

## Current Capabilities

- FastAPI backend with synchronous development job endpoint.
- React/Vite frontend for URL submission, generation progress, and chat.
- **Site crawler**: follows links from every page plus sitemaps, within a configurable page/depth budget (default 100 pages, depth 5); whole-site, section (`/docs/`) or single-page scope.
- robots.txt per RFC 9309 (longest match, `Allow`, wildcards, crawl-delay), polite request pacing, private-network (SSRF) guard.
- Static fetching with `aiohttp` (charset-safe decoding, content-type and size checks, `Retry-After` handling).
- JavaScript rendering with one shared Playwright/Chromium browser per crawl: auto-scroll for lazy/infinite content, waits for client-side content, expands `<details>`, flattens shadow DOM, keeps partial DOM on slow pages.
- Duplicate control: same URL after redirects/index files, canonical URLs, identical content, near-duplicates (SimHash, e.g. product colour variants).
- Structured extraction for metadata, headings, sections, paragraphs, lists, FAQs (headings, accordions, `<details>`, JSON-LD FAQPage), tables, links, contacts (mailto/tel/WhatsApp), JSON-LD, OpenGraph/Twitter metadata, and image metadata.
- Cross-page boilerplate removal: header/footer/menu text repeated across pages is removed from each page and kept once in a `site_wide` record.
- Trafilatura cross-check per page (`quality.main_text_coverage`) that also restores main-text paragraphs the structural pass dropped.
- Linked PDF and Word (`.docx`) documents extracted as `document` pages (pdfplumber / python-docx).
- JSON knowledge storage in `backend/knowledge_files/`, plus JSONL, per-page Markdown and a crawl report under `backend/knowledge_files/exports/`.
- Stable `website_id` generation from normalized URLs.
- RAG chunking with chunk types such as `section`, `paragraph_group`, `faq`, `table`, `image_context`, `structured_data`, and `social_link`.
- OpenAI embeddings and persistent local ChromaDB indexing.
- Hybrid retrieval: vector search plus lexical candidates plus deterministic reranking.
- RAG chat API with answer sources, warnings, and legacy prompt fallback.
- Customer-service evaluation tooling with smoke, essential, and full question sets.
- Optional metrics logging behind `ENABLE_METRICS_LOGGING=false` by default.

## High-Level Flow

```text
Website URL
  -> normalize URL and check JSON cache
  -> read robots.txt + sitemaps, seed the crawl frontier
  -> crawl: fetch static HTML, render with Playwright when needed,
     extract each page, follow its links, skip duplicates
  -> remove cross-page boilerplate (kept once as a site_wide page)
  -> save v2 JSON knowledge file + JSONL/Markdown/crawl report exports
  -> build RAG chunks
  -> embed chunks with OpenAI
  -> upsert chunks into local ChromaDB
  -> frontend sends website_id with chat questions
  -> retrieve relevant chunks by website_id
  -> hybrid rerank chunks
  -> build grounded RAG prompt
  -> OpenAI chat response with sources
```

If vector indexing or retrieval is unavailable, the backend can still use the legacy generated `system_prompt` fallback when the frontend provides it. The evaluation runner intentionally sends `website_id` only, so it tests true indexed RAG behavior.

## Project Structure

```text
backend/app/
  api/
    chat.py                 # POST /api/chat/
    jobs.py                 # POST /api/jobs/run
    health.py               # GET /api/health/
  core/config.py            # .env-backed settings
  models/                   # Pydantic request/response models
  services/
    scrape_pipeline.py      # main scrape/cache/job pipeline
    crawler.py              # frontier crawler: scope, budget, robots, dedup, report
    page_fetcher.py         # HTTP fetch + render-mode selection per URL
    browser_renderer.py     # shared Playwright browser session and rendering
    static_extractor.py     # structured HTML extraction (block model)
    main_content.py         # Trafilatura cross-check of main text
    document_extractor.py   # PDF / DOCX extraction
    boilerplate.py          # cross-page template removal + site_wide page
    knowledge_export.py     # JSONL / Markdown / crawl report export
    robots.py               # robots.txt parser (RFC 9309)
    text_fingerprint.py     # content hash + SimHash for duplicate detection
    url_discovery.py        # sitemap/link discovery, URL filters and ranking
    scraper_schema.py       # v2 JSON shape helpers
    chunker.py              # JSON knowledge -> RAG chunks
    embedding_service.py    # OpenAI embedding abstraction
    vector_store.py         # ChromaDB persistence/retrieval
    indexing_service.py     # JSON -> chunks -> embeddings -> Chroma
    retrieval_service.py    # website_id-filtered retrieval
    retrieval_ranker.py     # hybrid reranking rules
    rag_prompt.py           # grounded prompt and source formatting
    rag_quality.py          # retrieval quality warnings/debug
    metrics_logger.py       # optional local JSONL metrics

frontend/
  src/App.jsx               # React UI, generation flow, chat UI
  src/styles.css            # UI styles
  package.json              # Vite React setup

evaluation/
  questions/                # smoke, essential, full question banks
  results/                  # generated eval outputs, ignored by Git
  README.md                 # evaluation workflow

  scrape_benchmark/         # extraction benchmark sites, golden facts, results

scripts/
  run_chatbot_eval.py       # run question sets against /api/chat/
  summarize_chatbot_eval.py # summarize CSV/JSONL eval outputs
  run_scrape_benchmark.py   # crawl benchmark sites and measure fact recall

tests/                      # backend/service/tooling tests
requirements.txt            # backend dependencies
.env.example                # backend environment template
```

Generated runtime data is ignored by Git:

- `.env`
- `backend/knowledge_files/*.json`
- `backend/vector_store/`
- `evaluation/results/*.json`, `*.jsonl`, `*.csv`, `*.md`
- `frontend/dist/`
- `frontend/node_modules/`
- `__pycache__/`, `.pytest_cache/`, `*.pyc`

## Requirements

Backend:

- Python 3.11+ recommended. Current local tests have been run on Python 3.13.
- OpenAI API key for normal generation, embeddings, and chat.
- Chromium installed through Playwright for browser-rendered scraping.

Frontend:

- Node `^20.19.0` or `>=22.12.0` for Vite 7.

## Environment Setup

Create `.env` from the tracked example:

```bash
cp .env.example .env
```

Then set at least:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Current `.env.example` settings:

```dotenv
OPENAI_API_KEY=

SCRAPER_RENDER_MODE=auto
PLAYWRIGHT_TIMEOUT_MS=15000
PLAYWRIGHT_WAIT_MS=1000
PLAYWRIGHT_BLOCK_HEAVY_RESOURCES=true

CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

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

Optional metrics:

```dotenv
ENABLE_METRICS_LOGGING=false
METRICS_LOG_PATH=metrics_logs/chat_answers.jsonl
METRICS_JOB_LOG_PATH=metrics_logs/job_runs.jsonl
```

Metrics are disabled by default and should remain optional.

## Backend Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

On Linux/CI, Playwright may need system dependencies:

```bash
python -m playwright install --with-deps chromium
```

Start the backend:

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/health/
curl http://127.0.0.1:8000/openapi.json
```

## Frontend Setup

From the repo root:

```bash
cd frontend
npm install
```

Optional `frontend/.env`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000/api
```

Run the frontend:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

Build check:

```bash
npm run build
```

The frontend has no Supabase requirement. It directly shows the ChatSMITH workflow, stores `website_id` after generation, and sends `website_id` plus chat messages to the backend. Assistant messages render a safe Markdown subset for links, bullets, numbered lists, and paragraphs without rendering raw HTML.

## API Endpoints

Root health:

```http
GET /health
```

API health:

```http
GET /api/health/
```

Run the website generation pipeline:

```http
POST /api/jobs/run
Content-Type: application/json

{
  "url": "https://example.com",
  "force_refresh": false,
  "max_pages": 50
}
```

`max_pages` is optional (default `CRAWLER_MAX_PAGES`).

Successful job stats include fields such as:

- `website_id`
- `pages_scraped`
- `searches_run`
- `chunk_count`
- `vector_indexed`
- `vector_indexing_summary`
- `vector_indexing_warning`
- `crawl_report`: pages stored, duplicates, failed/skipped URLs by reason, pages rendered, duration
- `knowledge_exports`: paths of the JSONL / Markdown / crawl report exports
- `system_prompt` for temporary legacy fallback

Chat:

```http
POST /api/chat/
Content-Type: application/json

{
  "website_id": "WEBSITE_ID",
  "question": "How can I contact them?",
  "messages": [
    {"role": "user", "content": "How can I contact them?"}
  ],
  "system_prompt": "optional legacy fallback prompt"
}
```

Response shape:

```json
{
  "message": {"role": "assistant", "content": "..."},
  "answer": "...",
  "sources": [
    {
      "source_url": "https://example.com/contact",
      "page_title": "Contact",
      "chunk_type": "section",
      "score": 0.82,
      "distance": 0.18,
      "text_preview": "..."
    }
  ],
  "mode": "rag",
  "warnings": [],
  "retrieval_debug": {},
  "metadata": {}
}
```

## Scraping System

The crawler is designed for public website content only.

### Crawl scope and budget

The submitted URL decides the scope:

| URL | Scope | What is crawled |
| --- | --- | --- |
| `https://site.com` | `site` | the whole site (same host; subdomains with `CRAWLER_INCLUDE_SUBDOMAINS=true`) |
| `https://site.com/docs/` (ends with `/`) | `section` | only URLs under `/docs/` |
| `https://site.com/blog/post` | `page` | that page plus up to `CRAWLER_SPECIFIC_PAGE_MAX_PAGES` related pages |

The frontier is a priority queue: about/contact/FAQ/pricing pages and shallow pages come first, so a budget-limited crawl still gets the most useful pages. Links are followed from every fetched page and from sitemaps (`robots.txt` `Sitemap:` lines, `/sitemap.xml`, sitemap indexes, `.xml.gz`).

Budget settings: `CRAWLER_MAX_PAGES` (100), `CRAWLER_MAX_DEPTH` (5), `CRAWLER_TIME_BUDGET_SECONDS` (900), `CRAWLER_CONCURRENCY` (4), `CRAWLER_POLITENESS_DELAY_MS` (250, raised to the site's `Crawl-delay`).

### What is skipped (and reported)

Every skipped or failed URL is counted in `crawl_report`:

- disallowed by `robots.txt` (set `CRAWLER_RESPECT_ROBOTS=false` only for sites you own)
- off-site, login/cart/account, search-result and tracking URLs
- duplicates: same URL after redirects or `/index.html`, `rel=canonical` copies, identical text, near-identical text (SimHash distance ≤ `CRAWLER_NEAR_DUPLICATE_DISTANCE`)
- private/local addresses (`CRAWLER_ALLOW_PRIVATE_HOSTS=false`)
- pages left with no page-specific text after boilerplate removal

### Render modes

- `SCRAPER_RENDER_MODE=auto`: default. Static fetch first; Playwright when the page's own text is weak, the page says JavaScript is required, or the static fetch is refused (e.g. bot protection). The richer of the two results is kept.
- `SCRAPER_RENDER_MODE=browser`: Playwright for every page, static fallback if rendering fails.
- `SCRAPER_RENDER_MODE=static`: static HTML only, useful for tests and debugging.

Rendering uses one Chromium per crawl (`PLAYWRIGHT_MAX_CONCURRENT_PAGES` tabs). Each render waits for the server response, then DOMContentLoaded, network idle and stable text; scrolls `PLAYWRIGHT_SCROLL_STEPS` times for lazy/infinite content; waits up to `PLAYWRIGHT_CONTENT_WAIT_MS` when the page is still an empty app shell; and keeps the DOM built so far if `PLAYWRIGHT_TIMEOUT_MS` is exceeded. Images are not downloaded (their `src`/`alt` metadata is still extracted).

### Extraction

Each page is read as ordered text blocks (headings, paragraphs, list items, table rows, and text inside `div`/`span` components), so it works for article markup and for card/component layouts. From the blocks the extractor builds:

- title, description, canonical URL, language, OpenGraph and Twitter metadata
- headings and sections (text grouped under the nearest h1-h4)
- paragraphs, lists, tables (empty cells keep column alignment), FAQs
- links (used for crawling), contacts from `mailto:`/`tel:`/WhatsApp links
- image metadata, JSON-LD structured data
- `navigation`: menu labels removed from the content (kept once in `site_wide`)
- quality: content length, `core_text_length`, `main_text_coverage` (Trafilatura cross-check), content hash and SimHash

Overlays, cookie bars and ads are removed by whole class/id tokens and ARIA dialog roles; `body`, `main`, `article` and large wrappers are never removed.

Linked `.pdf` and `.docx` files (up to `CRAWLER_MAX_DOCUMENTS`) become `document` pages. Scanned PDFs without a text layer are reported; OCR is not enabled.

Image handling stores metadata only. It does not download image binaries, run OCR, or generate image captions.

## JSON Knowledge Storage

Official runtime directory:

```text
backend/knowledge_files/
```

Knowledge files are generated runtime data and are ignored by Git.

The v2 JSON shape contains:

- `metadata.website_id`
- `metadata.url`
- `metadata.normalized_url`
- `metadata.scraping_version`
- top-level `pages`
- `primary_content.pages` for backward compatibility
- `secondary_content.searches` when web search supplements are used

Old JSON files are handled defensively by compatibility helpers.

Each crawl also writes, next to the knowledge JSON:

```text
backend/knowledge_files/exports/<knowledge-file-name>/
  pages.jsonl          one JSON object per page (page record + Markdown)
  pages/NNN-slug.md    one Markdown file per page, front matter with source_url
  crawl_report.json    full crawl report (skipped/failed URLs, timings, settings)
```

Disable with `KNOWLEDGE_EXPORT_ENABLED=false`.

## Chunking

`backend/app/services/chunker.py` converts knowledge JSON into RAG chunks.

Supported chunk types:

- `page_summary`
- `section`
- `paragraph_group`
- `faq`
- `table`
- `image_context`
- `structured_data`
- `social_link`

Each chunk preserves source metadata such as:

- `website_id`
- website URL
- source page URL
- normalized URL
- page title
- page type
- extraction method
- reliability
- chunk type and index

`social_link` chunks are created from page links and structured data such as `sameAs`, so questions about Instagram, Facebook, YouTube, TikTok, Twitter/X, LinkedIn, Pinterest, and WhatsApp can retrieve exact URLs.

## Vector Indexing

Indexing flow:

```text
knowledge JSON -> chunks -> OpenAI embeddings -> ChromaDB collection
```

Default Chroma path:

```text
backend/vector_store/chroma
```

Default collection:

```text
website_chunks
```

Indexing runs best-effort after `/api/jobs/run`. If indexing fails, the job can still return the old chatbot prompt and a `vector_indexing_warning`.

Manual reindex of an existing JSON file:

```bash
python3 - <<'PY'
from backend.app.services.indexing_service import index_knowledge_file

summary = index_knowledge_file(
    "backend/knowledge_files/example_com_c984d06aafbe.json",
    force_reindex=True,
)
print(summary)
PY
```

Check how many chunks a website has in Chroma:

```bash
python3 - <<'PY'
from backend.app.services.vector_store import ChromaVectorStore

store = ChromaVectorStore()
website_id = "PUT_WEBSITE_ID_HERE"
print(store.count_website_chunks(website_id))
print(store.get_collection_stats())
PY
```

If evaluation or chat says no indexed chunks were found, the JSON file may exist but the website has not been indexed into Chroma yet. Re-run `/api/jobs/run` or manually call `index_knowledge_file(...)`.

## Retrieval and RAG Chat

Retrieval flow:

```text
question -> OpenAI query embedding -> Chroma website_id filter -> vector candidates
         -> lexical website chunks -> dedupe -> deterministic rerank -> final chunks
```

Hybrid reranking improves exact factual questions for:

- contact numbers
- WhatsApp
- email
- customer support hours
- store locations
- social links
- policies
- delivery
- payments
- product details

RAG prompt behavior:

- answer only from retrieved website context
- use professional customer-service formatting
- use Markdown links when exact URLs answer the question
- group store locations by city when possible
- format contact answers like a contact card
- avoid invented details
- say the website does not provide enough information when context is missing

Manual retrieval check:

```bash
python3 - <<'PY'
from backend.app.services.retrieval_service import retrieve_relevant_chunks

website_id = "PUT_WEBSITE_ID_HERE"
for q in [
    "How can I contact them?",
    "What is their contact number?",
    "Give me their Instagram link.",
    "Where are their stores located?",
]:
    print("\\nQUERY:", q)
    results = retrieve_relevant_chunks(website_id, q, top_k=5)
    for r in results:
        print(r.get("chunk_type"), r.get("page_title"), r.get("source_url"), r.get("rerank_score"))
        print(r.get("text", "")[:300])
PY
```

## Evaluation Tooling

Evaluation files:

```text
evaluation/questions/customer_service_smoke.json      # 10 questions
evaluation/questions/customer_service_essential.json  # 39 questions
evaluation/questions/customer_service_standard.json   # 100 questions
evaluation/results/                                   # generated, ignored
```

Run smoke evaluation:

```bash
python3 scripts/run_chatbot_eval.py \
  --website-id WEBSITE_ID \
  --question-file evaluation/questions/customer_service_smoke.json \
  --base-url http://127.0.0.1:8000 \
  --output-prefix evaluation/results/site_smoke
```

Run essential evaluation:

```bash
python3 scripts/run_chatbot_eval.py \
  --website-id WEBSITE_ID \
  --question-file evaluation/questions/customer_service_essential.json \
  --base-url http://127.0.0.1:8000 \
  --output-prefix evaluation/results/site_essential
```

Run full evaluation:

```bash
python3 scripts/run_chatbot_eval.py \
  --website-id WEBSITE_ID \
  --question-file evaluation/questions/customer_service_standard.json \
  --base-url http://127.0.0.1:8000 \
  --output-prefix evaluation/results/site_full
```

Summarize results:

```bash
python3 scripts/summarize_chatbot_eval.py evaluation/results/site_essential.csv
```

Generated outputs:

- `.jsonl` full records
- `.csv` manual review sheet
- `.md` human-readable report

Automatic flags are review helpers, not final grading:

- `missing_sources`
- `missing_url`
- `missing_markdown_link`
- `missing_phone`
- `missing_email`
- `missing_hours`
- `missing_bullets`
- `missing_city_grouping`
- `possible_hallucination`
- `non_rag_mode`
- `very_short_answer`
- `raw_unformatted_long_answer`

Manual scoring:

- `2`: correct, complete, well-formatted, useful source
- `1`: partially correct or formatting/source issue
- `0`: wrong, hallucinated, or says not enough information when website has the data
- `N/A`: website likely does not contain this information

See `evaluation/README.md` and `evaluation/manual_review_template.md` for the full review workflow.

## Testing

Run backend tests (no internet, OpenAI key or browser needed; the crawler is tested against a local test site):

```bash
python3 -m pytest
```

GitHub Actions (`.github/workflows/ci.yml`) runs the backend tests and the frontend build on every push to `main` and every pull request.

### Scrape benchmark

`evaluation/scrape_benchmark/sites.json` lists public sites of different kinds (static, JavaScript-rendered, infinite scroll, delayed rendering, tables, WordPress, Shopify, documentation) with *golden facts* known to be on each site. The benchmark crawls each site and reports how many facts reached the extracted knowledge:

```bash
python3 scripts/run_scrape_benchmark.py --output-prefix current
python3 scripts/run_scrape_benchmark.py --only quotes_javascript,hockey_tables
python3 scripts/run_scrape_benchmark.py --repo ../other-checkout --output-prefix other
```

Results go to `evaluation/scrape_benchmark/results/<prefix>.md` and `.json`.

Run frontend build:

```bash
cd frontend
npm run build
```

If dependencies changed:

```bash
pip install -r requirements.txt
cd frontend && npm install
```

## Common Workflows

### Generate and Chat Through the UI

1. Start backend:

   ```bash
   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

2. Start frontend:

   ```bash
   cd frontend
   npm run dev
   ```

3. Open `http://localhost:5173`.
4. Paste a public website URL.
5. Click `Run`.
6. Wait for the chatbot panel.
7. Ask customer-service questions.

### Force Refresh a Website

Use the frontend Force refresh checkbox or send:

```json
{
  "url": "https://example.com",
  "force_refresh": true
}
```

to `/api/jobs/run`.

### Reindex Existing Knowledge Before Evaluation

If a knowledge JSON already exists but Chroma has no chunks for its `website_id`, reindex it:

```bash
python3 - <<'PY'
from backend.app.services.indexing_service import index_knowledge_file

summary = index_knowledge_file(
    "backend/knowledge_files/pakwheels_com_41641900b34f.json",
    force_reindex=True,
)
print(summary)
PY
```

Then run evaluation with that `website_id`.

## Troubleshooting

### Backend imports but OpenAI calls fail

Make sure `.env` exists and contains:

```dotenv
OPENAI_API_KEY=...
```

### Settings in `.env` win over shell variables

`scrape_pipeline.py` calls `load_dotenv(override=True)`, so a value in `.env` overrides the same variable exported in your shell. Change `.env` (or remove the line) to change a setting.

### Playwright rendering fails

Install Chromium:

```bash
python -m playwright install chromium
```

On Linux/CI:

```bash
python -m playwright install --with-deps chromium
```

For debugging, set:

```dotenv
SCRAPER_RENDER_MODE=static
```

### Evaluation returns "not enough information" for every answer

This usually means the selected `website_id` has no indexed vector chunks.

Check:

```bash
python3 - <<'PY'
from backend.app.services.vector_store import ChromaVectorStore

store = ChromaVectorStore()
print(store.count_website_chunks("PUT_WEBSITE_ID_HERE"))
PY
```

If the count is `0`, re-run generation or manually reindex the matching knowledge JSON.

### Frontend cannot reach backend

Check `frontend/.env`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000/api
```

Check backend CORS:

```dotenv
CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

### Node/Vite install fails

Use Node `^20.19.0` or `>=22.12.0`.

### Generated files appear in the working tree

The project ignores generated runtime files. Do not commit:

- `.env`
- `backend/knowledge_files/*.json`
- `backend/vector_store/`
- `evaluation/results/*`
- `frontend/dist/`
- `frontend/node_modules/`

## Current Limitations

- Public websites only.
- No login, CAPTCHA bypass, private dashboards, or paid content scraping.
- No OCR or AI image captioning; image handling stores metadata only; scanned PDFs yield no text.
- Content that appears only after clicks (e.g. "load more" buttons, tabs that fetch on click) is not triggered; scrolling-based loading is.
- Web-search gap filling (OpenAI Agents) is off by default (`ENABLE_WEB_SEARCH_SUPPLEMENT=false`) because it adds content that is not from the website.
- No external reranking API; reranking is deterministic and local.
- Chat still keeps a legacy `system_prompt` fallback during migration.
- `/api/jobs/run` is synchronous and intended for development, not production queueing.

## License

MIT License.
