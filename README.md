# ChatSMITH - Website to Chatbot Generator

An intelligent AI system that automatically generates chatbots from any website URL using smart web scraping, gap detection, and multi-agent orchestration.

## ✨ Features (current stack)

- **Smart Website Scraping** - Directly extracts content from websites (PRIMARY SOURCE)
- **Intelligent Gap Detection** - Only runs web searches when necessary
- **JSON Knowledge Caching** - Instant load for previously processed websites
- **Polite Scraping** - Respects robots.txt, rate limiting, retry logic
- **React UI + FastAPI** - Auth, progress, and chat

## 🏗️ Architecture

### Multi-Agent System

1. **Smart Website Scraper (PRIMARY SOURCE)**
   - Parallel page discovery and fetching
   - Respects robots.txt and rate limits
   - Retry logic with exponential backoff
   - Extracts and cleans HTML content

2. **Gap Detection Agent**
   - Analyzes extracted content completeness
   - Only triggers web search when confidence < 7/10
   - Recommends specific search queries

3. **Web Search Agent (SECONDARY SOURCE)**
   - Runs only when gaps are detected
   - Maximum 5 targeted searches (reduced from 15)
   - Results marked as secondary source

4. **Knowledge Storage System**
   - JSON files saved to `knowledge_files/`
   - URL-based caching (instant reload)
   - Source attribution (primary vs secondary)

5. **Chatbot Generator**
   - GPT-4o-mini powered responses
   - Priority: Homepage > Key pages > Blog > Web search
   - Context-aware answers

### Workflow

```
URL → Check Cache → [If cached: Load instantly]
                  → [If not cached:]
                     → Scrape Website (PRIMARY)
                     → Analyze Gaps
                     → Optional Web Search (SECONDARY)
                     → Save to JSON Cache
                     → Generate Chatbot
```

## 🚀 Quick Start (current stack)

### Backend (FastAPI)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=your_openai_api_key_here
export SUPABASE_URL=https://your-project-id.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
export CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

uvicorn backend.app.main:app --reload --port 8000
```

### Frontend (Vite React)
Requires Node ^20.19.0 or >=22.12.0 for Vite 7.

```bash
cd frontend
cat > .env <<'EOF'
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_BASE_URL=http://127.0.0.1:8000/api
EOF
npm install
npm run dev   # opens on http://localhost:5173
```

### Optional metrics (feature-flagged)
- Set `ENABLE_METRICS_LOGGING=true` in your environment to capture Time-to-Chatbot-Ready (TCR), cache hit flags, and chat Q/A JSONL logs (`metrics_logs/chat_answers.jsonl`). Disabled by default to avoid any impact on existing flows.

### Usage
- Sign up (first/last/email/password) → OTP → auto-login.
- Generate chatbot: paste URL, optional Force refresh → Run. A brief summary (pages scraped, web searches) shows, then the chatbot appears.
- Forgot password: email → OTP → new password (separate steps).

## 📁 Project Structure

```
backend/            # FastAPI app and pipeline copy
frontend/           # Vite React UI (auth, run, chat)
knowledge_files/    # Cached knowledge JSONs (used by pipeline)
requirements.txt    # Backend dependencies
README.md           # This file
```

## 🔒 Authentication (Supabase)

- Use OTP (not magic links) in Supabase email settings for signup and password reset.
- Backend uses `SUPABASE_SERVICE_ROLE_KEY`; frontend uses `SUPABASE_ANON_KEY`.
- Reset flow: email → OTP → new password.

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see IMPROVEMENT_PLAN.md for planned enhancements.
