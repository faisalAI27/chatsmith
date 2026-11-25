import asyncio
import os
import json
import hashlib
import inspect
import re
import ssl
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from urllib.parse import urlparse

import aiohttp
import certifi
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from agents import Agent, WebSearchTool, Runner
from agents.model_settings import ModelSettings
from ..core.config import get_settings
from .metrics_logger import log_chat_answer
from .browser_renderer import (
    get_scraper_render_mode,
    render_page_with_playwright,
    should_render_with_browser,
)
from .scraper_schema import (
    convert_v2_page_to_legacy_page,
    ensure_v2_knowledge_shape,
    make_website_id,
    normalize_url_for_cache,
)
from . import page_fetcher
from .crawler import CrawlConfig, crawl_website
from .knowledge_export import export_crawl_outputs
from .static_extractor import extract_static_page
from .url_discovery import (
    classify_page_type,
    classify_crawl_intent,
    discover_candidate_urls,
    discover_candidate_urls_with_stats,
    extract_homepage_links,
    fetch_sitemap_urls,
    normalize_discovered_url,
)

# Initialize
# Official backend runtime cache directory. A legacy root-level knowledge_files/
# directory may exist in old clones, but new runtime reads and writes stay here.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = BACKEND_ROOT / "knowledge_files"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(override=True)
ENABLE_METRICS = (os.getenv("ENABLE_METRICS_LOGGING", "false") or "").strip().lower() == "true"
_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Create the OpenAI client only when an OpenAI request is actually made."""
    global _openai_client
    if _openai_client:
        return _openai_client
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI requests.")
    _openai_client = OpenAI()
    return _openai_client


async def _maybe_await(value):
    """Support simple synchronous stubs in fast unit tests."""
    if inspect.isawaitable(value):
        return await value
    return value


def _index_knowledge_best_effort(
    knowledge: Dict,
    stats: Dict,
    force_reindex: bool,
) -> None:
    """Index chunks for future RAG without blocking the current chatbot flow."""
    try:
        from .indexing_service import index_knowledge

        summary = index_knowledge(knowledge, force_reindex=force_reindex)
        stats["website_id"] = summary.get("website_id", "")
        stats["chunk_count"] = summary.get("chunks_built", 0)
        stats["vector_indexed"] = bool(
            summary.get("chunks_indexed", 0) and not summary.get("errors")
        )
        stats["vector_indexing_summary"] = summary
        if summary.get("errors"):
            stats["vector_indexing_warning"] = "; ".join(summary["errors"])
            print(f"⚠️ Vector indexing skipped: {stats['vector_indexing_warning']}")
    except Exception as exc:
        stats["chunk_count"] = 0
        stats["vector_indexed"] = False
        stats["vector_indexing_summary"] = {
            "website_id": "",
            "chunks_built": 0,
            "chunks_indexed": 0,
            "chunks_deleted": 0,
            "by_type": {},
            "errors": [str(exc)],
        }
        stats["vector_indexing_warning"] = str(exc)
        print(f"⚠️ Vector indexing skipped: {exc}")

# Create SSL context with certifi certificates (matches notebook behavior)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

print("✅ Imports loaded")

# ============================================================
# WEBSITE CRAWLER + SCRAPER - PRIMARY SOURCE
# ============================================================
# Fetching lives in page_fetcher.py, crawling in crawler.py, extraction in
# static_extractor.py / document_extractor.py. The names below are kept as
# thin wrappers so existing callers and tests keep working.

# Legacy constant: the old scraper stopped at 10 pages. The crawler budget now
# comes from CRAWLER_MAX_PAGES (default 100) or the job's max_pages.
MAX_PAGES_TO_SCRAPE = 10
REQUEST_TIMEOUT = page_fetcher.REQUEST_TIMEOUT
MAX_RETRIES = page_fetcher.MAX_RETRIES
RETRY_DELAY = page_fetcher.RETRY_DELAY
USER_AGENT = page_fetcher.USER_AGENT
fetch_page_with_retry = page_fetcher.fetch_page_with_retry
_render_quality_extra = page_fetcher._render_quality_extra
_add_page_quality_error = page_fetcher._add_page_quality_error


async def fetch_page(session: aiohttp.ClientSession, url: str) -> Tuple[str, str]:
    """Fetch a single page and return (url, html_content) - wrapper for compatibility"""
    url, html, _ = await fetch_page_with_retry(session, url)
    return url, html


def clean_html_content(
    html: str,
    page_url: str = "",
    page_type: str = "other",
    status_code: int = 200,
    errors: List[str] | None = None,
    extraction_method: str = "static",
    quality_extra: Dict | None = None,
) -> Dict:
    """
    Clean HTML and extract meaningful content.
    Returns a v2 page record with legacy-compatible title, sections, and content.
    """
    return extract_static_page(
        html=html,
        page_url=page_url,
        page_type=page_type,
        status_code=status_code,
        errors=errors,
        extraction_method=extraction_method,
        quality_extra=quality_extra,
    )


async def fetch_static_html_with_fallback(
    session: aiohttp.ClientSession,
    page_url: str,
    allow_http_fallback: bool = False,
) -> Tuple[str, str, str]:
    """Fetch static HTML, optionally falling back from HTTPS to HTTP for the homepage."""
    effective_url, html, error, _ = await page_fetcher.fetch_static_html_with_fallback(
        session,
        page_url,
        allow_http_fallback=allow_http_fallback,
        fetch_fn=fetch_page_with_retry,
    )
    return effective_url, html, error


async def extract_page_with_render_mode(
    session: aiohttp.ClientSession,
    page_url: str,
    page_type: str = "other",
    allow_http_fallback: bool = False,
) -> Tuple[Dict | None, str, str, str]:
    """
    Extract one page according to SCRAPER_RENDER_MODE.

    Returns (page_record, html_used_for_discovery, effective_url, error_message).
    The fetch/render functions are looked up on this module at call time so
    unit tests can monkeypatch them without a network or browser.
    """
    return await page_fetcher.extract_page_with_render_mode(
        session,
        page_url,
        page_type=page_type,
        allow_http_fallback=allow_http_fallback,
        fetch_fn=fetch_page_with_retry,
        render_fn=render_page_with_playwright,
    )


def discover_key_pages(html: str, base_url: str) -> List[str]:
    """
    Discover important internal pages from the homepage.
    Returns a list of URLs to scrape.
    """
    return discover_candidate_urls(
        homepage_html=html,
        base_url=base_url,
        max_pages=MAX_PAGES_TO_SCRAPE - 1,
    )


async def _crawler_extract(session, page_url: str, page_type: str = "other", allow_http_fallback: bool = False):
    return await extract_page_with_render_mode(
        session,
        page_url,
        page_type=page_type,
        allow_http_fallback=allow_http_fallback,
    )


async def scrape_website(url: str, max_pages: int | None = None, max_depth: int | None = None) -> Dict:
    """
    Crawl a website and extract every page within the crawl budget.

    Follows links from every fetched page and from sitemaps, respects
    robots.txt, renders JavaScript pages with one shared browser when needed,
    skips duplicates, and removes cross-page boilerplate.
    Returns {source_url, scraped_at, pages, total_pages, success, errors, crawl_report}.
    """
    config = CrawlConfig.from_settings(max_pages=max_pages, max_depth=max_depth)
    return await crawl_website(url, extract_fn=_crawler_extract, config=config)


def format_scraped_content_for_context(scraped_data: Dict) -> str:
    """Convert scraped data into a formatted string for the chatbot context."""
    if not scraped_data.get("success"):
        return ""
    
    parts = []
    parts.append(f"=== WEBSITE CONTENT (Primary Source) ===")
    parts.append(f"Source: {scraped_data['source_url']}")
    parts.append(f"Pages scraped: {scraped_data['total_pages']}")
    parts.append("")
    
    for page in scraped_data.get("pages", []):
        if page.get("title"):
            parts.append(f"## {page['title']}")
        page_url = page.get("page_url") or page.get("url")
        if page_url:
            parts.append(f"URL: {page_url}")
        if page.get("description"):
            parts.append(f"Description: {page['description']}")
        
        # Add sections
        for section in page.get("sections", [])[:5]:  # Limit sections per page
            if section.get("heading"):
                parts.append(f"\n### {section['heading']}")
            if section.get("content"):
                parts.append(section['content'][:500])
        
        # Add main content if no sections
        if not page.get("sections") and page.get("content"):
            parts.append(page['content'][:800])
        
        parts.append("\n---\n")
    
    return "\n".join(parts)


print("✅ Smart Scraper loaded (Phase 3: retry, robots.txt, rate limiting)")

# Search Agent Configuration
SEARCH_INSTRUCTIONS = "You are a research assistant. Given a search URL, you search the web for that URL and \
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 \
words. Capture the main points. Write succintly, no need to have complete sentences or good \
grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the \
essence and ignore any fluff. Do not include any additional commentary other than the summary itself."

search_agent = Agent(
    name="Search agent",
    instructions=SEARCH_INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="low")],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="required"),
)

# Planner Agent Configuration - REDUCED FOR GAP FILLING ONLY
HOW_MANY_SEARCHES = 5  # Reduced from 15 - only for filling gaps

PLANNER_INSTRUCTIONS = f"""You are a helpful research assistant. You will be given:
1. A URL
2. Content already extracted from that website (PRIMARY SOURCE)

Your job is to identify ONLY the gaps - information that is MISSING from the extracted content.
Come up with {HOW_MANY_SEARCHES} targeted web searches to fill these specific gaps.

DO NOT search for information that is already present in the extracted content.
Focus on: missing contact details, pricing not found, team info gaps, specific features unclear, etc.

If the extracted content is comprehensive, you can suggest fewer searches or very specific ones."""


class WebSearchItem(BaseModel):
    reason: str = Field(description="The specific gap this search will fill.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    has_significant_gaps: bool = Field(description="True if there are significant gaps that need web search.")
    searches: list[WebSearchItem] = Field(description="A list of web searches to fill the gaps.")


planner_agent = Agent(
    name="PlannerAgent",
    instructions=PLANNER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=WebSearchPlan,
)

# ============================================================
# GAP DETECTION AGENT
# ============================================================

GAP_DETECTION_INSTRUCTIONS = """You are a content analysis expert. You analyze extracted website content and determine if web searches are needed to fill gaps.

Analyze the provided website content and determine:
1. Is the content comprehensive enough for a chatbot to answer questions about this website?
2. What specific information gaps exist (if any)?
3. Should we run web searches to fill these gaps?

Be conservative - only recommend web searches if there are CLEAR gaps like:
- No contact information found
- Pricing/plans mentioned but not detailed
- Services listed but not explained
- Team/leadership mentioned but not detailed
- Key product features missing

If the website content covers the basics (what they do, who they are, how to contact), NO web search is needed."""


class GapAnalysis(BaseModel):
    has_gaps: bool = Field(description="True if significant information gaps exist")
    confidence_score: int = Field(description="1-10 score of how complete the extracted content is")
    gaps_found: list[str] = Field(description="List of specific gaps identified")
    recommended_searches: list[str] = Field(description="Specific search queries to fill gaps (max 5)")
    reasoning: str = Field(description="Brief explanation of the analysis")


gap_detection_agent = Agent(
    name="GapDetectionAgent",
    instructions=GAP_DETECTION_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=GapAnalysis,
)


async def analyze_content_gaps(scraped_content: str, url: str) -> GapAnalysis:
    """Analyze scraped content to determine if web searches are needed."""
    print("🔍 Analyzing content for gaps...")
    
    prompt = f"""Analyze this extracted website content and determine if web searches are needed to fill gaps.

URL: {url}

EXTRACTED CONTENT:
{scraped_content[:6000]}

Remember: Only recommend searches for CLEAR gaps. If basic info is present, return has_gaps=False."""
    
    result = await Runner.run(gap_detection_agent, prompt)
    
    analysis = result.final_output
    print(f"  📊 Confidence: {analysis.confidence_score}/10")
    print(f"  🔎 Has gaps: {analysis.has_gaps}")
    if analysis.gaps_found:
        print(f"  📋 Gaps: {', '.join(analysis.gaps_found[:3])}")
    
    return analysis


print("✅ Gap Detection Agent loaded")

# Writer Agent Configuration
WRITER_INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query About website URL. "
    "You will be provided with the original URL, and some initial research done by a research assistant.\n"
    "You should first come up with an outline for the Detail report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")


writer_agent = Agent(
    name="WriterAgent",
    instructions=WRITER_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=ReportData,
)

# Name Extractor Agent Configuration
NAME_AGENT_INSTRUCTIONS = (
    "You analyze the provided text and extract a single concise name "
    "that best represents the main subject (e.g., site/company/person/product). "
    "If a URL is provided, prefer the name associated with that URL. "
    "Return only the name, no extra words."
)


class NameExtraction(BaseModel):
    name: str = Field(description="The extracted name from the text")


name_extractor = Agent(
    name="NameExtractor",
    instructions=NAME_AGENT_INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=NameExtraction,
)

# Core Research Functions (Updated for new workflow)

async def plan_gap_searches(url: str, scraped_content: str):
    """Use the planner_agent to plan targeted searches based on gaps in scraped content."""
    print("Planning targeted searches for gaps...")
    prompt = f"""URL: {url}

ALREADY EXTRACTED CONTENT (PRIMARY SOURCE):
{scraped_content[:4000]}

Based on the above content, identify gaps and suggest {HOW_MANY_SEARCHES} specific searches to fill them.
If content is comprehensive, suggest fewer searches."""
    
    result = await Runner.run(planner_agent, prompt)
    print(f"Will perform {len(result.final_output.searches)} gap-filling searches")
    return result.final_output


async def search(item: WebSearchItem):
    """Use the search agent to run a web search for each item in the search plan"""
    input = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input)
    return result.final_output


async def perform_searches(search_plan: WebSearchPlan):
    """Call search() for each item in the search plan"""
    if not search_plan.searches:
        print("No searches needed")
        return []
    print(f"Searching ({len(search_plan.searches)} queries)...")
    tasks = [asyncio.create_task(search(item)) for item in search_plan.searches]
    results = await asyncio.gather(*tasks)
    print("Finished searching")
    return results


async def extract_name_from_text(text: str, url: str = "") -> str:
    """Extract the name from the text content"""
    prompt = (
        f"Text to analyze:\n{text[:2000]}\n\n"
        f"Original URL: {url}\n\n"
        "Return only the best fitting name for this website/company/organization."
    )
    result = await Runner.run(name_extractor, prompt)
    return (result.final_output.name or "").strip()

# ============================================================
# JSON KNOWLEDGE BASE - Storage & Caching
# ============================================================

def get_website_id(url: str) -> str:
    """Return the stable website id used by JSON cache files and future stores."""
    normalized_url = normalize_url_for_cache(url)
    return make_website_id(normalized_url)


def _cache_domain_slug(normalized_url: str) -> str:
    """Build a readable filename prefix from the normalized URL host."""
    domain = urlparse(normalized_url).netloc.replace("www.", "")
    slug = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
    return slug or "website"


def _legacy_cache_path(url: str) -> Path:
    """Return the old exact-URL cache filename in the official cache directory."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    domain = urlparse(url).netloc.replace("www.", "").replace(".", "_")
    return KNOWLEDGE_DIR / f"{domain}_{url_hash}.json"


def _cache_candidate_paths(url: str) -> List[Path]:
    """Return official-directory cache paths, newest normalized name first."""
    primary = Path(get_cache_path(url))
    legacy = _legacy_cache_path(url)
    if legacy != primary:
        return [primary, legacy]
    return [primary]


def get_cache_path(url: str) -> str:
    """Get the cache file path for a given URL."""
    normalized_url = normalize_url_for_cache(url)
    website_id = get_website_id(normalized_url)
    domain = _cache_domain_slug(normalized_url)
    return str(KNOWLEDGE_DIR / f"{domain}_{website_id}.json")


def is_cached(url: str) -> bool:
    """Check if knowledge for a URL is already cached."""
    return any(cache_path.exists() for cache_path in _cache_candidate_paths(url))


def get_cached_knowledge(url: str) -> Dict | None:
    """Load cached knowledge if available. Returns None if not cached."""
    for cache_path in _cache_candidate_paths(url):
        if cache_path.exists():
            try:
                knowledge = load_knowledge_json(str(cache_path), fallback_url=url)
                print(f"📂 Loaded from cache: {cache_path}")
                return knowledge
            except Exception as e:
                print(f"⚠️ Cache read error ({cache_path}): {e}")
                continue
    return None


def ensure_knowledge_metadata(knowledge: Dict, fallback_url: str = "") -> Dict:
    """Backfill metadata fields needed by the normalized JSON cache format."""
    return ensure_v2_knowledge_shape(knowledge, fallback_url=fallback_url)


def create_knowledge_json(url: str, scraped_data: Dict, web_search_results: List = None, name: str = "") -> Dict:
    """Create a structured JSON knowledge base from all sources."""
    normalized_url = normalize_url_for_cache(url)
    v2_pages = [
        {key: value for key, value in page.items() if not str(key).startswith("_")}
        for page in scraped_data.get("pages", [])
        if isinstance(page, dict)
    ]
    legacy_pages = [convert_v2_page_to_legacy_page(page) for page in v2_pages]
    knowledge = {
        "metadata": {
            "website_id": make_website_id(normalized_url),
            "url": url,
            "normalized_url": normalized_url,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "scraping_version": "v2",
            "pages_scraped": scraped_data.get("total_pages", 0),
            "has_web_search_supplement": bool(web_search_results),
        },
        "primary_content": {
            "source": "website_scraping",
            "reliability": "high",
            "pages": legacy_pages
        },
        "pages": v2_pages,
        "secondary_content": {
            "source": "web_search",
            "reliability": "medium",
            "searches": []
        }
    }
    
    crawl_report = scraped_data.get("crawl_report")
    if isinstance(crawl_report, dict):
        knowledge["metadata"]["crawler_version"] = CRAWLER_VERSION
        knowledge["metadata"]["crawl"] = summarize_crawl_report(crawl_report)

    # Add web search results if available
    if web_search_results:
        for i, result in enumerate(web_search_results):
            knowledge["secondary_content"]["searches"].append({
                "index": i + 1,
                "result": str(result)[:1000]
            })
    
    return ensure_v2_knowledge_shape(knowledge, fallback_url=url)


CRAWLER_VERSION = "frontier-1"
_TITLE_SEPARATORS = re.compile(r"\s+[|\-–—:·»]\s+")
_SITE_NAME_TYPES = {"organization", "website", "educationalorganization", "collegeoruniversity", "localbusiness",
                    "store", "onlinestore", "corporation", "governmentorganization", "ngo", "school"}


def _setting(name: str, default):
    try:
        return getattr(get_settings(), name, default)
    except Exception:
        return default


def web_search_supplement_enabled() -> bool:
    """Web-search gap filling adds non-website content; the project scope
    lists it as out of scope, so it is opt-in (ENABLE_WEB_SEARCH_SUPPLEMENT)."""
    return bool(_setting("enable_web_search_supplement", False))


def summarize_crawl_report(report: Dict) -> Dict:
    """Small crawl summary stored in knowledge metadata (full report goes to exports)."""
    skipped = report.get("skipped") or {}
    return {
        "scope": report.get("scope"),
        "pages_stored": report.get("pages_stored", 0),
        "documents_stored": report.get("documents_stored", 0),
        "pages_attempted": report.get("pages_attempted", 0),
        "duplicates": report.get("duplicate_count", 0),
        "failed": len(report.get("failed") or []),
        "skipped": dict(skipped),
        "sitemap_urls": report.get("sitemap_urls", 0),
        "duration_seconds": report.get("duration_seconds"),
        "pages_rendered": (report.get("render") or {}).get("pages_rendered", 0),
        "boilerplate_blocks_removed": (report.get("boilerplate") or {}).get("blocks_removed", 0),
    }


def derive_site_name(scraped_data: Dict) -> str:
    """Site name from page metadata (no LLM call): og:site_name, JSON-LD, titles."""
    pages = [page for page in scraped_data.get("pages", []) if isinstance(page, dict) and page.get("page_type") != "site_wide"]
    if not pages:
        return ""
    homepage = next((page for page in pages if page.get("page_type") == "homepage"), pages[0])
    open_graph = (homepage.get("metadata") or {}).get("open_graph") or {}
    if open_graph.get("site_name"):
        return str(open_graph["site_name"]).strip()

    stack = list(homepage.get("structured_data") or [])
    while stack:
        node = stack.pop(0)
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            types = node.get("@type")
            types = types if isinstance(types, list) else [types]
            if any(str(t).lower() in _SITE_NAME_TYPES for t in types if t) and isinstance(node.get("name"), str) and node["name"].strip():
                return node["name"].strip()
            stack.extend(value for value in node.values() if isinstance(value, (dict, list)))

    segment_counts: Dict[str, int] = {}
    for page in pages:
        for segment in {seg.strip() for seg in _TITLE_SEPARATORS.split(page.get("title") or "") if seg.strip()}:
            segment_counts[segment] = segment_counts.get(segment, 0) + 1
    repeated = [segment for segment, count in segment_counts.items() if count >= 2 and len(segment) <= 80]
    if len(pages) >= 2 and repeated:
        return max(repeated, key=lambda segment: (segment_counts[segment], -len(segment)))
    title_parts = [seg.strip() for seg in _TITLE_SEPARATORS.split(homepage.get("title") or "") if seg.strip()]
    if title_parts and title_parts[0].lower() not in {"home", "homepage", "welcome", "index"}:
        return title_parts[0][:80]
    return ""


def _export_knowledge_best_effort(scraped_data: Dict, knowledge: Dict, knowledge_filepath, stats: Dict) -> None:
    if not _setting("knowledge_export_enabled", True) or not knowledge_filepath:
        return
    try:
        stats["knowledge_exports"] = export_crawl_outputs(
            scraped_data.get("pages", []),
            knowledge,
            knowledge_filepath,
            crawl_report=scraped_data.get("crawl_report"),
        )
    except Exception as exc:
        print(f"⚠️ Knowledge export skipped: {exc}")


def _knowledge_has_pages(knowledge: Dict | None) -> bool:
    if not isinstance(knowledge, dict):
        return False
    metadata = knowledge.get("metadata") or {}
    if int(metadata.get("pages_scraped") or 0) > 0:
        return True
    primary = knowledge.get("primary_content") or {}
    return bool(knowledge.get("pages") or primary.get("pages"))


def save_knowledge_json(knowledge: Dict, url: str) -> str:
    """Save knowledge JSON to file. Returns filepath."""
    knowledge = ensure_knowledge_metadata(knowledge, fallback_url=url)
    filepath = Path(get_cache_path(url))
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(f"Could not save knowledge JSON to {filepath}: {exc}") from exc
    
    print(f"💾 Knowledge saved to: {filepath}")
    return str(filepath)


def load_knowledge_json(filepath: str, fallback_url: str = "") -> Dict:
    """Load knowledge from a JSON file."""
    path = Path(filepath)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid knowledge JSON at {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Could not read knowledge JSON at {path}: {exc}") from exc
    return ensure_knowledge_metadata(knowledge, fallback_url=fallback_url)


def knowledge_to_chatbot_context(knowledge: Dict) -> str:
    """
    Convert JSON knowledge to a formatted string for chatbot context.
    IMPROVED: Prioritizes homepage/about content for better answers.
    """
    parts = []
    
    # Metadata
    meta = knowledge.get("metadata", {})
    parts.append(f"=== WEBSITE INFORMATION ===")
    parts.append(f"Name: {meta.get('name', 'Unknown')}")
    parts.append(f"URL: {meta.get('url', '')}")
    parts.append(f"Pages analyzed: {meta.get('pages_scraped', 0)}")
    parts.append("")
    
    # Primary content (website scraping) - PRIORITIZE HOMEPAGE AND KEY PAGES
    primary = knowledge.get("primary_content", {})
    pages = primary.get("pages", [])
    
    # Separate pages by priority
    homepage = None
    key_pages = []  # about, contact, services, books, etc.
    blog_pages = []  # blog posts (lower priority for context)
    
    key_page_keywords = ['about', 'contact', 'services', 'products', 'team', 
                         'pricing', 'faq', 'books', 'publications', 'cv', 'resume']
    
    for page in pages:
        page_type = page.get("page_type", "")
        url_lower = page.get("url", "").lower()
        
        if page_type == "homepage":
            homepage = page
        elif any(kw in url_lower for kw in key_page_keywords):
            key_pages.append(page)
        elif 'blog' in url_lower or '/20' in url_lower:  # blog posts often have dates
            blog_pages.append(page)
        else:
            key_pages.append(page)  # Default to key pages
    
    parts.append("=== PRIMARY SOURCE (Website Content) ===")
    parts.append("[This is the most reliable information - directly from the website]")
    parts.append("")
    
    # 1. HOMEPAGE FIRST (most important - give it full space)
    if homepage:
        parts.append("## HOMEPAGE (Main Information)")
        if homepage.get("title"):
            parts.append(f"Title: {homepage['title']}")
        if homepage.get("description"):
            parts.append(f"Description: {homepage['description']}")
        
        # Include ALL sections from homepage (this is where key bio info is)
        for section in homepage.get("sections", []):
            if section.get("heading"):
                parts.append(f"\n### {section['heading']}")
            if section.get("content"):
                parts.append(section['content'][:800])  # More space for homepage
        
        # Also include main content
        if homepage.get("content"):
            parts.append(f"\nMain content: {homepage['content'][:1500]}")
        
        parts.append("\n---\n")
    
    # 2. KEY PAGES (about, contact, books, etc.)
    for page in key_pages[:5]:  # Limit to 5 key pages
        if page.get("title"):
            parts.append(f"## {page['title']}")
        if page.get("description"):
            parts.append(f"Description: {page['description']}")
        
        for section in page.get("sections", [])[:4]:
            if section.get("heading"):
                parts.append(f"\n### {section['heading']}")
            if section.get("content"):
                parts.append(section['content'][:400])
        
        if not page.get("sections") and page.get("content"):
            parts.append(page['content'][:600])
        
        parts.append("\n---\n")
    
    # 3. BLOG PAGES (summaries only - less important for chatbot context)
    if blog_pages:
        parts.append("\n## BLOG/ARTICLES (Recent posts)")
        for page in blog_pages[:3]:  # Only top 3 blog posts
            title = page.get("title", "")
            desc = page.get("description", "")
            if title:
                parts.append(f"- {title}")
            if desc:
                parts.append(f"  {desc[:200]}")
        parts.append("\n---\n")
    
    # Secondary content (web search)
    secondary = knowledge.get("secondary_content", {})
    if secondary.get("searches"):
        parts.append("\n=== SECONDARY SOURCE (Web Search Supplement) ===")
        parts.append("[Use this only if primary source doesn't have the answer]")
        parts.append("")
        
        for search in secondary.get("searches", [])[:5]:
            parts.append(f"Search result {search.get('index', '')}:")
            parts.append(search.get('result', '')[:500])
            parts.append("")
    
    return "\n".join(parts)


print("✅ JSON Knowledge Base functions loaded (with caching)")

# ============================================================
# UI CONSTANTS
# ============================================================

CHAT_PLACEHOLDER = "Ask anything about the website..."
DEFAULT_STATUS_TEXT = "➡️ Enter a URL and click **Generate Chatbot** to start."
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
:root {
  --brand-1: #7c3aed;
  --brand-2: #22d3ee;
  --panel: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.12);
}
body, * {
  font-family: 'Space Grotesk', 'Segoe UI', sans-serif !important;
}
.app-shell {
  background: radial-gradient(120% 120% at 20% 20%, rgba(124,58,237,0.12), transparent 50%),
              radial-gradient(120% 120% at 80% 0%, rgba(34,211,238,0.14), transparent 45%),
              #0f1115;
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 12px 40px rgba(0,0,0,0.25);
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
}
.primary-btn button {
  background: linear-gradient(120deg, var(--brand-1), var(--brand-2));
  color: #0f1115 !important;
  border: none;
  box-shadow: 0 10px 30px rgba(124,58,237,0.35);
}
.secondary-btn button {
  border: 1px solid var(--border);
}
.input-wide input {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
}
.badge {
  color: #a5f3fc;
}
"""

# ============================================================
# UI HELPER FUNCTIONS (Updated for new workflow + Phase 3 Error Handling)
# ============================================================

def build_status_new(percent: float, current_step: int, selected_name: str | None = None, 
                     finished: bool = False, stats: Dict = None, from_cache: bool = False,
                     errors: List[str] = None) -> str:
    """
    Build status text with percentage, steps, and progress bar
    Updated steps for the new scraper-first workflow
    """
    steps = [
        "Scraping website (PRIMARY SOURCE)",
        "Analyzing content gaps",
        "Running targeted searches (if needed)",
        "Building knowledge base",
        "Extracting name & preparing chatbot",
    ]

    # Progress bar line
    bar_len = 24
    filled = int(bar_len * percent / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    # Step list with icons
    lines = []
    for i, label in enumerate(steps):
        if finished or i < current_step:
            icon = "✅"
        elif i == current_step:
            icon = "🔄"
        else:
            icon = "⏳"
        lines.append(f"- {icon} Step {i+1}: {label}")

    text = f"### Progress: {percent:.0f}%\n\n`{bar}`\n\n" + "\n".join(lines)

    # Add stats if available
    if stats:
        text += f"\n\n📊 **Stats:**"
        if "pages_scraped" in stats:
            text += f"\n- Pages scraped: {stats['pages_scraped']}"
        if "searches_run" in stats:
            text += f"\n- Web searches: {stats['searches_run']}"
        if "gaps_found" in stats:
            text += f"\n- Gaps filled: {stats['gaps_found']}"

    # Show errors if any (Phase 3 enhancement)
    if errors and len(errors) > 0:
        text += f"\n\n⚠️ **Warnings ({len(errors)}):**"
        for err in errors[:3]:  # Show max 3 errors
            text += f"\n- {err[:60]}..."

    if finished:
        if from_cache:
            text += f"\n\n⚡ **Loaded from cache** (instant!)"
        if selected_name:
            text += f"\n\n**Selected name:** `{selected_name}`"
        text += "\n\n🤖 Chatbot is ready. Ask your questions below."

    return text


def build_error_status(error_type: str, details: str = "") -> str:
    """Build a user-friendly error status message."""
    error_messages = {
        "invalid_url": "❌ **Invalid URL**\n\nPlease enter a valid website URL (e.g., https://example.com)",
        "connection_failed": f"❌ **Connection Failed**\n\nCouldn't connect to the website. Please check:\n- The URL is correct\n- The website is online\n- Your internet connection\n\n{details}",
        "scrape_failed": f"❌ **Scraping Failed**\n\nCouldn't extract content from this website.\n\nPossible reasons:\n- Website blocks automated access\n- Browser rendering failed or is not installed\n- Login, CAPTCHA, or private content restrictions\n- robots.txt restrictions\n\n{details}",
        "api_error": f"❌ **API Error**\n\nAn error occurred while processing.\n\n{details}\n\nPlease try again.",
        "timeout": "❌ **Timeout**\n\nThe request took too long. The website might be slow or unresponsive.\n\nTry again or use a different URL.",
    }
    return error_messages.get(error_type, f"❌ **Error**\n\n{details}")


# ============================================================
# NEW MAIN RESEARCH PIPELINE (Scraper-First Approach with Caching + Error Handling)
# ============================================================

async def run_full_research_new(
    url: str,
    force_refresh: bool = False,
    progress=gr.Progress(),
    max_pages: int | None = None,
):
    """
    NEW workflow: Scrape first, then fill gaps with targeted searches.
    With caching support and improved error handling (Phase 3).
    """
    stats = {"pages_scraped": 0, "searches_run": 0, "gaps_found": 0, "cache_hit": False}
    start_time = time.time() if ENABLE_METRICS else None
    errors = []  # Track errors for UI feedback
    
    # ===== Check Cache First =====
    if not force_refresh and is_cached(url):
        progress(0.5, desc="Loading from cache...")
        
        cached_knowledge = get_cached_knowledge(url)
        if cached_knowledge and not _knowledge_has_pages(cached_knowledge):
            print("⚠️ Cached knowledge has no pages; crawling again.")
            cached_knowledge = None
        if cached_knowledge:
            stats["cache_hit"] = True
            progress(0.9, desc="Preparing chatbot from cache...")
            
            # Extract name from cached data
            raw_name = cached_knowledge.get("metadata", {}).get("name", "the site")
            stats["pages_scraped"] = cached_knowledge.get("metadata", {}).get("pages_scraped", 0)
            _index_knowledge_best_effort(cached_knowledge, stats, force_reindex=False)
            
            chatbot_context = knowledge_to_chatbot_context(cached_knowledge)
            
            # Build system prompt
            system_prompt = f"""You are an AI assistant for {raw_name} ({url}).

RULES:
1. Answer ONLY from the knowledge base below - never make things up.
2. Search the knowledge carefully before saying "I don't know".
3. For bio questions, check the HOMEPAGE section first.
4. Give partial info if available (e.g., "The site mentions X but not Y...").
5. Keep answers concise and helpful.

=== KNOWLEDGE BASE ===

{chatbot_context[:10000]}

=== END ===
"""
            if start_time is not None:
                stats["tcr_seconds"] = time.time() - start_time
            progress(1.0, desc="Done (from cache)!")
            status_text = build_status_new(100, current_step=4, selected_name=raw_name, 
                                           finished=True, stats=stats, from_cache=True)
            
            msg_update = gr.update(interactive=True, placeholder="Ask anything about the website...")
            send_btn_update = gr.update(interactive=True)
            
            return status_text, system_prompt, raw_name, [], msg_update, send_btn_update, stats
    
    # ===== Step 1: Scrape Website (PRIMARY SOURCE) =====
    progress(0.05, desc="Scraping website...")
    status_text = build_status_new(5, current_step=0, stats=stats)
    
    try:
        if max_pages:
            scraped_data = await _maybe_await(scrape_website(url, max_pages=max_pages))
        else:
            scraped_data = await _maybe_await(scrape_website(url))
        if isinstance(scraped_data.get("crawl_report"), dict):
            stats["crawl_report"] = summarize_crawl_report(scraped_data["crawl_report"])
        stats["pages_scraped"] = scraped_data.get("total_pages", 0)
        errors.extend(scraped_data.get("errors", []))  # Collect scraping errors
        
        if not scraped_data.get("success"):
            print("⚠️ Scraping failed, falling back to web search only...")
            scraped_content = ""
        else:
            scraped_content = format_scraped_content_for_context(scraped_data)
    except Exception as e:
        print(f"❌ Scraping error: {e}")
        scraped_content = ""
        scraped_data = {"pages": [], "total_pages": 0, "success": False}
        errors.append(f"Scraping error: {str(e)[:50]}")
    
    # ===== Step 2: Analyze Content Gaps =====
    progress(0.25, desc="Analyzing content gaps...")
    status_text = build_status_new(25, current_step=1, stats=stats, errors=errors)
    
    search_results = []
    web_search_enabled = web_search_supplement_enabled()
    stats["web_search_enabled"] = web_search_enabled

    if not web_search_enabled:
        # Default: the knowledge base holds only content from the website itself.
        progress(0.45, desc="Using website content only")
        status_text = build_status_new(45, current_step=2, stats=stats, errors=errors)
    elif scraped_content:
        try:
            gap_analysis = await _maybe_await(analyze_content_gaps(scraped_content, url))
            stats["gaps_found"] = len(gap_analysis.gaps_found)
            
            # ===== Step 3: Run Targeted Searches (if gaps exist) =====
            if gap_analysis.has_gaps:
                progress(0.45, desc="Running targeted searches...")
                status_text = build_status_new(45, current_step=2, stats=stats, errors=errors)
                
                search_items = []
                for query in gap_analysis.recommended_searches[:HOW_MANY_SEARCHES]:
                    search_items.append(WebSearchItem(
                        reason=f"Filling gap: {query}",
                        query=f"{url} {query}"
                    ))
                
                # Fallback: if agent surfaced gaps but returned no recommended searches, plan them
                if not search_items:
                    fallback_plan = await _maybe_await(plan_gap_searches(url, scraped_content))
                    search_items = fallback_plan.searches
                
                if search_items:
                    search_plan = WebSearchPlan(has_significant_gaps=True, searches=search_items)
                    search_results = await perform_searches(search_plan)
                    stats["searches_run"] = len(search_results)
                else:
                    print("⚠️ Gaps detected but no searches were generated.")
            else:
                progress(0.45, desc="Content comprehensive, skipping web search")
                status_text = build_status_new(45, current_step=2, stats=stats, errors=errors)
                print("✅ Content is comprehensive, no web search needed!")
        except Exception as e:
            print(f"⚠️ Gap analysis error: {e}")
            errors.append(f"Analysis error: {str(e)[:50]}")
    else:
        # Fallback to web search when scraping fails
        progress(0.45, desc="Fallback: Running web searches...")
        status_text = build_status_new(45, current_step=2, stats=stats, errors=errors)
        
        try:
            search_plan = await _maybe_await(plan_gap_searches(url, ""))
            search_results = await _maybe_await(perform_searches(search_plan))
            stats["searches_run"] = len(search_results)
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            errors.append(f"Search error: {str(e)[:50]}")
    
    # Check if we have any content at all
    if not scraped_content and not search_results:
        crawl_errors = "; ".join(str(error)[:120] for error in (scraped_data.get("errors") or [])[:3])
        error_status = build_error_status("scrape_failed",
            f"Could not extract content from {url}. Try a different URL or check if the site is accessible."
            + (f"\n\nDetails: {crawl_errors}" if crawl_errors else ""))
        return (
            error_status,
            "",
            "the site",
            [],
            gr.update(interactive=False),
            gr.update(interactive=False),
            stats,
        )
    
    # ===== Step 4: Build Knowledge Base =====
    progress(0.70, desc="Building knowledge base...")
    status_text = build_status_new(70, current_step=3, stats=stats, errors=errors)
    
    raw_name = derive_site_name(scraped_data)
    if not raw_name:
        try:
            name_source = scraped_content[:2000] if scraped_content else str(search_results)[:2000]
            raw_name = await _maybe_await(extract_name_from_text(name_source, url))
        except Exception as e:
            print(f"⚠️ Name extraction error: {e}")
            raw_name = ""
    
    if not raw_name:
        try:
            host = urlparse(url).netloc
            raw_name = host.replace("www.", "").split('.')[0].title() or "the site"
        except Exception:
            raw_name = "the site"
    
    knowledge = create_knowledge_json(url, scraped_data, search_results, raw_name)
    
    knowledge_filepath = None
    try:
        knowledge_filepath = save_knowledge_json(knowledge, url)
    except Exception as e:
        print(f"⚠️ Could not save cache: {e}")
        errors.append(f"Cache save failed: {str(e)[:30]}")
    _export_knowledge_best_effort(scraped_data, knowledge, knowledge_filepath, stats)

    _index_knowledge_best_effort(knowledge, stats, force_reindex=True)
    
    # ===== Step 5: Prepare Chatbot =====
    progress(0.90, desc="Preparing chatbot...")
    status_text = build_status_new(90, current_step=4, stats=stats, errors=errors)
    
    chatbot_context = knowledge_to_chatbot_context(knowledge)
    
    # IMPROVED SYSTEM PROMPT - Concise for faster responses
    system_prompt = f"""You are an AI assistant for {raw_name} ({url}).

RULES:
1. Answer ONLY from the knowledge base below - never make things up.
2. Search the knowledge carefully before saying "I don't know".
3. For bio questions, check the HOMEPAGE section first.
4. Give partial info if available (e.g., "The site mentions X but not Y...").
5. Keep answers concise and helpful.

=== KNOWLEDGE BASE ===

{chatbot_context[:10000]}

=== END ===
"""
    
    progress(1.0, desc="Done!")
    if start_time is not None:
        stats["tcr_seconds"] = time.time() - start_time
    status_text = build_status_new(100, current_step=4, selected_name=raw_name, 
                                   finished=True, stats=stats, errors=errors)
    
    # Return empty list for chatbot and update other components
    msg_update = gr.update(interactive=True, placeholder="Ask anything about the website...")
    send_btn_update = gr.update(interactive=True)
    
    # Return empty list directly for chatbot (not gr.update)
    return status_text, system_prompt, raw_name, [], msg_update, send_btn_update, stats


# ============================================================
# CHATBOT FUNCTIONS - Fixed response extraction
# ============================================================

def chat_fn(message, history, system_prompt, name, user=None):
    """Handle chatbot conversation - Gradio 6.x uses dict format"""
    # Ensure history is a list
    if history is None:
        history = []
    
    if not message or not message.strip():
        return "", history

    if not system_prompt:
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please generate a chatbot first! Enter a URL above and click 'Generate Chatbot'."}
        ]

    # Build messages for OpenAI API
    messages = [{"role": "system", "content": system_prompt}]

    # Convert history to OpenAI format
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            # Ensure content is a string (fix for malformed responses)
            content = msg["content"]
            if isinstance(content, list):
                # Handle case where content is a list of dicts like [{'text': '...', 'type': 'text'}]
                content = " ".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            messages.append({"role": msg["role"], "content": str(content)})

    # Add new user message
    messages.append({"role": "user", "content": message})

    # Call OpenAI with error handling
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        
        # Extract answer - handle different response formats
        answer = response.choices[0].message.content
        
        # Ensure answer is a plain string
        if answer is None:
            answer = "I couldn't generate a response. Please try again."
        elif isinstance(answer, list):
            # Handle list format response
            answer = " ".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in answer
            )
        else:
            answer = str(answer)
            
    except Exception as e:
        print(f"❌ Chat error: {e}")
        answer = f"⚠️ Sorry, there was an error generating a response. Please try again.\n\nError: {str(e)[:100]}"

    if ENABLE_METRICS:
        provenance = "primary_plus_secondary" if system_prompt and "SECONDARY SOURCE" in system_prompt else "primary_only"
        user_email = None
        if isinstance(user, dict):
            user_email = user.get("email")
        else:
            user_email = getattr(user, "email", None)
        log_chat_answer(
            question=message,
            answer=answer,
            provenance=provenance,
            user=user_email,
        )

    # Return in Gradio 6.x format
    return "", history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]


async def handle_run_research(url, force_refresh, progress=gr.Progress()):
    """Handle research button click - uses the NEW workflow with caching and error handling"""
    if not url or not url.strip():
        return (
            build_error_status("invalid_url"),
            "",
            "the site",
            [],
            gr.update(interactive=False),
            gr.update(interactive=False),
            {},
        )
    
    # Basic URL validation
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Show cache status
    if not force_refresh and is_cached(url):
        print(f"📂 Cache found for {url}, loading instantly...")
    elif force_refresh and is_cached(url):
        print(f"🔄 Force refresh requested, re-processing {url}...")

    try:
        result = await run_full_research_new(url, force_refresh=force_refresh, progress=progress)
        return result
    except Exception as e:
        print(f"❌ Research error: {e}")
        return (
            build_error_status("api_error", str(e)[:200]),
            "",
            "the site",
            [],
            gr.update(interactive=False),
            gr.update(interactive=False),
            {},
        )


# ============================================================
# GRADIO UI - Gradio 6.x compatible with Caching & Refresh & Error Handling
# ============================================================

with gr.Blocks(title="ChatSMITH - Website to Chatbot") as demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    gr.Markdown("""
    <div class="card">
      <div class="pill">Website to Chatbot</div>
      <h1 style="margin:6px 0 0 0;">🤖 ChatSMITH</h1>
      <p style="margin:4px 0 4px 0; color:#cbd5e1;">Generate a chatbot from a website URL.</p>
      <p style="margin:0; color:#94a3b8;">Enter a URL → Generate → Chat. Cached sites reload instantly.</p>
    </div>
    """)

    user_state = gr.State({"email": "local"})
    system_prompt_state = gr.State("")
    name_state = gr.State("the site")
    stats_state = gr.State({})

    with gr.Column():
        with gr.Row():
            url_in = gr.Textbox(
                label="Website URL",
                placeholder="https://example.com",
                scale=4,
                elem_classes="input-wide",
            )
            force_refresh = gr.Checkbox(
                label="🔄 Force Refresh",
                value=False,
                info="Re-scrape the website even if cached",
            )
            run_btn = gr.Button("🚀 Generate Chatbot", variant="primary", scale=1, elem_classes="primary-btn")

        status_box = gr.Markdown(DEFAULT_STATUS_TEXT)

        gr.Markdown("---")
        gr.Markdown("### 💬 Chat with the website")

        chatbot = gr.Chatbot(label="Chat", height=400, value=[])

        with gr.Row():
            msg = gr.Textbox(
                label="Your question",
                placeholder=CHAT_PLACEHOLDER,
                scale=4,
                interactive=False,
                elem_classes="input-wide",
            )
            send_btn = gr.Button("Send", scale=1, interactive=False)

    run_btn.click(
        fn=handle_run_research,
        inputs=[url_in, force_refresh],
        outputs=[status_box, system_prompt_state, name_state, chatbot, msg, send_btn, stats_state],
    )

    send_btn.click(
        fn=chat_fn,
        inputs=[msg, chatbot, system_prompt_state, name_state, user_state],
        outputs=[msg, chatbot],
    )

    msg.submit(
        fn=chat_fn,
        inputs=[msg, chatbot, system_prompt_state, name_state, user_state],
        outputs=[msg, chatbot],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
