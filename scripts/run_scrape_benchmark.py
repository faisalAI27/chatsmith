#!/usr/bin/env python3
"""Measure crawl/extraction quality on a fixed set of websites.

For every site in evaluation/scrape_benchmark/sites.json the scraper is run
and we check which "golden facts" (strings known to be on the site) appear
anywhere in the extracted knowledge. This turns "does the scraper work?" into
numbers we can compare across versions:

    python3 scripts/run_scrape_benchmark.py --output-prefix current
    python3 scripts/run_scrape_benchmark.py --repo /path/to/old/checkout --output-prefix legacy
    python3 scripts/run_scrape_benchmark.py --only quotes_javascript,hockey_tables
    python3 scripts/run_scrape_benchmark.py --baseline evaluation/scrape_benchmark/results/legacy_main.json

Outputs <prefix>.json (full detail) and <prefix>.md (summary table) in
evaluation/scrape_benchmark/results/. No OpenAI key is needed: only the
crawler/extractor run (no embeddings, no chat).
"""

import argparse
import asyncio
import importlib
import inspect
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITES = ROOT / "evaluation" / "scrape_benchmark" / "sites.json"
DEFAULT_RESULTS = ROOT / "evaluation" / "scrape_benchmark" / "results"
QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(QUOTE_MAP).lower()
    return re.sub(r"\s+", " ", text).strip()


def page_text(page: dict) -> str:
    """Everything a chatbot could retrieve from one page record."""
    parts = [page.get("title", ""), page.get("description", ""), page.get("content", "")]
    for section in page.get("sections") or []:
        parts.extend([section.get("heading", ""), section.get("content", "")])
    for faq in page.get("faqs") or []:
        parts.extend([faq.get("question", ""), faq.get("answer", "")])
    for table in page.get("tables") or []:
        parts.append(" ".join(table.get("headers") or []))
        parts.extend(" ".join(str(cell) for cell in row) for row in table.get("rows") or [])
    for item in page.get("lists") or []:
        parts.extend(item.get("items") or [])
    return normalize(" \n ".join(str(part) for part in parts if part))


def load_scraper(repo: Path):
    sys.path.insert(0, str(repo))
    module = importlib.import_module("backend.app.services.scrape_pipeline")
    return module


async def run_site(scraper, site: dict) -> dict:
    started = time.monotonic()
    scrape = scraper.scrape_website
    try:
        if "max_pages" in inspect.signature(scrape).parameters:
            result = await scrape(site["url"], max_pages=site.get("max_pages"))
        else:  # legacy scraper: fixed 10-page limit
            result = await scrape(site["url"])
        error = ""
    except Exception as exc:  # report, do not abort the benchmark
        result, error = {"pages": [], "errors": [str(exc)]}, f"{type(exc).__name__}: {exc}"
    duration = time.monotonic() - started

    pages = [page for page in result.get("pages", []) if isinstance(page, dict)]
    real_pages = [page for page in pages if page.get("page_type") != "site_wide"]
    corpus = "\n".join(page_text(page) for page in pages)
    found = [fact for fact in site["facts"] if normalize(fact) in corpus]
    missing = [fact for fact in site["facts"] if fact not in found]
    methods: dict = {}
    for page in real_pages:
        methods[page.get("extraction_method", "?")] = methods.get(page.get("extraction_method", "?"), 0) + 1
    report = result.get("crawl_report") or {}
    return {
        "name": site["name"],
        "url": site["url"],
        "kind": site.get("kind", ""),
        "pages": len(real_pages),
        "documents": sum(1 for page in real_pages if page.get("page_type") == "document"),
        "facts_total": len(site["facts"]),
        "facts_found": len(found),
        "fact_recall": round(len(found) / len(site["facts"]), 3) if site["facts"] else None,
        "missing_facts": missing,
        "extraction_methods": methods,
        "chars_extracted": sum(len(page.get("content") or "") for page in pages),
        "empty_pages": sum(1 for page in real_pages if not (page.get("content") or "").strip()),
        "duplicates_skipped": report.get("duplicate_count", 0),
        "failed": len(report.get("failed") or []) or len(result.get("errors") or []),
        "boilerplate_blocks_removed": (report.get("boilerplate") or {}).get("blocks_removed", 0),
        "duration_seconds": round(duration, 1),
        "error": error,
    }


def comparison_markdown(rows: list[dict], baseline: list[dict], baseline_label: str) -> str:
    by_name = {row["name"]: row for row in baseline}
    lines = [
        f"## Compared with {baseline_label}",
        "",
        "| Site | Pages (before → after) | Fact recall (before → after) | Time s (before → after) |",
        "| --- | ---: | ---: | ---: |",
    ]
    before_found = before_total = after_found = after_total = 0
    for row in rows:
        old = by_name.get(row["name"])
        if old is None:
            continue
        before_found += old["facts_found"]
        before_total += old["facts_total"]
        after_found += row["facts_found"]
        after_total += row["facts_total"]
        lines.append(
            f"| {row['name']} | {old['pages']} → {row['pages']} | "
            f"{old['facts_found']}/{old['facts_total']} → {row['facts_found']}/{row['facts_total']} | "
            f"{old['duration_seconds']} → {row['duration_seconds']} |"
        )
    lines.append(
        f"| **Total** | | **{before_found}/{before_total} ({before_found / max(1, before_total):.0%}) → "
        f"{after_found}/{after_total} ({after_found / max(1, after_total):.0%})** | |"
    )
    return "\n".join(lines) + "\n"


def to_markdown(label: str, rows: list[dict]) -> str:
    total_found = sum(row["facts_found"] for row in rows)
    total_facts = sum(row["facts_total"] for row in rows)
    lines = [
        f"# Scrape benchmark: {label}",
        "",
        f"Run: {datetime.now().isoformat(timespec='seconds')}  ",
        f"Overall fact recall: **{total_found}/{total_facts} ({total_found / max(1, total_facts):.0%})**",
        "",
        "| Site | Kind | Pages | Fact recall | Methods | Dups skipped | Failed | Time (s) |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        methods = ", ".join(f"{name}×{count}" for name, count in sorted(row["extraction_methods"].items()))
        lines.append(
            f"| {row['name']} | {row['kind']} | {row['pages']} | {row['facts_found']}/{row['facts_total']} | "
            f"{methods or '-'} | {row['duplicates_skipped']} | {row['failed']} | {row['duration_seconds']} |"
        )
    lines.append("")
    for row in rows:
        if row["missing_facts"] or row["error"]:
            lines.append(f"- **{row['name']}** missing: {', '.join(repr(fact) for fact in row['missing_facts'])} {row['error']}")
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=ROOT, help="checkout whose scraper to benchmark")
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--only", default="", help="comma-separated site names")
    parser.add_argument("--output-prefix", default="current")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--baseline", type=Path, help="earlier results .json to compare against")
    args = parser.parse_args()

    sites = json.loads(args.sites.read_text(encoding="utf-8"))["sites"]
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        sites = [site for site in sites if site["name"] in wanted]

    scraper = load_scraper(args.repo.resolve())
    rows = []
    for site in sites:
        print(f"\n### {site['name']}: {site['url']}", flush=True)
        row = await run_site(scraper, site)
        rows.append(row)
        print(f"    -> pages={row['pages']} facts={row['facts_found']}/{row['facts_total']} time={row['duration_seconds']}s", flush=True)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    label = f"{args.output_prefix} ({args.repo.resolve().name})"
    (args.results_dir / f"{args.output_prefix}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = to_markdown(label, rows)
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        markdown += "\n" + comparison_markdown(rows, baseline, args.baseline.stem)
    (args.results_dir / f"{args.output_prefix}.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)


if __name__ == "__main__":
    asyncio.run(main())
