"""Trafilatura cross-check for the structural extractor.

Trafilatura (Barbaresi, 2021) is a strong main-text extractor for article-like
pages but drops listings, tables and page furniture, so it is not used as the
primary extractor. Instead it audits our extraction:

* ``main_text_coverage``: share of Trafilatura's main-text paragraphs that our
  extraction also contains (a per-page extraction-quality signal);
* ``missing``: paragraphs Trafilatura found that we did not, which the
  extractor adds back as a safety net.
"""

import re
from typing import List, Tuple

MIN_PARAGRAPH_CHARS = 40
COMPARE_PREFIX_CHARS = 80


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", (text or "").lower())


def trafilatura_available() -> bool:
    try:
        import trafilatura  # noqa: F401
    except Exception:
        return False
    return True


def main_text_paragraphs(html: str, url: str = "") -> List[str]:
    try:
        import trafilatura
    except Exception:
        return []
    try:
        text = trafilatura.extract(
            html,
            url=url or None,
            favor_recall=True,
            include_comments=False,
            include_tables=False,
            deduplicate=True,
        )
    except Exception:
        return []
    paragraphs = []
    for line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) >= MIN_PARAGRAPH_CHARS:
            paragraphs.append(line)
    return paragraphs


def compare_with_main_text(html: str, url: str, extracted_text: str) -> Tuple[float | None, List[str]]:
    """Return (coverage, missing_paragraphs). Coverage is None when not measurable."""
    paragraphs = main_text_paragraphs(html, url)
    if not paragraphs:
        return None, []
    haystack = _compact(extracted_text)
    missing = []
    for paragraph in paragraphs:
        needle = _compact(paragraph)[:COMPARE_PREFIX_CHARS]
        if needle and needle not in haystack:
            missing.append(paragraph)
    coverage = round(1 - len(missing) / len(paragraphs), 3)
    return coverage, missing
