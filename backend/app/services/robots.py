"""robots.txt parsing (RFC 9309 semantics).

* Rules are taken from the most specific matching user-agent group
  (``chatsmith`` before ``*``); groups for the same agent are merged.
* The longest matching rule wins; ``Allow`` wins a tie.
* ``*`` wildcards and a trailing ``$`` anchor are supported.
* Paths are case-sensitive (the old parser lowercased them).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from urllib.parse import quote, unquote, urlparse

DEFAULT_AGENT_TOKENS = ("chatsmith", "chatsmithbot")


@dataclass
class RobotsRules:
    rules: List[Tuple[bool, str]] = field(default_factory=list)  # (allow, pattern)
    crawl_delay: float | None = None
    sitemaps: List[str] = field(default_factory=list)
    fetched: bool = False

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        path = _normalize_path(path)

        best_length = -1
        allowed = True
        for allow, pattern in self.rules:
            if not pattern:
                continue
            if _pattern_matches(pattern, path):
                length = len(pattern)
                if length > best_length or (length == best_length and allow):
                    best_length = length
                    allowed = allow
        return allowed

    @property
    def disallow_count(self) -> int:
        return sum(1 for allow, pattern in self.rules if not allow and pattern)


def _normalize_path(path: str) -> str:
    # Compare percent-encoded forms consistently (RFC 9309 section 2.2.2).
    return quote(unquote(path), safe="/?=&*$%:@!,;~+-._")


_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _pattern_matches(pattern: str, path: str) -> bool:
    regex = _PATTERN_CACHE.get(pattern)
    if regex is None:
        anchored = pattern.endswith("$")
        body = pattern[:-1] if anchored else pattern
        body = _normalize_path(body)
        expression = "".join(".*" if char == "*" else re.escape(char) for char in body)
        regex = re.compile(expression + ("$" if anchored else ""))
        _PATTERN_CACHE[pattern] = regex
    return regex.match(path) is not None


def parse_robots_txt(text: str, agent_tokens: Tuple[str, ...] = DEFAULT_AGENT_TOKENS) -> RobotsRules:
    """Parse robots.txt text into the rules that apply to our crawler."""
    groups: Dict[str, Dict[str, list]] = {}
    sitemaps: List[str] = []
    current_agents: List[str] = []
    last_was_agent = False

    for raw_line in (text or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "user-agent":
            if not last_was_agent:
                current_agents = []
            current_agents.append(value.lower())
            groups.setdefault(value.lower(), {"rules": [], "delay": []})
            last_was_agent = True
            continue
        if key == "sitemap":
            if value:
                sitemaps.append(value)
            last_was_agent = False
            continue

        last_was_agent = False
        if not current_agents:
            continue
        for agent in current_agents:
            group = groups.setdefault(agent, {"rules": [], "delay": []})
            if key in {"allow", "disallow"}:
                group["rules"].append((key == "allow", value))
            elif key == "crawl-delay":
                try:
                    group["delay"].append(float(value))
                except ValueError:
                    pass

    tokens = {token.lower() for token in agent_tokens}
    selected = None
    for agent_name, group in groups.items():
        product = agent_name.split("/", 1)[0].strip()
        if product and product != "*" and product in tokens:
            selected = group
            break
    if selected is None:
        selected = groups.get("*", {"rules": [], "delay": []})

    delay = max(selected["delay"]) if selected["delay"] else None
    return RobotsRules(rules=list(selected["rules"]), crawl_delay=delay, sitemaps=sitemaps, fetched=True)
