"""Text fingerprints used to detect exact and near-duplicate pages."""

import hashlib
import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)
SIMHASH_BITS = 64


def normalize_for_fingerprint(text: str) -> str:
    """Lowercase and collapse text so cosmetic differences do not change hashes."""
    return " ".join(_WORD_RE.findall((text or "").lower()))


def content_hash(text: str) -> str:
    """Return a stable hash of normalized text (exact-duplicate detection)."""
    normalized = normalize_for_fingerprint(text)
    if not normalized:
        return ""
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def simhash(text: str, shingle_size: int = 3) -> int:
    """Return a 64-bit SimHash over word shingles (near-duplicate detection).

    Pages that differ only by a few words (for example the same product in two
    colours) produce hashes that differ in only a few bits.
    """
    words = normalize_for_fingerprint(text).split()
    if not words:
        return 0
    if len(words) < shingle_size:
        shingles = [" ".join(words)]
    else:
        shingles = [" ".join(words[i:i + shingle_size]) for i in range(len(words) - shingle_size + 1)]

    weights = [0] * SIMHASH_BITS
    for shingle in shingles:
        value = int.from_bytes(hashlib.md5(shingle.encode("utf-8")).digest()[:8], "big")
        for bit in range(SIMHASH_BITS):
            weights[bit] += 1 if value >> bit & 1 else -1

    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            fingerprint |= 1 << bit
    return fingerprint


def simhash_hex(text: str) -> str:
    value = simhash(text)
    return f"{value:016x}" if value else ""


def hamming_distance(first: int | str, second: int | str) -> int:
    a = int(first, 16) if isinstance(first, str) else int(first)
    b = int(second, 16) if isinstance(second, str) else int(second)
    return bin(a ^ b).count("1")
