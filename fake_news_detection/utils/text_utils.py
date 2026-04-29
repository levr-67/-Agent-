"""Utility helpers for text analysis."""

from __future__ import annotations

import re
import string
from typing import List

# ---------------------------------------------------------------------------
# Sensationalism vocabulary (common clickbait / fake-news patterns)
# ---------------------------------------------------------------------------
_SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "explosive", "bombshell", "breaking",
    "exclusive", "urgent", "exposed", "scandal", "outrage", "outrageous",
    "terrifying", "horrifying", "incredible", "amazing", "incredible",
    "miracle", "secret", "hidden", "conspiracy", "hoax", "cover-up",
    "coverup", "deep state", "fake", "fraud", "lie", "lies", "liar",
    "unprecedented", "historic", "emergency", "alert", "warning",
    "must read", "must see", "must share", "viral", "banned", "censored",
}

# ---------------------------------------------------------------------------
# Known low-credibility domains (illustrative sample)
# ---------------------------------------------------------------------------
_LOW_CREDIBILITY_DOMAINS = {
    "infowars.com", "naturalnews.com", "yournewswire.com",
    "beforeitsnews.com", "newspunch.com", "worldnewsdailyreport.com",
    "empirenews.net", "theonion.com",  # satire
    "clickhole.com", "babylonbee.com",  # satire
    "mediamass.net", "huzlers.com",
}

# Known high-credibility domains (illustrative sample)
_HIGH_CREDIBILITY_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "economist.com", "wsj.com", "ft.com", "bloomberg.com",
    "npr.org", "pbs.org", "abc.net.au", "cbc.ca",
    "nature.com", "sciencemag.org", "nejm.org",
}


def count_words(text: str) -> int:
    """Return the number of words in *text*."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Return an approximate sentence count."""
    # Split on sentence-ending punctuation followed by whitespace or EOS
    sentences = re.split(r"[.!?]+(?:\s|$)", text.strip())
    return max(1, len([s for s in sentences if s.strip()]))


def caps_ratio(text: str) -> float:
    """Return the fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def count_sensational_words(text: str) -> int:
    """Count how many sensationalism-indicator words appear in *text*."""
    words = set(re.sub(r"[^\w\s'-]", "", text.lower()).split())
    return sum(1 for w in words if w in _SENSATIONAL_WORDS)


def count_quoted_sources(text: str) -> int:
    """Count the number of quoted passages (" … ") as a proxy for sourcing."""
    return len(re.findall(r'"[^"]{10,}"', text))


def count_numeric_claims(text: str) -> int:
    """Count numeric claims (numbers possibly followed by % or units)."""
    return len(re.findall(r"\b\d+(?:[.,]\d+)?(?:\s*%|\s*percent)?\b", text))


def readability_score(text: str) -> float:
    """Simplified Flesch-Kincaid–style readability (0 = hard, 1 = easy)."""
    words = count_words(text)
    sentences = count_sentences(text)
    syllables = _count_syllables(text)
    if words == 0 or sentences == 0:
        return 0.5
    fk = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    # Normalise to [0, 1]; Flesch typically ranges 0–100
    return max(0.0, min(1.0, fk / 100.0))


def extract_entities(text: str) -> List[str]:
    """Naive named-entity extractor: returns Title-Case multi-word phrases."""
    # Match consecutive capitalised words (proxy for proper nouns)
    matches = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    # Deduplicate while preserving order
    seen: set = set()
    entities = []
    for m in matches:
        if m not in seen and len(m) > 3:
            seen.add(m)
            entities.append(m)
    return entities


def extract_claims(text: str) -> List[str]:
    """Extract declarative sentences that likely contain factual claims."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Heuristic: declarative sentence (not a question) with a number or
        # a proper-noun entity
        if not sent.endswith("?") and (
            re.search(r"\b\d+\b", sent) or re.search(r"\b[A-Z][a-z]+\b", sent)
        ):
            claims.append(sent)
    return claims[:10]  # cap at 10


def source_credibility_score(domain: str) -> float:
    """Return a [0, 1] credibility score for the given domain."""
    d = domain.lower().strip()
    if d in _HIGH_CREDIBILITY_DOMAINS:
        return 1.0
    if d in _LOW_CREDIBILITY_DOMAINS:
        return 0.0
    # Unknown domain: neutral score
    return 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_syllables(text: str) -> int:
    """Very rough syllable counter (counts vowel clusters)."""
    words = re.sub(r"[^a-zA-Z\s]", "", text.lower()).split()
    total = 0
    for word in words:
        count = len(re.findall(r"[aeiou]+", word))
        total += max(1, count)
    return total
