"""Mock embedding model and mock LLM.

DO NOT MODIFY THIS FILE. The grader uses its own copy.

Both functions are deterministic and need no network or API key, so you can
focus on the backend logic instead of model plumbing.
"""

import hashlib
import math
import re

DIM = 256

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "does", "do",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def embed(text: str) -> list[float]:
    """Return a unit-length vector of size DIM (or all zeros for empty text).

    Because vectors are normalized, cosine similarity == dot product.
    """
    vec = [0.0] * DIM
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def generate(question: str, contexts: list[str]) -> str:
    """Mock LLM. Pass the retrieved chunk texts, best match first."""
    if not contexts:
        return "I don't know."
    return f"Based on {len(contexts)} source(s): {contexts[0][:200]}"
