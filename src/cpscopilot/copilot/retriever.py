"""Deterministic Okapi BM25 retriever over knowledge-base chunks.

BM25 is used instead of dense embeddings for the reference implementation
because it is exact, dependency-free and bit-for-bit reproducible. Any object
with the same ``search`` signature (e.g. a dense retriever whose embedding
model is fitted on the knowledge base only) can be substituted.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from cpscopilot.copilot.knowledge_base import KBChunk

__all__ = ["BM25Retriever", "RetrievedChunk", "STOPWORDS", "tokenize"]

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be been but by can could do does for from had has have if in into is "
    "it its may might must no not of on or our should so such than that the their them then "
    "there these they this those to under up use used using was we were what when where which "
    "while who will with within without would you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stop words."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    chunk: KBChunk


class BM25Retriever:
    """Okapi BM25 (k1, b) with deterministic tie-breaking by chunk id."""

    def __init__(self, chunks: list[KBChunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("retriever needs at least one chunk")
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.retrieval_text())) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        self._idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.by_id = {c.chunk_id: c for c in self.chunks}

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = []
        for tf, length in zip(self._tf, self._len):
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self._avg)
                    s += self._idf[t] * f * (self.k1 + 1) / norm
            scores.append(s)
        return scores

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        return [RetrievedChunk(rank=r + 1, score=float(scores[i]), chunk=self.chunks[i])
                for r, i in enumerate(order[:k]) if scores[i] > 0]
