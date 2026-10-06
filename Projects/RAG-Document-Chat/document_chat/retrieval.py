"""Lexical and rank-fusion retrieval helpers."""

import math
import re
from collections import Counter
from typing import Iterable

from document_chat.models import Chunk

METHODS = ("hybrid", "semantic", "keyword")
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25:
    def __init__(self, chunks: list[Chunk]):
        self.terms = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average = sum(self.lengths) / max(len(self.lengths), 1)
        self.df = Counter(term for row in self.terms for term in row)
        self.n = len(chunks)

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        words = set(tokenize(query))
        scores = []
        for i, terms in enumerate(self.terms):
            score = 0.0
            for word in words:
                freq = terms.get(word, 0)
                if not freq:
                    continue
                idf = math.log(1 + (self.n - self.df[word] + 0.5) / (self.df[word] + 0.5))
                denom = freq + 1.2 * (0.25 + 0.75 * self.lengths[i] / max(self.average, 1))
                score += idf * freq * 2.2 / denom
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]


def reciprocal_rank_fusion(rankings: Iterable[Iterable[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return scores
