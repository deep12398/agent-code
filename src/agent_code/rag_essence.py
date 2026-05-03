from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    reason: str


def chunk_text(text: str, *, chunk_size: int = 80, overlap: int = 10) -> list[str]:
    words = text.split()
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    chunks = []
    cursor = 0
    while cursor < len(words):
        chunks.append(" ".join(words[cursor : cursor + chunk_size]))
        cursor += chunk_size - overlap
    return chunks


class NaiveRAGRetriever:
    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)

    def search(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        results = [
            SearchResult(document, _cosine(_tf(query), _tf(document.text)), "term-vector")
            for document in self.documents
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return [result for result in results[:limit] if result.score > 0]

    def answer(self, query: str) -> dict[str, object]:
        contexts = self.search(query)
        context_text = "\n".join(result.document.text for result in contexts)
        return {
            "query": query,
            "contexts": [result.document.id for result in contexts],
            "prompt": f"Answer using only this context:\n{context_text}\n\nQuestion: {query}",
        }


class HybridSearch:
    def __init__(self, documents: Iterable[Document], *, keyword_weight: float = 0.45) -> None:
        self.documents = list(documents)
        self.keyword_weight = keyword_weight

    def search(self, query: str, *, limit: int = 3) -> list[SearchResult]:
        query_terms = set(_terms(query))
        query_vector = _tf(query)
        results = []
        for document in self.documents:
            keyword = _keyword_score(query_terms, set(_terms(document.text)))
            semantic = _cosine(query_vector, _tf(document.text))
            score = self.keyword_weight * keyword + (1 - self.keyword_weight) * semantic
            results.append(SearchResult(document, score, "hybrid"))
        results.sort(key=lambda item: item.score, reverse=True)
        return [result for result in results[:limit] if result.score > 0]


class Reranker:
    def rerank(self, query: str, results: list[SearchResult], *, limit: int = 3) -> list[SearchResult]:
        query_terms = set(_terms(query))
        reranked = []
        for result in results:
            phrase_bonus = 0.25 if query.lower() in result.document.text.lower() else 0.0
            coverage_bonus = _keyword_score(query_terms, set(_terms(result.document.text))) * 0.5
            reranked.append(
                SearchResult(
                    result.document,
                    result.score + phrase_bonus + coverage_bonus,
                    f"{result.reason}+rerank",
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:limit]


class RAGEvaluator:
    def evaluate(self, query: str, answer: str, contexts: list[Document]) -> dict[str, float]:
        context_terms = set(_terms(" ".join(document.text for document in contexts)))
        answer_terms = set(_terms(answer))
        query_terms = set(_terms(query))
        grounded = len(answer_terms & context_terms) / max(1, len(answer_terms))
        relevance = len(query_terms & context_terms) / max(1, len(query_terms))
        return {
            "faithfulness": round(grounded, 3),
            "context_relevance": round(relevance, 3),
            "hallucination_rate": round(1.0 - grounded, 3),
        }


def build_finetune_pairs(documents: Iterable[Document]) -> list[dict[str, str]]:
    pairs = []
    for document in documents:
        title = (document.metadata or {}).get("title", document.id)
        first_sentence = re.split(r"[.!?]", document.text.strip())[0]
        pairs.append({"query": f"What does {title} say?", "positive": first_sentence, "document_id": document.id})
    return pairs


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf(text: str) -> dict[str, float]:
    vector: dict[str, float] = {}
    for term in _terms(text):
        vector[term] = vector.get(term, 0.0) + 1.0
    return vector


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm)


def _keyword_score(query_terms: set[str], doc_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & doc_terms) / len(query_terms)
