from __future__ import annotations

import time
from dataclasses import dataclass, field

from .rag import InMemoryVectorStore
from .text import ScoredText, cosine, term_counts


@dataclass
class MemoryFact:
    id: str
    user_id: str
    type: str
    key: str
    value: str
    confidence: float = 0.8
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    deprecated: bool = False

    @property
    def text(self) -> str:
        return f"{self.key}: {self.value}"


def extract_facts(user_id: str, utterance: str) -> list[MemoryFact]:
    facts: list[MemoryFact] = []
    if "上海" in utterance:
        facts.append(MemoryFact(f"{user_id}:location", user_id, "semantic", "location", "上海浦东", 0.9, 0.8))
    if "产品经理" in utterance:
        facts.append(MemoryFact(f"{user_id}:occupation", user_id, "semantic", "occupation", "产品经理", 0.9, 0.7))
    if "简洁" in utterance:
        facts.append(MemoryFact(f"{user_id}:style", user_id, "semantic", "response_style", "偏好简洁回答", 0.86, 0.6))
    if "喜欢辣" in utterance:
        facts.append(MemoryFact(f"{user_id}:taste", user_id, "semantic", "taste", "喜欢辣", 0.82, 0.5))
    if "出差" in utterance:
        facts.append(MemoryFact(f"{user_id}:travel:{int(time.time())}", user_id, "episodic", "travel_event", utterance, 0.75, 0.4))
    return facts


class MemoryStore:
    def __init__(self):
        self.facts: dict[str, MemoryFact] = {}

    def add_or_update(self, fact: MemoryFact) -> str:
        existing = self.facts.get(fact.id)
        if existing and existing.value == fact.value:
            existing.confidence = max(existing.confidence, fact.confidence)
            existing.importance = max(existing.importance, fact.importance)
            return "NOOP"
        if existing:
            existing.deprecated = True
            self.facts[f"{existing.id}:old:{int(time.time())}"] = existing
            self.facts[fact.id] = fact
            return "UPDATE"
        self.facts[fact.id] = fact
        return "ADD"

    def delete(self, user_id: str, key: str) -> int:
        count = 0
        for fact in self.facts.values():
            if fact.user_id == user_id and fact.key == key and not fact.deprecated:
                fact.deprecated = True
                count += 1
        return count

    def retrieve(self, user_id: str, query: str, k: int = 5, fact_type: str | None = None) -> list[ScoredText]:
        now = time.time()
        query_vector = term_counts(query)
        scored: list[ScoredText] = []
        for fact in self.facts.values():
            if fact.user_id != user_id or fact.deprecated:
                continue
            if fact_type and fact.type != fact_type:
                continue
            if fact.expires_at and fact.expires_at < now:
                continue
            age_days = (now - fact.created_at) / 86400
            recency = 1 / (1 + age_days)
            semantic = cosine(query_vector, term_counts(fact.text))
            score = semantic + fact.importance * 0.5 + fact.confidence * 0.3 + recency * 0.2
            scored.append(ScoredText(fact.text, score, {"fact_id": fact.id, "type": fact.type, "key": fact.key}))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def profile(self, user_id: str) -> list[MemoryFact]:
        return [
            fact
            for fact in self.facts.values()
            if fact.user_id == user_id and fact.type == "semantic" and not fact.deprecated
        ]


def build_demo_store() -> MemoryStore:
    store = MemoryStore()
    for line in ["我家在上海，刚搬到浦东。", "我是产品经理，喜欢简洁的回答。", "我最近出差去成都。"]:
        for fact in extract_facts("u1", line):
            store.add_or_update(fact)
    return store


class Mem0LikeEngine:
    def __init__(self, store: MemoryStore):
        self.store = store

    def decide_and_apply(self, user_id: str, utterance: str) -> list[tuple[str, MemoryFact]]:
        operations: list[tuple[str, MemoryFact]] = []
        for fact in extract_facts(user_id, utterance):
            operation = self.store.add_or_update(fact)
            operations.append((operation, fact))
        return operations

