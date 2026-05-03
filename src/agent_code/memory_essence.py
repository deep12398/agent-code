from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from agent_code.llm_call import ChatMessage, estimate_message_tokens, estimate_tokens


@dataclass(frozen=True)
class MemoryItem:
    role: str
    content: str

    def as_chat_message(self) -> ChatMessage:
        return ChatMessage(self.role, self.content)


@dataclass
class BufferMemory:
    messages: list[MemoryItem] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(MemoryItem(role, content))

    def context(self) -> list[ChatMessage]:
        return [item.as_chat_message() for item in self.messages]


@dataclass
class WindowMemory(BufferMemory):
    window_size: int = 6

    def context(self) -> list[ChatMessage]:
        return [item.as_chat_message() for item in self.messages[-self.window_size :]]


@dataclass
class SummaryMemory(BufferMemory):
    summary: str = ""
    keep_last: int = 4

    def refresh_summary(self) -> str:
        older = self.messages[: -self.keep_last] if self.keep_last else self.messages
        facts = []
        for item in older:
            sentence = item.content.split(".")[0].strip()
            if sentence:
                facts.append(f"{item.role}: {sentence}")
        self.summary = " | ".join(facts[-8:])
        return self.summary

    def context(self) -> list[ChatMessage]:
        self.refresh_summary()
        messages: list[ChatMessage] = []
        if self.summary:
            messages.append(ChatMessage("system", f"Conversation summary: {self.summary}"))
        messages.extend(item.as_chat_message() for item in self.messages[-self.keep_last :])
        return messages


@dataclass
class VectorMemory:
    records: list[tuple[str, str]] = field(default_factory=list)

    def add_fact(self, user_id: str, fact: str) -> None:
        self.records.append((user_id, fact))

    def search(self, user_id: str, query: str, *, limit: int = 3) -> list[str]:
        query_vector = _term_frequency(query)
        scored: list[tuple[float, str]] = []
        for owner, fact in self.records:
            if owner != user_id:
                continue
            scored.append((_cosine(query_vector, _term_frequency(fact)), fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [fact for score, fact in scored[:limit] if score > 0]


@dataclass
class LayeredMemory:
    user_id: str
    short_term: WindowMemory = field(default_factory=lambda: WindowMemory(window_size=4))
    summary: SummaryMemory = field(default_factory=lambda: SummaryMemory(keep_last=2))
    long_term: VectorMemory = field(default_factory=VectorMemory)

    def add_turn(self, user: str, assistant: str) -> None:
        self.short_term.add("user", user)
        self.short_term.add("assistant", assistant)
        self.summary.add("user", user)
        self.summary.add("assistant", assistant)

    def remember_fact(self, fact: str) -> None:
        self.long_term.add_fact(self.user_id, fact)

    def build_context(self, query: str, *, max_tokens: int = 300) -> list[ChatMessage]:
        long_term_facts = self.long_term.search(self.user_id, query)
        base = []
        if long_term_facts:
            base.append(ChatMessage("system", "Relevant long-term facts: " + " | ".join(long_term_facts)))
        if summary_text := self.summary.refresh_summary():
            base.append(ChatMessage("system", f"Conversation summary: {summary_text}"))
        base.extend(self.short_term.context())
        return fit_messages_to_budget(base, max_tokens)


def fit_messages_to_budget(messages: list[ChatMessage], max_tokens: int) -> list[ChatMessage]:
    selected: list[ChatMessage] = []
    for message in reversed(messages):
        candidate = [message] + selected
        if estimate_message_tokens(candidate) <= max_tokens:
            selected = candidate
    return selected


def budget_breakdown(messages: Iterable[ChatMessage]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for message in messages:
        breakdown[message.role] = breakdown.get(message.role, 0) + estimate_tokens(message.content)
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def extract_candidate_facts(text: str) -> list[str]:
    facts = []
    for sentence in re.split(r"[.!?]\s+", text):
        lowered = sentence.lower()
        if any(marker in lowered for marker in ["i prefer", "my ", "i work", "i live", "remember"]):
            facts.append(sentence.strip())
    return [fact for fact in facts if fact]


def _term_frequency(text: str) -> dict[str, float]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    vector: dict[str, float] = {}
    for word in words:
        vector[word] = vector.get(word, 0.0) + 1.0
    return vector


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm)
