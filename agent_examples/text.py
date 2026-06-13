from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def term_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_score(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def summarize_sentences(text: str, max_sentences: int = 2) -> str:
    parts = [part.strip() for part in re.split(r"[。！？!?]\s*", text) if part.strip()]
    return "。".join(parts[:max_sentences]) + ("。" if parts else "")


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


@dataclass(frozen=True)
class ScoredText:
    text: str
    score: float
    metadata: dict[str, str]

