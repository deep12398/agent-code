from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .text import ScoredText, cosine, summarize_sentences, term_counts, tokenize, unique_preserve_order


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    parent_text: str | None = None


def stable_id(text: str, prefix: str = "id") -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def sample_corpus() -> list[Document]:
    return [
        Document(
            "hr_leave",
            "第三章 假期制度。第一节 年假天数。入职满 1 年至 5 年的员工享有 5 个工作日带薪年假，5 年至 10 年享有 10 个工作日，10 年以上享有 15 个工作日。",
            {"category": "hr", "title": "假期制度"},
        ),
        Document(
            "expense",
            "第二章 报销制度。差旅报销额度根据职级而定。总监及以上 800 元/天，经理级 500 元/天，普通员工 300 元/天。出差天数超过 5 天的，每天额度上浮 20%。",
            {"category": "finance", "title": "差旅报销"},
        ),
        Document(
            "phone",
            "产品规格。本机支持 5G SA/NSA 双模，兼容主流运营商频段。电池容量 5000mAh，支持 67W 快充，典型续航 36 小时。",
            {"category": "product", "title": "手机规格"},
        ),
        Document(
            "remote",
            "办公制度。公司支持远程办公，员工需提前一天在系统提交 WFH 申请，直属经理审批后生效。",
            {"category": "hr", "title": "远程办公"},
        ),
        Document(
            "security",
            "安全制度。所有高危工具调用必须经过人工审批，包含转账、删除数据、修改权限和导出敏感数据。",
            {"category": "security", "title": "工具安全"},
        ),
    ]


def fixed_size_chunks(document: Document, size: int = 80, overlap: int = 12) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    while start < len(document.text):
        end = min(len(document.text), start + size)
        text = document.text[start:end]
        metadata = {**document.metadata, "source_id": document.id, "strategy": "fixed"}
        chunks.append(Chunk(stable_id(document.id + text, "chunk"), text, metadata, parent_text=document.text))
        if end == len(document.text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def sentence_chunks(document: Document) -> list[Chunk]:
    parts = [part.strip() for part in re.split(r"[。；;]\s*", document.text) if part.strip()]
    chunks = []
    for idx, part in enumerate(parts):
        text = part + "。"
        metadata = {**document.metadata, "source_id": document.id, "strategy": "sentence"}
        chunks.append(Chunk(f"{document.id}_s{idx}", text, metadata, parent_text=document.text))
    return chunks


def structure_aware_chunks(document: Document) -> list[Chunk]:
    parts = [part.strip() for part in re.split(r"[。]\s*", document.text) if part.strip()]
    title_path = document.metadata.get("title", "")
    chunks: list[Chunk] = []
    buffer: list[str] = []
    for part in parts:
        if re.search(r"第[一二三四五六七八九十0-9]+[章节]|制度|规格", part):
            title_path = f"{document.metadata.get('title', '')} > {part}".strip(" >")
            continue
        buffer.append(part)
        if len("。".join(buffer)) >= 45:
            text = f"上下文：{title_path}。内容：" + "。".join(buffer) + "。"
            metadata = {**document.metadata, "source_id": document.id, "strategy": "structure", "title_path": title_path}
            chunks.append(Chunk(stable_id(document.id + text, "chunk"), text, metadata, parent_text=document.text))
            buffer.clear()
    if buffer:
        text = f"上下文：{title_path}。内容：" + "。".join(buffer) + "。"
        metadata = {**document.metadata, "source_id": document.id, "strategy": "structure", "title_path": title_path}
        chunks.append(Chunk(stable_id(document.id + text, "chunk"), text, metadata, parent_text=document.text))
    return chunks


def build_chunks(documents: Iterable[Document], strategy: str = "structure") -> list[Chunk]:
    chunker: Callable[[Document], list[Chunk]]
    if strategy == "fixed":
        chunker = fixed_size_chunks
    elif strategy == "sentence":
        chunker = sentence_chunks
    else:
        chunker = structure_aware_chunks
    return [chunk for document in documents for chunk in chunker(document)]


class InMemoryVectorStore:
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks = list(chunks)
        self._vectors = {chunk.id: term_counts(chunk.text) for chunk in self.chunks}

    def search(self, query: str, k: int = 5, filters: dict[str, str] | None = None) -> list[ScoredText]:
        query_vector = term_counts(query)
        candidates = self.chunks
        if filters:
            candidates = [
                chunk
                for chunk in candidates
                if all(chunk.metadata.get(key) == value for key, value in filters.items())
            ]
        scored = [
            ScoredText(chunk.text, cosine(query_vector, self._vectors[chunk.id]), {**chunk.metadata, "chunk_id": chunk.id})
            for chunk in candidates
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]


def query_rewrite(query: str) -> str:
    replacements = {
        "我们公司允许远程办公吗": "公司远程办公政策和申请规定",
        "我去年入职，今年能休几天假": "入职满 1 年的员工的年假天数是多少",
        "休几天假": "年假天数",
        "年假怎么算": "带薪年休假天数计算规则",
        "在家办公": "远程办公 WFH 制度",
        "远程办公吗": "远程办公政策和申请规定",
        "电池续航怎么样": "电池容量 快充 典型续航",
        "支持 5G 吗": "5G SA NSA 双模 支持频段",
    }
    rewritten = query
    for source, target in replacements.items():
        rewritten = rewritten.replace(source, target)
    rewritten = re.sub(r"(?<![我你他她它])我(?!们)", "员工", rewritten)
    return rewritten


def hyde_answer(query: str) -> str:
    if "续航" in query or "电池" in query:
        return "产品规格说明包含电池容量、快充功率、典型续航时间。"
    if "年假" in query or "休假" in query:
        return "员工手册说明入职年限对应的带薪年假天数。"
    if "远程" in query or "在家" in query:
        return "办公制度说明员工远程办公申请和审批流程。"
    return f"{query} 的相关制度、规则、参数和审批流程。"


def multi_queries(query: str) -> list[str]:
    variants = [query, query_rewrite(query), hyde_answer(query)]
    if "远程" in query or "在家" in query:
        variants.extend(["WFH 申请制度", "弹性办公地点规定"])
    if "年假" in query or "休" in query:
        variants.extend(["带薪年休假", "假期制度 年假天数"])
    return unique_preserve_order(variants)


def reciprocal_rank_fusion(result_sets: list[list[ScoredText]], k: int = 60, top_n: int = 5) -> list[ScoredText]:
    scores: dict[str, float] = {}
    payloads: dict[str, ScoredText] = {}
    for results in result_sets:
        for rank, item in enumerate(results, start=1):
            chunk_id = item.metadata.get("chunk_id", item.text)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            payloads[chunk_id] = item
    fused = [
        ScoredText(payloads[chunk_id].text, score, payloads[chunk_id].metadata)
        for chunk_id, score in scores.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)[:top_n]


def bm25_like_search(chunks: Iterable[Chunk], query: str, k: int = 5) -> list[ScoredText]:
    query_tokens = set(tokenize(query))
    scored: list[ScoredText] = []
    for chunk in chunks:
        tokens = tokenize(chunk.text)
        overlap = sum(1 for token in query_tokens if token in tokens)
        score = overlap / max(1, len(query_tokens))
        scored.append(ScoredText(chunk.text, score, {**chunk.metadata, "chunk_id": chunk.id}))
    return sorted(scored, key=lambda item: item.score, reverse=True)[:k]


def rerank(query: str, candidates: Iterable[ScoredText], top_n: int = 3) -> list[ScoredText]:
    query_tokens = set(tokenize(query_rewrite(query)))
    reranked: list[ScoredText] = []
    for item in candidates:
        tokens = set(tokenize(item.text))
        overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
        answer_bonus = 0.2 if any(marker in item.text for marker in ["享有", "支持", "申请", "额度", "必须"]) else 0.0
        reranked.append(ScoredText(item.text, item.score + overlap + answer_bonus, item.metadata))
    return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_n]


def answer_from_context(query: str, contexts: Iterable[ScoredText]) -> str:
    snippets = [summarize_sentences(item.text, 2) for item in contexts if item.text]
    if not snippets:
        return "没有找到足够可靠的资料，建议补充知识库。"
    return f"问题：{query}\n依据：{' '.join(snippets)}"


def context_precision(results: list[ScoredText], expected_source_id: str) -> float:
    if not results:
        return 0.0
    hits = [item for item in results if item.metadata.get("source_id") == expected_source_id]
    return len(hits) / len(results)
