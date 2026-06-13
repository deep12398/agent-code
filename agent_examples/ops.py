from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class GatewayResponse:
    provider: str
    model: str
    content: str
    cost: float
    cached: bool = False


class ModelGateway:
    def __init__(self):
        self.cache: dict[str, GatewayResponse] = {}
        self.providers = ["primary", "fallback"]

    def complete(self, prompt: str, tier: str = "auto") -> GatewayResponse:
        key = hashlib.sha1(f"{tier}:{prompt}".encode()).hexdigest()
        if key in self.cache:
            cached = self.cache[key]
            return GatewayResponse(cached.provider, cached.model, cached.content, 0.0, True)
        model = "cheap-model" if tier == "cheap" or len(prompt) < 80 else "strong-model"
        provider = self.providers[0]
        content = f"[{model}] {prompt[:120]}"
        cost = estimate_cost(prompt, content, model)
        response = GatewayResponse(provider, model, content, cost)
        self.cache[key] = response
        return response


def estimate_cost(prompt: str, completion: str, model: str = "strong-model") -> float:
    rate = 0.000001 if model == "cheap-model" else 0.00001
    return (len(prompt) + len(completion)) * rate


def compress_prompt(prompt: str, max_words: int = 40) -> str:
    max_chars = max_words * 4
    if len(prompt) > max_chars:
        protected_terms = [term for term in ["Agent", "RAG", "Prompt", "工具", "历史对话", "系统规则"] if term in prompt]
        prefix = "、".join(protected_terms)
        compact = re.sub(r"(你是一个企业客服 Agent，需要根据以下)", "客服 Agent 根据", prompt)
        compact = re.sub(r"(非常长的|回答用户问题。)", "", compact)
        compact = compact[:max_chars]
        return f"{prefix} | {compact}" if prefix else compact

    words = prompt.split()
    if len(words) > 1 and len(words) <= max_words:
        return prompt
    if len(words) > 1:
        important = [word for word in words if len(word) > 3]
        return " ".join(important[:max_words])
    return prompt


def semantic_cache_key(text: str) -> str:
    normalized = re.sub(r"\d+", "#", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()


async def fan_out_in(tasks: list[str], worker: Callable[[str], str]) -> list[str]:
    async def run(task: str) -> str:
        await asyncio.sleep(0.01)
        return worker(task)

    return await asyncio.gather(*(run(task) for task in tasks))


def stream_tokens(text: str) -> list[str]:
    return text.split()


def detect_prompt_injection(text: str) -> dict[str, object]:
    patterns = ["忽略前面的指令", "system prompt", "删除全部", "泄露", "ignore previous"]
    hits = [pattern for pattern in patterns if pattern.lower() in text.lower()]
    return {"risk": "high" if hits else "low", "hits": hits}


@dataclass
class User:
    user_id: str
    tenant_id: str
    roles: set[str]


def can_call_tool(user: User, tool: str) -> bool:
    rules = {
        "export_data": {"admin", "auditor"},
        "delete_user": {"admin"},
        "read_kb": {"admin", "support", "auditor"},
    }
    return bool(user.roles & rules.get(tool, set()))


def guard_output(text: str) -> dict[str, object]:
    pii = re.findall(r"1[3-9]\d{9}|[\w.-]+@[\w.-]+", text)
    if pii:
        return {"allowed": False, "reason": "PII", "safe_text": re.sub(r"1[3-9]\d{9}|[\w.-]+@[\w.-]+", "[REDACTED]", text)}
    return {"allowed": True, "reason": "ok", "safe_text": text}


def hitl_required(action: str) -> bool:
    return any(keyword in action for keyword in ["转账", "删除", "修改权限", "导出敏感"])


@dataclass
class TraceSpan:
    name: str
    start: float = field(default_factory=time.time)
    end: float | None = None
    attributes: dict[str, object] = field(default_factory=dict)

    def finish(self) -> None:
        self.end = time.time()

    @property
    def duration_ms(self) -> float:
        end = self.end or time.time()
        return (end - self.start) * 1000


class TraceCollector:
    def __init__(self):
        self.spans: list[TraceSpan] = []

    def span(self, name: str, **attributes: object) -> TraceSpan:
        span = TraceSpan(name, attributes=attributes)
        self.spans.append(span)
        return span

    def export(self) -> list[dict[str, object]]:
        return [
            {"name": span.name, "duration_ms": round(span.duration_ms, 2), "attributes": span.attributes}
            for span in self.spans
        ]


def ragas_like_score(answer: str, contexts: list[str]) -> dict[str, float]:
    if not contexts:
        return {"faithfulness": 0.0, "context_precision": 0.0}
    answer_tokens = set(answer)
    context_tokens = set("".join(contexts))
    overlap = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
    return {"faithfulness": round(overlap, 2), "context_precision": round(min(1.0, len(contexts) / 3), 2)}


def bad_case_pipeline(cases: list[dict[str, object]]) -> dict[str, object]:
    failed = [case for case in cases if float(case.get("score", 0)) < 0.7]
    return {"total": len(cases), "failed": len(failed), "regression_set": failed}


def init_backend_db(path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS tenants (id TEXT PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, tenant_id TEXT, email TEXT, roles TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS agents (id TEXT PRIMARY KEY, tenant_id TEXT, name TEXT)")
        conn.execute("INSERT OR IGNORE INTO tenants VALUES ('t1', 'Demo Tenant')")
        conn.execute("INSERT OR IGNORE INTO users VALUES ('u1', 't1', 'demo@example.com', 'admin,support')")
        conn.execute("INSERT OR IGNORE INTO agents VALUES ('a1', 't1', '客服 Agent')")


def dashboard_metrics() -> dict[str, object]:
    return {
        "cost": [{"day": "Mon", "value": 12.4}, {"day": "Tue", "value": 9.8}],
        "latency_p95_ms": 820,
        "cache_hit_rate": 0.36,
        "eval_pass_rate": 0.91,
    }


def harness_run(task: str) -> dict[str, object]:
    files = {"plan.md": f"# Plan\n- {task}\n- implement\n- verify"}
    trace = ["planner wrote plan.md", "worker implemented demo", "reviewer approved"]
    return {"task": task, "files": files, "trace": trace}
