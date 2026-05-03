from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in self.messages],
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        return body


@dataclass(frozen=True)
class UsageRecord:
    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_json_line(self) -> str:
        return json.dumps(
            {
                "request_id": self.request_id,
                "model": self.model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            ensure_ascii=True,
        )


@dataclass
class StreamAssembler:
    content: list[str] = field(default_factory=list)
    tool_arguments: dict[str, list[str]] = field(default_factory=dict)
    usage: dict[str, int] | None = None

    def push_event(self, event: dict[str, Any]) -> None:
        if usage := event.get("usage"):
            self.usage = usage

        choices = event.get("choices", [])
        if not choices:
            return

        delta = choices[0].get("delta", {})
        if text := delta.get("content"):
            self.content.append(text)

        for tool_call in delta.get("tool_calls", []):
            call_id = tool_call.get("id") or str(tool_call.get("index", 0))
            function = tool_call.get("function", {})
            arguments = function.get("arguments", "")
            self.tool_arguments.setdefault(call_id, []).append(arguments)

    def text(self) -> str:
        return "".join(self.content)

    def parsed_tool_arguments(self) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for call_id, chunks in self.tool_arguments.items():
            raw = "".join(chunks)
            parsed[call_id] = json.loads(raw) if raw else {}
        return parsed


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    pieces = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    return max(1, int(len(pieces) * 1.25))


def estimate_message_tokens(messages: Iterable[ChatMessage]) -> int:
    return sum(estimate_tokens(message.role) + estimate_tokens(message.content) + 4 for message in messages)


def trim_messages_to_budget(messages: list[ChatMessage], max_prompt_tokens: int) -> list[ChatMessage]:
    if max_prompt_tokens <= 0:
        return []

    system_messages = [message for message in messages if message.role == "system"]
    remaining = [message for message in messages if message.role != "system"]
    selected: list[ChatMessage] = []

    for message in reversed(remaining):
        candidate = system_messages + [message] + list(reversed(selected))
        if estimate_message_tokens(candidate) > max_prompt_tokens:
            continue
        selected.append(message)

    return system_messages + list(reversed(selected))


def append_usage_record(path: str | Path, record: UsageRecord) -> None:
    usage_path = Path(path)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    with usage_path.open("a", encoding="utf-8") as handle:
        handle.write(record.as_json_line() + "\n")


def retry_with_backoff(
    operation: Callable[[], Any],
    *,
    attempts: int = 3,
    base_delay: float = 0.05,
    jitter: float = 0.0,
    retryable: tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except retryable as error:
            last_error = error
            if attempt == attempts - 1:
                break
            delay = base_delay * (2**attempt)
            if jitter:
                delay += random.uniform(0, jitter)
            sleep(delay)
    raise last_error or RuntimeError("operation failed without an exception")


def build_raw_http_request(request: LLMRequest, api_key: str = "$OPENAI_API_KEY") -> str:
    payload = json.dumps(request.payload(), ensure_ascii=True, indent=2)
    return "\n".join(
        [
            "POST /v1/chat/completions HTTP/1.1",
            "Host: api.openai.com",
            f"Authorization: Bearer {api_key}",
            "Content-Type: application/json",
            "",
            payload,
        ]
    )


def sample_next_token(probabilities: dict[str, float], *, temperature: float, seed: int = 7) -> str:
    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    if temperature <= 0:
        return max(probabilities.items(), key=lambda item: item[1])[0]

    adjusted = {token: probability ** (1.0 / temperature) for token, probability in probabilities.items()}
    total = sum(adjusted.values())
    randomizer = random.Random(seed)
    threshold = randomizer.random()
    cursor = 0.0
    for token, value in adjusted.items():
        cursor += value / total
        if cursor >= threshold:
            return token
    return next(reversed(adjusted))
