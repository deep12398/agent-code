from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import unified_diff
from typing import Any


JSONType = dict[str, Any]


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    template: str
    required_variables: tuple[str, ...]

    def render(self, **variables: str) -> str:
        missing = [key for key in self.required_variables if key not in variables]
        if missing:
            raise KeyError(f"missing prompt variables: {', '.join(missing)}")
        return self.template.format(**variables)


@dataclass
class PromptRegistry:
    templates: dict[str, list[PromptTemplate]] = field(default_factory=dict)

    def add(self, template: PromptTemplate) -> None:
        versions = self.templates.setdefault(template.name, [])
        if any(item.version == template.version for item in versions):
            raise ValueError(f"duplicate prompt version: {template.name}@{template.version}")
        versions.append(template)

    def get(self, name: str, version: str | None = None) -> PromptTemplate:
        versions = self.templates.get(name, [])
        if not versions:
            raise KeyError(name)
        if version is None:
            return sorted(versions, key=lambda item: item.version)[-1]
        for template in versions:
            if template.version == version:
                return template
        raise KeyError(f"{name}@{version}")

    def diff(self, name: str, before: str, after: str) -> str:
        old = self.get(name, before).template.splitlines(keepends=True)
        new = self.get(name, after).template.splitlines(keepends=True)
        return "".join(unified_diff(old, new, fromfile=before, tofile=after))


def validate_json_schema(data: JSONType, schema: JSONType) -> list[str]:
    errors: list[str] = []
    if schema.get("type") == "object" and not isinstance(data, dict):
        return ["root must be object"]

    required = schema.get("required", [])
    for key in required:
        if key not in data:
            errors.append(f"{key} is required")

    properties = schema.get("properties", {})
    for key, rules in properties.items():
        if key not in data:
            continue
        expected_type = rules.get("type")
        if expected_type and not _matches_type(data[key], expected_type):
            errors.append(f"{key} must be {expected_type}")
        if "enum" in rules and data[key] not in rules["enum"]:
            errors.append(f"{key} must be one of {rules['enum']}")
    return errors


def _matches_type(value: Any, expected_type: str) -> bool:
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return isinstance(value, type_map[expected_type])


def classify_ticket_as_tool_call(text: str) -> JSONType:
    lowered = text.lower()
    category = "general"
    if any(word in lowered for word in ["refund", "invoice", "payment", "reimburse"]):
        category = "finance"
    elif any(word in lowered for word in ["vpn", "laptop", "password", "email"]):
        category = "it"
    elif any(word in lowered for word in ["leave", "salary", "onboarding", "hr"]):
        category = "hr"

    urgency = "high" if any(word in lowered for word in ["blocked", "urgent", "asap", "failed"]) else "normal"
    return {
        "tool_name": "route_ticket",
        "arguments": {"category": category, "urgency": urgency, "summary": text[:120]},
    }


def parse_natural_language_classifier(output: str) -> JSONType:
    category_match = re.search(r"category\s*[:=]\s*(\w+)", output, re.IGNORECASE)
    urgency_match = re.search(r"urgency\s*[:=]\s*(\w+)", output, re.IGNORECASE)
    if not category_match or not urgency_match:
        raise ValueError("cannot parse free-form classifier output")
    return {"category": category_match.group(1).lower(), "urgency": urgency_match.group(1).lower()}


def estimate_cot_cost(question: str, answer: str, reasoning: str = "") -> dict[str, int]:
    from agent_code.llm_call import estimate_tokens

    direct_tokens = estimate_tokens(question) + estimate_tokens(answer)
    cot_tokens = direct_tokens + estimate_tokens(reasoning)
    return {
        "direct_tokens": direct_tokens,
        "cot_tokens": cot_tokens,
        "extra_tokens": cot_tokens - direct_tokens,
    }


def dumps_strict_json(data: JSONType, schema: JSONType) -> str:
    errors = validate_json_schema(data, schema)
    if errors:
        raise ValueError("; ".join(errors))
    return json.dumps(data, ensure_ascii=True, sort_keys=True)
