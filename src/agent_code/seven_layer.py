from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureLayer:
    id: str
    name: str
    responsibility: str
    common_failures: tuple[str, ...]
    example_tools: tuple[str, ...]


LAYERS = (
    ArchitectureLayer("L1", "Model gateway", "Model routing, retries, rate limits, usage logging", ("429", "timeout", "bill spike"), ("LiteLLM", "Portkey")),
    ArchitectureLayer("L2", "Memory and retrieval", "Conversation memory, RAG, token budget, permissions-aware context", ("wrong context", "privacy leak", "missing recall"), ("pgvector", "Redis", "RAGAS")),
    ArchitectureLayer("L3", "Tool execution", "Tool schema, idempotency, sandboxing, error translation", ("wrong tool", "duplicate side effect", "tool timeout"), ("MCP", "Temporal")),
    ArchitectureLayer("L4", "Orchestration", "Agent loop, workflows, multi-agent coordination, state machines", ("infinite loop", "task drift", "deadlock"), ("LangGraph", "CrewAI")),
    ArchitectureLayer("L5", "Interaction", "Streaming, UI state, human-in-the-loop, client protocol", ("broken stream", "stale UI", "missing approval"), ("SSE", "WebSocket")),
    ArchitectureLayer("L6", "Security and governance", "Prompt injection defense, RBAC, audit, PII controls", ("data exfiltration", "privilege escalation", "unsafe action"), ("Guardrails", "OPA")),
    ArchitectureLayer("L7", "Evaluation and feedback", "Regression sets, traces, online feedback, rollback gates", ("silent regression", "bad prompt rollout", "unknown root cause"), ("Langfuse", "LangSmith", "Harness")),
)


FRAMEWORK_MAPPING = {
    "OpenAI SDK": ("L1",),
    "LangChain AgentExecutor": ("L3", "L4"),
    "LangGraph": ("L4",),
    "CrewAI": ("L4",),
    "MCP": ("L3",),
    "RAGAS": ("L2", "L7"),
    "Langfuse": ("L7",),
    "Guardrails": ("L6",),
}


SYMPTOM_LOOKUP = {
    "rate limit": ("L1", "Check retry policy, backoff, queueing, and fallback model routing."),
    "wrong answer": ("L2", "Check retrieved context, memory selection, and prompt assembly trace."),
    "wrong tool": ("L3", "Check tool descriptions, parameter schema, and negative examples."),
    "infinite loop": ("L4", "Check max iterations, repeated-action detection, and state transitions."),
    "stream cut": ("L5", "Check SSE buffering, client reconnect, and partial JSON assembly."),
    "private data": ("L6", "Check RBAC filters, PII redaction, and indirect prompt injection."),
    "prompt changed": ("L7", "Run regression evaluation and compare traces before rollout."),
}


def diagnose(symptom: str) -> list[dict[str, str]]:
    lowered = symptom.lower()
    matches = []
    for keyword, (layer_id, action) in SYMPTOM_LOOKUP.items():
        if keyword in lowered:
            layer = next(item for item in LAYERS if item.id == layer_id)
            matches.append({"layer": layer.id, "name": layer.name, "action": action})
    return matches or [{"layer": "unknown", "name": "Needs triage", "action": "Start from trace, then map evidence to L1-L7."}]


def printable_fault_table() -> str:
    lines = ["Layer | Failure | First check", "--- | --- | ---"]
    for layer in LAYERS:
        for failure in layer.common_failures:
            lines.append(f"{layer.id} {layer.name} | {failure} | {layer.responsibility}")
    return "\n".join(lines)


def sample_customer_service_breakdown() -> dict[str, str]:
    return {
        "L1": "Use one model for routing and another for final answer; persist request id and token usage.",
        "L2": "Window memory plus permission-filtered RAG over HR/IT/finance documents.",
        "L3": "Ticket, refund, and knowledge lookup tools with idempotency keys.",
        "L4": "Route by intent, execute tools, stop on budget or repeated actions.",
        "L5": "SSE response with reconnect and human approval for side effects.",
        "L6": "RBAC, PII masking, prompt-injection scan, full audit log.",
        "L7": "Golden question set, trace comparison, bad-case labeling, rollout gates.",
    }
